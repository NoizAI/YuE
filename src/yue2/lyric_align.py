"""Forced-align submitted lyrics to generated audio and report per-line timings.

Lines follow the MV's `lyricLines()` rules (blank lines and [Section] tags are
dropped), so the timings line up 1:1 with what the player displays. The primary
aligner is MMS CTC (wav2vec2) with a `<star>` token between lines so intros and
breaks are not swallowed by a lyric line. Timings that fail sanity checks fall
back to stable-ts on Whisper, and are rejected if that fails too.

Heavy dependencies are imported in `LyricAligner.load()` so the API process and
CPU tests do not need torch.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata

log = logging.getLogger("yue2.lyric_align")

SR = 16_000
MMS_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
# uroman language codes. The CTC vocabulary is roman letters, so every line is romanized.
ISO = {"zh": "cmn", "ja": "jpn", "ko": "kor", "en": "eng"}
# A non-final line longer than this swallowed an instrumental break.
MAX_MIDDLE_S = 15.0
MIN_LINE_S = 0.12

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def phrase_lines(line: str) -> list[str]:
    """Keep in sync with `phraseLines` in the MV frontend's lyrics splitter."""
    parts = [p for p in re.split(r"\s+", line) if p]
    if len(parts) < 2 or sum(1 for p in parts if CJK.search(p)) < 2:
        return [line]
    out: list[str] = []
    latin: list[str] = []

    def flush() -> None:
        if latin:
            out.append(" ".join(latin))
            latin.clear()

    for part in parts:
        if CJK.search(part):
            flush()
            out.append(part)
        else:
            latin.append(part)
    flush()
    return out


def lyric_lines(lyrics: str) -> list[str]:
    text = lyrics
    flat = "\n" not in text
    if flat:
        text = re.sub(r"\s*(\[[^\]]+\])\s*", r"\n\1\n", text)
        text = re.sub(r"([.!?。！？])\s+", r"\1\n", text)
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or re.fullmatch(r"\[.*\]", line):
            continue
        out.extend(phrase_lines(line) if flat else [line])
    return out


def guess_language(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def usable(segs: list[list[float]], duration: float) -> bool:
    if duration <= 0:
        return False
    prev = -1.0
    for i, (start, end) in enumerate(segs):
        span = end - start
        last = i == len(segs) - 1
        if start < 0 or end > duration + 0.5 or span < MIN_LINE_S:
            return False
        if not last and span > MAX_MIDDLE_S:
            return False
        if start < prev - 0.05:
            return False
        prev = start
    # A collapsed alignment pins every line to the intro. Real songs sing past the middle.
    if duration >= 30 and len(segs) >= 4 and segs[-1][0] < duration * 0.45:
        return False
    return True


class LyricAligner:
    """MMS CTC resident on `device`; the Whisper fallback loads on CPU on first use."""

    def __init__(self, *, mms_model=MMS_MODEL, whisper_model="base", device="cuda",
                 local_files_only=False):
        self.mms_model, self.whisper_model = mms_model, whisper_model
        self.device, self.local_files_only = device, local_files_only
        self.mms = self.tokenizer = self.uroman = self.whisper = None
        self.lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer
        from uroman import Uroman

        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mms_model, local_files_only=self.local_files_only)
        self.mms = AutoModelForCTC.from_pretrained(
            self.mms_model, dtype=dtype, local_files_only=self.local_files_only,
        ).to(self.device).eval()
        self.uroman = Uroman()

    def close(self):
        self.mms = self.tokenizer = self.whisper = None

    def align(self, audio_path, lyrics):
        """Return `([{"text", "start", "end"}, ...], engine)` or raise ValueError."""
        lines = lyric_lines(lyrics)
        if not lines:
            raise ValueError("no lyric lines")
        lang = guess_language(lyrics)
        with self.lock:
            wav = self._load_mono(audio_path)
            duration = wav.shape[0] / SR
            engine = "mms"
            try:
                segs = self._align_mms(wav, lines, lang)
                if not usable(segs, duration):
                    raise ValueError("mms timings failed checks")
            except Exception as error:
                log.warning("MMS alignment fell back to Whisper: %s", error)
                engine = "whisper"
                try:
                    segs = self._align_whisper(audio_path, lines, lang)
                except Exception as fallback_error:
                    log.warning("Whisper alignment failed: %s", fallback_error)
                    raise ValueError("MMS and Whisper alignment both failed") from None
                if not usable(segs, duration):
                    raise ValueError("alignment failed checks") from None
        return [{"text": line, "start": start, "end": end}
                for line, (start, end) in zip(lines, segs)], engine

    @staticmethod
    def _load_mono(path):
        import soundfile
        import torch
        import torchaudio

        audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.mean(axis=1))
        if rate != SR:
            wav = torchaudio.functional.resample(wav, rate, SR)
        if wav.numel() == 0:
            raise ValueError("empty audio")
        return wav

    def _romanize(self, text, lang):
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\w\s']+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        rom = self.uroman.romanize_string(text, lcode=ISO.get(lang, "eng")).lower()
        rom = re.sub(r"[^a-z']+", "", rom)
        return " ".join(rom)

    def _emissions(self, wav):
        """Log-probs over the CTC vocabulary plus a final column for the star token.

        The star column is all zeros, so it outscores every real label and absorbs the
        frames between lines (intros, breaks) instead of stretching a lyric line across them.
        Long files are scored in 30s windows with 2s of context, then the context is trimmed.
        """
        import torch

        model = self.mms
        ratio = model.config.inputs_to_logits_ratio
        window, context = 30 * SR, 2 * SR
        window -= window % ratio
        context -= context % ratio
        if wav.shape[0] <= window:
            chunks, ctx, pad = wav.unsqueeze(0), 0, 0
        else:
            pad = (window - wav.shape[0] % window) % window
            padded = torch.nn.functional.pad(wav, (context, context + pad))
            chunks = padded.unfold(0, window + 2 * context, window)
            ctx = context
        parts = []
        with torch.inference_mode():
            for i in range(chunks.shape[0]):
                chunk = chunks[i : i + 1].to(self.device, dtype=model.dtype)
                parts.append(model(chunk).logits[0].float().cpu())
        ctx_frames, win_frames = ctx // ratio, window // ratio
        frames = [p[ctx_frames : ctx_frames + win_frames] if ctx else p for p in parts]
        emis = torch.log_softmax(torch.cat(frames, 0), dim=-1)
        if pad:
            emis = emis[: -(pad // ratio)]
        return torch.cat([emis, torch.zeros(emis.shape[0], 1)], 1)

    def _align_mms(self, wav, lines, lang):
        import torch
        import torchaudio

        vocab = {k.lower(): v for k, v in self.tokenizer.get_vocab().items()}
        vocab["<star>"] = len(vocab)
        blank = vocab.get("<blank>", self.tokenizer.pad_token_id)
        pieces: list[str] = []
        groups: list[list[int]] = []
        for line in lines:
            chars = [ch for ch in self._romanize(line, lang).split(" ") if ch in vocab]
            if not chars:
                raise ValueError(f"no tokens for {line!r}")
            pieces.append("<star>")
            groups.append([len(pieces) + i for i in range(len(chars))])
            pieces.extend(chars)
        targets = torch.tensor([[vocab[p] for p in pieces]], dtype=torch.int32)
        path, _ = torchaudio.functional.forced_align(
            self._emissions(wav).unsqueeze(0), targets, blank=blank)
        labels = path[0].tolist()
        runs: list[list[int]] = []
        prev = blank
        for i, lab in enumerate(labels):
            if lab == blank:
                prev = blank
                continue
            if lab == prev:
                runs[-1][2] = i
            else:
                runs.append([lab, i, i])
            prev = lab
        if len(runs) != len(pieces):
            raise ValueError(f"{len(runs)} aligned tokens for {len(pieces)} targets")
        stride = self.mms.config.inputs_to_logits_ratio / SR
        return [[round(runs[g[0]][1] * stride, 2), round((runs[g[-1]][2] + 1) * stride, 2)]
                for g in groups]

    def _align_whisper(self, audio_path, lines, lang):
        if self.whisper is None:
            import stable_whisper
            self.whisper = stable_whisper.load_model(self.whisper_model, device="cpu")
        result = self.whisper.align(str(audio_path), "\n".join(lines), language=lang,
                                    original_split=True, verbose=None)
        segs = [[round(s.start, 2), round(s.end, 2)] for s in result.segments]
        if len(segs) != len(lines):
            raise ValueError(f"{len(segs)} segments for {len(lines)} lines")
        return segs
