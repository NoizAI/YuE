"""Line splitting, sanity checks and engine fallback; model execution is GPU-validated separately."""
from types import SimpleNamespace

import pytest

from yue2.lyric_align import LyricAligner, SR, guess_language, lyric_lines, usable


def test_lyric_lines_drop_sections_and_blank_lines():
    lyrics = "[Verse]\n晨光落在窗边\n\n  我们走向新的一天  \n[Chorus]\nLet the song stay\n[Outro]\n"
    assert lyric_lines(lyrics) == ["晨光落在窗边", "我们走向新的一天", "Let the song stay"]


def test_flat_lyrics_split_on_sections_sentences_and_cjk_phrases():
    assert lyric_lines("[Verse] Hello there. How are you? [Chorus] 晨光 落在 窗边 oh yeah") == [
        "Hello there.", "How are you?", "晨光", "落在", "窗边", "oh yeah"]


@pytest.mark.parametrize("text,lang", [("こんにちは", "ja"), ("안녕하세요", "ko"),
                                       ("晨光 hello", "zh"), ("hello", "en")])
def test_guess_language(text, lang):
    assert guess_language(text) == lang


@pytest.mark.parametrize("segs,duration,ok", [
    ([[1, 2], [2.5, 4]], 10, True),
    ([[1, 2], [2.5, 4]], 0, False),
    ([[-0.1, 2]], 10, False),
    ([[1, 1.05]], 10, False),
    ([[1, 20], [21, 22]], 30, False),
    ([[1, 2], [5, 30.2]], 30, True),
    ([[5, 6], [1, 2]], 10, False),
    ([[1, 2], [2, 3], [3, 4], [4, 5]], 60, False),
])
def test_usable(segs, duration, ok):
    assert usable(segs, duration) is ok


def aligner(monkeypatch, *, mms, whisper):
    instance = LyricAligner(device="cpu")
    monkeypatch.setattr(instance, "_load_mono", lambda path: SimpleNamespace(shape=(10 * SR,)))
    monkeypatch.setattr(instance, "_align_mms", mms)
    monkeypatch.setattr(instance, "_align_whisper", whisper)
    return instance


def test_align_prefers_mms_and_labels_each_line(monkeypatch):
    def whisper(*args):
        raise AssertionError("Whisper must not run when MMS timings pass checks")

    instance = aligner(monkeypatch, mms=lambda wav, lines, lang: [[1.0, 2.0], [2.5, 4.0]],
                       whisper=whisper)
    timings, engine = instance.align("song.flac", "[Verse]\n晨光落在窗边\n我们走向新的一天")
    assert engine == "mms"
    assert timings == [{"text": "晨光落在窗边", "start": 1.0, "end": 2.0},
                       {"text": "我们走向新的一天", "start": 2.5, "end": 4.0}]


def test_align_falls_back_to_whisper_when_mms_fails_checks(monkeypatch):
    seen = {}

    def whisper(path, lines, lang):
        seen.update(path=path, lines=lines, lang=lang)
        return [[1.0, 2.0]]

    instance = aligner(monkeypatch, mms=lambda *args: [[1.0, 1.01]], whisper=whisper)
    timings, engine = instance.align("song.flac", "Hello world\n")
    assert engine == "whisper" and timings[0]["start"] == 1.0
    assert seen == {"path": "song.flac", "lines": ["Hello world"], "lang": "en"}


def test_align_rejects_when_both_engines_fail(monkeypatch):
    def mms(*args):
        raise RuntimeError("CUDA out of memory")

    def whisper(*args):
        raise RuntimeError("model download failed")

    with pytest.raises(ValueError, match="both failed"):
        aligner(monkeypatch, mms=mms, whisper=whisper).align("song.flac", "Hello")
    with pytest.raises(ValueError, match="failed checks"):
        aligner(monkeypatch, mms=mms, whisper=lambda *args: [[20.0, 21.0]]).align("song.flac", "Hello")
    with pytest.raises(ValueError, match="no lyric lines"):
        aligner(monkeypatch, mms=mms, whisper=whisper).align("song.flac", "[Intro]\n\n[Outro]")
