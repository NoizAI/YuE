"""FastAPI + durable queue + one exclusive inference thread per GPU.

The API process imports no torch until its inference worker starts. A test can
inject a tiny pipeline without downloading weights or requiring a GPU.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import gc
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .service_store import IdempotencyConflict, JobStore, QueueFull

log = logging.getLogger("yue2.service")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=16, repr=False)
    data_dir: Path = Path("outputs/service")
    model: str = "m-a-p/YuE2-3B"
    vae: str = "m-a-p/YuE2-Vae"
    revision: str | None = None
    vae_revision: str | None = None
    device: str = "cuda"
    backend: Literal["torch", "torch-eager", "vllm"] = "torch"
    resident_models: bool = True
    memory_budget_gib: float = Field(default=30, gt=2, allow_inf_nan=False)
    quantization: Literal["none", "fp8"] = "none"
    ode_steps: int = Field(default=32, ge=1, le=64)
    vae_core_frames: int = Field(default=1024, ge=64, le=4096)
    max_pending: int = Field(default=16, ge=1, le=10000)
    task_timeout_seconds: float = Field(default=1200, gt=0, allow_inf_nan=False)
    warmup: bool = True
    local_files_only: bool = False

    @model_validator(mode="after")
    def compatible_backend(self):
        if self.backend == "vllm" and self.resident_models:
            raise ValueError("vLLM requires YUE2_RESIDENT_MODELS=false")
        if self.backend == "vllm" and self.quantization != "none":
            raise ValueError("The current vLLM adapter does not support FP8")
        return self

    @classmethod
    def from_env(cls):
        return cls(**{name: os.environ[f"YUE2_{name.upper()}"] for name in cls.model_fields
                      if f"YUE2_{name.upper()}" in os.environ})


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    style: str = Field(min_length=1, max_length=2000)
    lyrics: str = Field(min_length=1, max_length=16000)
    cot: Literal["full", "melody", "off"] = "full"
    seed: int = Field(default=831001, ge=0, lt=2**63, strict=True)
    abc: str | None = Field(default=None, min_length=1, max_length=64000)
    cfg_scale: float | None = Field(default=None, ge=0, le=20)

    @field_validator("style", "lyrics", "abc")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @model_validator(mode="after")
    def score_mode(self):
        if self.abc is not None and self.cot == "off":
            raise ValueError("ABC requires cot=full or melody")
        return self


def build_pipeline(settings):
    from .pipeline import YuE2Pipeline
    from .protocol import GenerationConfig
    return YuE2Pipeline.from_pretrained(
        settings.model, vae=settings.vae, revision=settings.revision, vae_revision=settings.vae_revision,
        device=settings.device, backend=settings.backend, resident_models=settings.resident_models,
        memory_budget_gib=settings.memory_budget_gib, quantization=settings.quantization,
        vae_core_frames=settings.vae_core_frames, local_files_only=settings.local_files_only,
        generation_config=GenerationConfig(ode_steps=settings.ode_steps), progress=False)


class JobWorker:
    def __init__(self, settings, factory):
        self.settings, self.factory = settings, factory
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(self.root / "jobs.sqlite3")
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.active = None
        self.ready = False
        self.startup_error = False
        self.pipeline = None
        self.thread = None
        self.lock_file = None

    def start(self):
        self.lock_file = (self.root / "worker.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("This data directory already has a worker. Use one Uvicorn worker per GPU.") from None
        try:
            self.store.recover()
            self.thread = threading.Thread(target=self._run, name="yue2-gpu", daemon=True)
            self.thread.start()
        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise

    def stop(self):
        self.ready = False
        self.stop_event.set()
        with self.lock:
            if self.active is not None:
                self.active[1].set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join()  # cooperative cancellation; supervisor may force-stop a stuck CUDA driver
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    def cancel(self, job_id):
        job = self.store.cancel(job_id)
        with self.lock:
            if self.active is not None and self.active[0] == job_id:
                self.active[1].set()
        self.wake.set()
        return job

    def _load(self):
        self.ready = False
        self.pipeline = self.factory(self.settings)
        self.pipeline.preload()
        if self.settings.warmup and not self.stop_event.is_set():
            # Short disposable generation warms libraries, not a cached production song.
            self.pipeline(style="English piano pop", lyrics="[Verse]\nA new day begins",
                          cot="full", seed=42, abc_sampling={"min_tokens": 16, "max_tokens": 32},
                          semantic_sampling={"min_tokens": 16, "max_tokens": 32},
                          cancelled=self.stop_event.is_set)
        self.ready = not self.stop_event.is_set()

    def _run(self):
        try:
            self._load()
            while not self.stop_event.is_set():
                self.wake.clear()
                claimed = self.store.claim()
                if claimed is None:
                    self.wake.wait(0.5)
                    continue
                job, request = claimed
                self._execute(job, request)
        except Exception:
            if not self.stop_event.is_set():
                self.startup_error = True
                log.exception("Inference worker unavailable")
        finally:
            self.ready = False
            if self.pipeline is not None:
                self.pipeline.close()

    def _execute(self, job, request):
        job_id, cancel_event = job["id"], threading.Event()
        with self.lock:
            self.active = (job_id, cancel_event)
        if self.store.get(job_id)["cancel_requested"]:
            cancel_event.set()
        deadline = time.monotonic() + self.settings.task_timeout_seconds
        counts, last_report = {"abc": 0, "semantic": 0}, 0.0

        def cancelled():
            return cancel_event.is_set() or self.stop_event.is_set() or time.monotonic() >= deadline

        def stage(name):
            if cancelled():
                raise InterruptedError("Cancelled at stage boundary")
            self.store.progress(job_id, stage=name, tokens=dict(counts))

        def token(phase, value):
            nonlocal last_report
            counts[phase] = counts.get(phase, 0) + 1
            now = time.monotonic()
            if now - last_report >= 1:
                self.store.progress(job_id, tokens=dict(counts))
                last_report = now

        rebuild = False
        try:
            if cancelled():
                raise InterruptedError("Cancelled before generation")
            result = self.pipeline(**request, cancelled=cancelled, on_token=token, on_stage=stage)
            stage("saving")
            output = self.root / "artifacts" / job_id
            output.mkdir(parents=True, exist_ok=False)
            saved = result.save_artifacts(output)
            if cancelled():
                raise InterruptedError("Cancelled after saving")
            truncated = any(saved["truncated"].values())
            self.store.finish(job_id, "truncated" if truncated else "succeeded", result={
                "audio_url": f"/v1/jobs/{job_id}/audio", "score_url": f"/v1/jobs/{job_id}/score" if result.abc else None,
                "audio_seconds": saved["audio_seconds"], "sample_rate": saved["sample_rate"],
                "truncated": saved["truncated"], "timing": saved["timing"],
                "configuration": result.config,
            })
        except InterruptedError:
            if time.monotonic() >= deadline and not cancel_event.is_set() and not self.stop_event.is_set():
                self.store.finish(job_id, "failed", error={"code": "timeout", "message": "Generation exceeded its execution deadline."})
            else:
                self.store.finish(job_id, "cancelled")
        except Exception as error:
            log.exception("Generation failed: job=%s", job_id)
            invalid = isinstance(error, (ValueError, TypeError))
            self.store.finish(job_id, "failed", error={
                "code": "invalid_generation" if invalid else "inference_failed",
                "message": "Input could not be generated; check score and context length." if invalid else "Inference failed; see worker logs with this job ID.",
            })
            # Release exception traceback frames before loading another model:
            # they may retain the tensors that caused an out-of-memory failure.
            self.ready = False
            rebuild = True
        finally:
            with self.lock:
                self.active = None
        if rebuild:
            self.pipeline.close()
            self.pipeline = None
            gc.collect()
            if not self.stop_event.is_set():
                self._load()


class BodyLimit:
    """Bound JSON buffering before validation, including chunked requests."""
    def __init__(self, app, limit=256 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"detail": "Request body exceeds 256 KiB"}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(settings=None, pipeline_factory=None):
    settings = settings or Settings.from_env()
    worker = JobWorker(settings, pipeline_factory or build_pipeline)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        try:
            yield
        finally:
            await asyncio.to_thread(worker.stop)

    app = FastAPI(title="Noiz YuE2", version="0.1.0", lifespan=lifespan)
    app.state.worker = worker
    app.add_middleware(BodyLimit)
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    def get_job(job_id):
        job = worker.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if not worker.ready:
            return JSONResponse({"status": "failed" if worker.startup_error else "loading"}, status_code=503)
        return {"status": "ready"}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def submit(request: GenerateRequest, idempotency_key: str | None = Header(default=None, min_length=1, max_length=128)):
        if not worker.ready or worker.stop_event.is_set():
            raise HTTPException(503, "Inference worker is not ready", headers={"Retry-After": "5"})
        try:
            job, created = worker.store.submit(request.model_dump(), settings.max_pending, idempotency_key)
        except QueueFull:
            raise HTTPException(429, "Queue is full", headers={"Retry-After": "5"}) from None
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency-Key was already used with different input") from None
        worker.wake.set()
        return JSONResponse(job, status_code=202 if created else 200, headers={"Location": f"/v1/jobs/{job['id']}"})

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def status(job_id: str):
        return get_job(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel(job_id: str):
        job = worker.cancel(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def artifact(job_id, name, media):
        job = get_job(job_id)
        if job["status"] not in {"succeeded", "truncated"}:
            raise HTTPException(409, "Artifact is not available for this job state")
        path = worker.root / "artifacts" / job["id"] / name
        if not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type=media, filename=f"{job['id']}-{name}")

    @app.get("/v1/jobs/{job_id}/audio", dependencies=[Depends(authorize)])
    def audio(job_id: str):
        return artifact(job_id, "audio.flac", "audio/flac")

    @app.get("/v1/jobs/{job_id}/score", dependencies=[Depends(authorize)])
    def score(job_id: str):
        return artifact(job_id, "score.abc", "text/plain; charset=utf-8")

    return app


def main():
    import uvicorn
    uvicorn.run("yue2.service:create_app", factory=True,
                host=os.environ.get("YUE2_HOST", "127.0.0.1"),
                port=int(os.environ.get("YUE2_PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
