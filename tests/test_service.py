"""Exercise HTTP/queue lifecycle without model downloads; GPU tests are separate."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import weakref
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from yue2.service import GenerateRequest, JobWorker, Settings, create_app
from yue2.service_store import JobStore, QueueFull

KEY = "test-only-api-key-123456789"
HEADERS = {"Authorization": f"Bearer {KEY}"}
REQUEST = {"style": "piano pop", "lyrics": "[Verse]\nAn original song", "seed": 42}


class FakePipeline:
    def __init__(self, *, blocked=False, truncated=False, failure=False):
        self.started, self.release = threading.Event(), threading.Event()
        if not blocked:
            self.release.set()
        self.truncated, self.failure = truncated, failure
        self.calls, self.preloaded, self.closed = 0, False, False

    def preload(self):
        self.preloaded = True

    def __call__(self, *, cancelled, on_stage=None, on_token=None, **kwargs):
        self.calls += 1
        self.started.set()
        if on_stage:
            on_stage("planning")
        if on_token:
            on_token("abc", 1)
        while not self.release.wait(.005):
            if cancelled():
                raise InterruptedError()
        if self.failure:
            raise RuntimeError("private model path /not-for-http-response")
        if on_stage:
            on_stage("decode")

        def save(output):
            Path(output, "audio.flac").write_bytes(b"fake-audio-for-transport-test")
            Path(output, "score.abc").write_text("X:1\nK:C\nCDEF|")
            return {"truncated": {"abc": False, "semantic": self.truncated},
                    "audio_seconds": 1, "sample_rate": 48000, "timing": {"e2e_seconds": .01}}
        return SimpleNamespace(abc="score", config={"resident_models": True}, save_artifacts=save)

    def close(self):
        self.closed = True


def wait_for(check, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(.005)
    pytest.fail("Timed out waiting for worker state")


@contextmanager
def service(tmp_path, fake=None, **options):
    fake = fake or FakePipeline()
    app = create_app(Settings(api_key=KEY, data_dir=tmp_path, warmup=False, **options), lambda settings: fake)
    with TestClient(app, headers=HEADERS) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        yield client, fake, app


def submit(client, **kwargs):
    response = client.post("/v1/jobs", json={**REQUEST, **kwargs})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def terminal(client, job_id):
    def check():
        value = client.get(f"/v1/jobs/{job_id}").json()
        return value if value["status"] in {"succeeded", "truncated", "failed", "cancelled"} else None
    return wait_for(check)


def test_http_success_auth_artifacts_and_reuse(tmp_path):
    with service(tmp_path) as (client, pipe, app):
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/jobs", json=REQUEST, headers={"Authorization": "Bearer wrong"}).status_code == 401
        first = submit(client)
        result = terminal(client, first)
        assert result["status"] == "succeeded"
        assert result["result"]["sample_rate"] == 48000
        assert result["tokens"]["abc"] == 1
        assert client.get(result["result"]["audio_url"]).content == b"fake-audio-for-transport-test"
        assert client.get(result["result"]["score_url"]).text.startswith("X:1")
        assert client.get(result["result"]["audio_url"], headers={"Authorization": ""}).status_code == 401
        second = submit(client, seed=43)
        assert terminal(client, second)["status"] == "succeeded"
        assert pipe.calls == 2 and pipe.preloaded and not pipe.closed
        assert client.get("/v1/jobs/unknown").status_code == 404
        assert client.post("/v1/jobs/unknown/cancel").status_code == 404
        assert client.get("/openapi.json").status_code == 200
    assert pipe.closed


def test_bounded_admission_idempotency_and_cancellation(tmp_path):
    fake = FakePipeline(blocked=True)
    with service(tmp_path, fake, max_pending=2) as (client, _, app):
        first_response = client.post("/v1/jobs", json=REQUEST, headers={"Idempotency-Key": "retry-1"})
        first = first_response.json()["id"]
        assert fake.started.wait(2)
        second = submit(client, seed=43)
        assert client.post("/v1/jobs", json={**REQUEST, "seed": 44}).status_code == 429
        duplicate = client.post("/v1/jobs", json=REQUEST, headers={"Idempotency-Key": "retry-1"})
        assert duplicate.status_code == 200 and duplicate.json()["id"] == first
        assert client.post("/v1/jobs", json={**REQUEST, "seed": 45}, headers={"Idempotency-Key": "retry-1"}).status_code == 409
        assert client.get(f"/v1/jobs/{first}/audio").status_code == 409
        assert client.post(f"/v1/jobs/{second}/cancel").json()["status"] == "cancelled"
        client.post(f"/v1/jobs/{first}/cancel")
        assert terminal(client, first)["status"] == "cancelled"
        fake.release.set()
        third = submit(client, seed=46)
        assert terminal(client, third)["status"] == "succeeded"
        assert fake.calls == 2  # queued cancelled work never runs


def test_timeout_then_next_job(tmp_path):
    fake = FakePipeline(blocked=True)
    with service(tmp_path, fake, task_timeout_seconds=.05) as (client, _, app):
        job = terminal(client, submit(client))
        assert job["status"] == "failed" and job["error"]["code"] == "timeout"
        fake.release.set()
        assert terminal(client, submit(client))["status"] == "succeeded"


def test_truncated_output_is_explicit_and_downloadable(tmp_path):
    with service(tmp_path, FakePipeline(truncated=True)) as (client, _, app):
        job = terminal(client, submit(client))
        assert job["status"] == "truncated" and job["result"]["truncated"]["semantic"]
        assert client.get(job["result"]["audio_url"]).status_code == 200


def test_restart_recovers_running_but_preserves_queued(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    active, _ = store.submit(REQUEST, 3)
    store.claim()
    queued, _ = store.submit({**REQUEST, "seed": 43}, 3)
    with service(tmp_path) as (client, fake, app):
        recovered = client.get(f"/v1/jobs/{active['id']}").json()
        assert recovered["status"] == "failed" and recovered["error"]["code"] == "worker_interrupted"
        assert terminal(client, queued["id"])["status"] == "succeeded"
        assert fake.calls == 1


def test_exclusive_worker_lock(tmp_path):
    with service(tmp_path):
        other = create_app(Settings(api_key=KEY, data_dir=tmp_path, warmup=False), lambda settings: FakePipeline())
        with pytest.raises(RuntimeError, match="already has a worker"):
            with TestClient(other):
                pass
    with service(tmp_path):
        pass  # lock released on graceful shutdown


def test_initialization_failure_readiness(tmp_path):
    def fail(settings):
        raise RuntimeError("no GPU")
    app = create_app(Settings(api_key=KEY, data_dir=tmp_path), fail)
    with TestClient(app, headers=HEADERS) as client:
        wait_for(lambda: app.state.worker.startup_error)
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/jobs", json=REQUEST).status_code == 503


def test_generation_failure_is_sanitized_and_pipeline_rebuilt(tmp_path):
    pipes = [FakePipeline(failure=True), FakePipeline()]
    factories = iter(pipes)
    app = create_app(Settings(api_key=KEY, data_dir=tmp_path, warmup=False), lambda settings: next(factories))
    with TestClient(app, headers=HEADERS) as client:
        wait_for(lambda: app.state.worker.ready)
        job = terminal(client, submit(client))
        assert job["status"] == "failed" and job["error"]["code"] == "inference_failed"
        assert "private" not in str(job)
        wait_for(lambda: app.state.worker.ready and pipes[0].closed)
        assert terminal(client, submit(client))["status"] == "succeeded"


def test_rebuild_does_not_retain_failed_inference_frames(tmp_path):
    references = []
    class Allocation:
        pass
    class FailingPipeline(FakePipeline):
        def __call__(self, **kwargs):
            tensor_placeholder = Allocation()
            references.append(weakref.ref(tensor_placeholder))
            raise RuntimeError("simulated allocation failure")
    calls = 0
    def factory(settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FailingPipeline()
        # Logging handlers can retain tracebacks externally; disable capture in
        # this test so it measures only the worker's own frame lifetime.
        assert references[0]() is None
        return FakePipeline()
    app = create_app(Settings(api_key=KEY, data_dir=tmp_path, warmup=False), factory)
    from unittest.mock import patch
    with patch("yue2.service.log.exception"), TestClient(app, headers=HEADERS) as client:
        wait_for(lambda: app.state.worker.ready)
        assert terminal(client, submit(client))["status"] == "failed"
        wait_for(lambda: calls == 2 and app.state.worker.ready)
        assert terminal(client, submit(client))["status"] == "succeeded"


def test_shutdown_cancels_active_and_leaves_queue_for_restart(tmp_path):
    with service(tmp_path, FakePipeline(blocked=True)) as (client, fake, app):
        first = submit(client)
        assert fake.started.wait(2)
        queued = submit(client, seed=43)
    assert app.state.worker.store.get(first)["status"] == "cancelled"
    assert app.state.worker.store.get(queued)["status"] == "queued"


@pytest.mark.parametrize("change", [{"seed": True}, {"seed": 2**63}, {"lyrics": " "},
                                    {"style": "a" * 2001}, {"cot": "off", "abc": "X:1"},
                                    {"unknown": 1}, {"cfg_scale": float("nan")}])
def test_request_validation(change):
    with pytest.raises(ValidationError):
        GenerateRequest(**{**REQUEST, **change})


def test_body_limit_before_parsing(tmp_path):
    with service(tmp_path) as (client, _, app):
        assert client.post("/v1/jobs", content=b" " * (256 * 1024 + 1)).status_code == 413
        assert client.post("/v1/jobs", json={**REQUEST, "abc": "", "cot": "full"}).status_code == 422


def test_store_atomic_capacity_under_parallel_submissions(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    def submit_one(seed):
        try:
            store.submit({**REQUEST, "seed": seed}, 3)
            return True
        except QueueFull:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(submit_one, range(20))) == 3


def test_claim_many_is_atomic_and_ordered(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    jobs = [store.submit({**REQUEST, "seed": seed}, 4)[0] for seed in (4, 3, 2)]
    claimed = store.claim_many(2)
    assert [job["id"] for job, _ in claimed] == [jobs[0]["id"], jobs[1]["id"]]
    assert all(store.get(job["id"])["status"] == "running" for job, _ in claimed)
    assert store.get(jobs[2]["id"])["status"] == "queued"


def test_worker_batches_vllm_ar_and_serializes_render(tmp_path):
    class ParallelPipeline:
        def __init__(self):
            self.barrier = threading.Barrier(2)
            self.lock = threading.Lock()
            self.ar_active = self.ar_peak = self.render_active = self.render_peak = 0

        def parallel_ar_eligible(self, request):
            return True

        def generate_ar(self, *, seed, on_stage, on_token, **kwargs):
            on_stage("planning")
            with self.lock:
                self.ar_active += 1
                self.ar_peak = max(self.ar_peak, self.ar_active)
            self.barrier.wait(timeout=2)
            on_token("abc", seed)
            with self.lock:
                self.ar_active -= 1
            return seed

        def render_ar(self, seed, *, on_stage, **kwargs):
            on_stage("synthesis")
            with self.lock:
                self.render_active += 1
                self.render_peak = max(self.render_peak, self.render_active)
            time.sleep(.01)
            with self.lock:
                self.render_active -= 1
            def save(output):
                Path(output, "audio.flac").write_bytes(b"audio")
                return {"truncated": {"abc": False, "semantic": False},
                        "audio_seconds": 1, "sample_rate": 48000, "timing": {"seed": seed}}
            return SimpleNamespace(abc=None, config={"backend": "vllm"}, save_artifacts=save)

    settings = Settings(api_key=KEY, data_dir=tmp_path, warmup=False, ar_concurrency=2,
                        vllm_max_num_seqs=2, nar_batch_size=2)
    pipe = ParallelPipeline()
    worker = JobWorker(settings, lambda _: pipe)
    worker.pipeline = pipe
    submitted = [worker.store.submit({**REQUEST, "seed": seed}, 2)[0] for seed in (41, 42)]
    assert worker._execute_claimed(worker.store.claim_many(2)) is False
    assert pipe.ar_peak == 2 and pipe.render_peak == 1
    assert all(worker.store.get(job["id"])["status"] == "succeeded" for job in submitted)


def test_worker_batches_nar_in_fifo_windows_and_keeps_vae_serial(tmp_path):
    class BatchedPipeline:
        def __init__(self):
            self.events = []
            self.decode_active = self.decode_peak = 0

        def parallel_ar_eligible(self, request):
            return True

        def generate_ar(self, *, seed, **kwargs):
            return seed

        def nar_batch_admission(self, results):
            self.events.append(("admit", list(results)))
            return {"allowed": len(results) <= 2}

        def generate_nar_batch(self, results, **kwargs):
            self.events.append(("nar", list(results)))
            return [("latent", seed) for seed in results]

        def render_nar(self, nar_result, **kwargs):
            seed = nar_result[1]
            self.events.append(("decode", seed))
            self.decode_active += 1
            self.decode_peak = max(self.decode_peak, self.decode_active)
            self.decode_active -= 1

            def save(output):
                self.events.append(("save", seed))
                Path(output, "audio.flac").write_bytes(b"audio")
                return {"truncated": {"abc": False, "semantic": False},
                        "audio_seconds": 1, "sample_rate": 48000, "timing": {"seed": seed}}
            return SimpleNamespace(abc=None, config={"nar_batch_size": 2}, save_artifacts=save)

        def render_ar(self, seed, **kwargs):
            raise AssertionError(f"Unexpected serial NAR fallback for {seed}")

    settings = Settings(api_key=KEY, data_dir=tmp_path, warmup=False)
    pipe = BatchedPipeline()
    worker = JobWorker(settings, lambda _: pipe)
    worker.pipeline = pipe
    submitted = [worker.store.submit({**REQUEST, "seed": seed}, 4)[0]
                 for seed in (41, 42, 43, 44)]
    assert worker._execute_claimed(worker.store.claim_many(4)) is False
    assert [event for event in pipe.events if event[0] == "nar"] == [
        ("nar", [41, 42]), ("nar", [43, 44])]
    assert [event[1] for event in pipe.events if event[0] == "save"] == [41, 42, 43, 44]
    assert pipe.decode_peak == 1
    assert all(worker.store.get(job["id"])["status"] == "succeeded" for job in submitted)


def test_worker_retries_runtime_nar_oom_as_two_fifo_pairs(tmp_path):
    class RuntimeFallbackPipeline:
        def __init__(self):
            self.calls = []

        def parallel_ar_eligible(self, request):
            return True

        def generate_ar(self, *, seed, **kwargs):
            return seed

        def nar_batch_admission(self, results):
            return {"allowed": True}

        def generate_nar_batch(self, results, **kwargs):
            self.calls.append(list(results))
            if len(results) == 4:
                raise MemoryError("simulated runtime pressure")
            return list(results)

        def render_nar(self, seed, **kwargs):
            def save(output):
                Path(output, "audio.flac").write_bytes(b"audio")
                return {"truncated": {"abc": False, "semantic": False},
                        "audio_seconds": 1, "sample_rate": 48000, "timing": {"seed": seed}}
            return SimpleNamespace(abc=None, config={}, save_artifacts=save)

        def render_ar(self, seed, **kwargs):
            raise AssertionError(f"Unexpected single-row fallback for {seed}")

    settings = Settings(api_key=KEY, data_dir=tmp_path, warmup=False)
    pipe = RuntimeFallbackPipeline()
    worker = JobWorker(settings, lambda _: pipe)
    worker.pipeline = pipe
    jobs = [worker.store.submit({**REQUEST, "seed": seed}, 4)[0] for seed in range(4)]
    assert worker._execute_claimed(worker.store.claim_many(4)) is False
    assert pipe.calls == [[0, 1, 2, 3], [0, 1], [2, 3]]
    assert all(worker.store.get(job["id"])["status"] == "succeeded" for job in jobs)


def test_worker_keeps_nar_row_failure_independent(tmp_path):
    class RowFailurePipeline:
        def parallel_ar_eligible(self, request):
            return True

        def generate_ar(self, *, seed, **kwargs):
            return seed

        def nar_batch_admission(self, results):
            return {"allowed": True}

        def generate_nar_batch(self, results, **kwargs):
            return [ValueError("bad first row"), results[1]]

        def render_nar(self, seed, **kwargs):
            def save(output):
                Path(output, "audio.flac").write_bytes(b"audio")
                return {"truncated": {"abc": False, "semantic": False},
                        "audio_seconds": 1, "sample_rate": 48000, "timing": {"seed": seed}}
            return SimpleNamespace(abc=None, config={}, save_artifacts=save)

        def render_ar(self, seed, **kwargs):
            raise AssertionError(f"Unexpected single-row fallback for {seed}")

    settings = Settings(api_key=KEY, data_dir=tmp_path, warmup=False, nar_batch_size=2)
    pipe = RowFailurePipeline()
    worker = JobWorker(settings, lambda _: pipe)
    worker.pipeline = pipe
    jobs = [worker.store.submit({**REQUEST, "seed": seed}, 2)[0] for seed in (1, 2)]
    assert worker._execute_claimed(worker.store.claim_many(2)) is False
    assert [worker.store.get(job["id"])["status"] for job in jobs] == ["failed", "succeeded"]


def test_worker_does_not_reorder_fifo_across_ineligible_job(tmp_path):
    class MixedPipeline:
        def __init__(self):
            self.saved = []
            self.started = []

        def parallel_ar_eligible(self, request):
            return request.get("cot") != "off"

        def generate_ar(self, *, seed, **kwargs):
            self.started.append(seed)
            return seed

        def render_ar(self, seed, **kwargs):
            return self.result(seed)

        def __call__(self, *, seed, **kwargs):
            self.started.append(seed)
            return self.result(seed)

        def result(self, seed):
            def save(output):
                self.saved.append(seed)
                Path(output, "audio.flac").write_bytes(b"audio")
                return {"truncated": {"abc": False, "semantic": False},
                        "audio_seconds": 1, "sample_rate": 48000, "timing": {"seed": seed}}
            return SimpleNamespace(abc=None, config={}, save_artifacts=save)

    settings = Settings(api_key=KEY, data_dir=tmp_path, warmup=False, nar_batch_size=4)
    pipe = MixedPipeline()
    worker = JobWorker(settings, lambda _: pipe)
    worker.pipeline = pipe
    requests = [{**REQUEST, "seed": 1}, {**REQUEST, "seed": 2, "cot": "off"},
                {**REQUEST, "seed": 3}]
    jobs = [worker.store.submit(request, 3)[0] for request in requests]
    assert worker._execute_claimed(worker.store.claim_many(3)) is False
    assert pipe.started == [1, 2, 3]
    assert pipe.saved == [1, 2, 3]
    assert all(worker.store.get(job["id"])["status"] == "succeeded" for job in jobs)


def test_cancel_racing_success_cannot_publish_result(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job, _ = store.submit(REQUEST, 1)
    store.claim()
    store.cancel(job["id"])
    store.finish(job["id"], "succeeded", result={"audio_url": "should-not-publish"})
    assert store.get(job["id"])["status"] == "cancelled"
    assert store.get(job["id"])["result"] is None


def test_environment_settings_validate_and_hide_secret(monkeypatch):
    monkeypatch.setenv("YUE2_API_KEY", KEY)
    monkeypatch.delenv("YUE2_RESIDENT_MODELS", raising=False)
    monkeypatch.setenv("YUE2_BACKEND", "vllm")
    settings = Settings.from_env()
    assert settings.resident_models and settings.backend == "vllm"
    assert settings.ar_concurrency == settings.vllm_max_num_seqs == 4
    assert settings.vllm_gpu_memory_utilization == .3
    assert settings.vllm_max_num_batched_tokens == 8192
    assert settings.nar_batch_size == 4
    assert settings.ar_batch_wait_ms == 50
    assert KEY not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(api_key=KEY, vllm_gpu_memory_utilization=1)
    with pytest.raises(ValidationError):
        Settings(api_key=KEY, vllm_max_num_batched_tokens=24577)
    compatible = Settings(api_key=KEY, ar_concurrency=2, vllm_max_num_seqs=2)
    assert compatible.nar_batch_size == 4
