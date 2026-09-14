"""Durable single-machine job queue. SQLite transactions bound admission atomically."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


TERMINAL = {"succeeded", "truncated", "failed", "cancelled"}


class QueueFull(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class JobStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL,
                request TEXT NOT NULL, request_hash TEXT NOT NULL,
                idem_key TEXT UNIQUE, snapshot TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _write(db, job):
        job["updated_at"] = time.time()
        db.execute("UPDATE jobs SET status=?, snapshot=? WHERE id=?",
                   (job["status"], json.dumps(job), job["id"]))

    def recover(self):
        # Queued work survives; a process killed mid-generation cannot resume exact GPU state.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT snapshot FROM jobs WHERE status='running'").fetchall():
                job = json.loads(row[0])
                job.update(status="failed", stage="finished", finished_at=time.time(),
                           error={"code": "worker_interrupted", "message": "Worker stopped during generation; submit a new job."})
                self._write(db, job)

    def submit(self, request, max_pending, idem_key=None):
        raw = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem_key is not None:
                existing = db.execute("SELECT request_hash, snapshot FROM jobs WHERE idem_key=?", (idem_key,)).fetchone()
                if existing:
                    if existing[0] != digest:
                        raise IdempotencyConflict()
                    return json.loads(existing[1]), False
            pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending >= max_pending:
                raise QueueFull()
            now, job_id = time.time(), uuid.uuid4().hex
            job = dict(id=job_id, status="queued", stage="queued", created_at=now, updated_at=now,
                       started_at=None, finished_at=None, cancel_requested=False,
                       tokens={"abc": 0, "semantic": 0}, result=None, error=None)
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                       (job_id, "queued", now, raw, digest, idem_key, json.dumps(job)))
            return job, True

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT snapshot, request FROM jobs WHERE status='queued' ORDER BY created, id LIMIT 1").fetchone()
            if row is None:
                return None
            job = json.loads(row[0])
            job.update(status="running", stage="loading", started_at=time.time())
            self._write(db, job)
            return job, json.loads(row[1])

    def progress(self, job_id, **fields):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            job = json.loads(row[0])
            if job["status"] == "running":
                job.update(fields)
                self._write(db, job)

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            job = json.loads(row[0])
            if job["status"] not in TERMINAL:
                job["cancel_requested"] = True
                if job["status"] == "queued":
                    job.update(status="cancelled", stage="finished", finished_at=time.time())
                self._write(db, job)
            return job

    def finish(self, job_id, status, *, result=None, error=None):
        if status not in TERMINAL:
            raise ValueError("Expected terminal status")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            if job["cancel_requested"]:
                status, result, error = "cancelled", None, None
            job.update(status=status, stage="finished", finished_at=time.time(), result=result, error=error)
            self._write(db, job)
