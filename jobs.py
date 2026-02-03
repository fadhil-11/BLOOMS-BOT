from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class Job:
    id: str
    status: str
    step: str
    progress: int
    error: Optional[str]
    result: Optional[Dict[str, Any]]
    created_at: str


JOBS: Dict[str, Job] = {}
JOBS_LOCK = Lock()


def create_job() -> Job:
    job = Job(
        id=uuid4().hex,
        status="queued",
        step="Queued",
        progress=0,
        error=None,
        result=None,
        created_at=datetime.utcnow().isoformat(),
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> Optional[Job]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        return job


def run_in_thread(job_id: str, fn, *args, **kwargs) -> Thread:
    def runner() -> None:
        try:
            update_job(job_id, status="running")
            result = fn(*args, **kwargs)
            update_job(
                job_id,
                status="done",
                step="Done",
                progress=100,
                result=result,
            )
        except Exception as exc:  # pragma: no cover - defensive
            update_job(
                job_id,
                status="error",
                step="Error",
                error=str(exc),
            )

    thread = Thread(target=runner, daemon=True)
    thread.start()
    return thread
