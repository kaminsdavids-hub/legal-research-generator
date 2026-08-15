"""Long pipeline steps as jobs: submit, poll, collect.

Every route that drives a model can run for minutes -- the backend's own budget
is ``LRG_LLM_TIMEOUT_SECONDS=300`` -- and anything in front of it has a shorter
patience than that. Netlify's synchronous functions cap at 26 seconds, most
reverse proxies default to 60, and a browser tab on a phone gives up sooner
still. A request that takes five minutes is not a request any of them can carry,
however it is written.

So the slow work stops being a request. ``POST`` submits and returns immediately
with an id; the client polls a route that answers in milliseconds. Every hop in
between now sees only fast calls.

**In-process and not durable.** Jobs live in this process's memory and die with
it, exactly like ``_sessions`` in ``app.py``. That is consistent rather than
lazy: the blackboard a job mutates is itself in-process, so persisting the job
without persisting the session would buy nothing. A restart loses both, and
:meth:`JobStore.submit` says so rather than implying a durability it lacks.

**One job per session at a time.** Steps mutate a shared blackboard in place, so
two running against one session would interleave writes and corrupt it. A second
submission is refused rather than queued: queueing would hide from the caller
that their step has not started, and the pipeline's steps are ordered anyway --
research before draft, draft before verify -- so a queue would mostly be a way
to run them in the wrong order.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Job", "JobState", "JobStore", "SessionBusy"]


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SessionBusy(Exception):
    """A step is already running for this session."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"a step is already running for this session (job {job_id})")
        self.job_id = job_id


@dataclass
class Job:
    id: str
    session_id: str
    step: str
    state: JobState = JobState.RUNNING
    #: Set when the step fails. Carried as a string because the client is across
    #: a network and an exception object is not; the traceback stays server-side
    #: in the log, since handing a caller our stack frames is a disclosure with
    #: no benefit to them.
    error: str = ""
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def done(self) -> bool:
        return self.state is not JobState.RUNNING

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the step finishes. For tests, not for request handlers."""

        return self._finished.wait(timeout)


class JobStore:
    """Runs steps on worker threads and remembers how they ended.

    Threads rather than an async task group because the pipeline is synchronous
    and blocking -- it makes network calls to local model servers with an
    ordinary client. Awaiting it on the event loop would stall every other
    request in the process, including the poll that is supposed to answer in
    milliseconds, which would defeat the entire point of this module.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._by_session: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(self, session_id: str, step: str, work: Callable[[], Any]) -> Job:
        """Start ``work`` on a thread and return its job immediately.

        Raises :class:`SessionBusy` if a step is already running for this
        session.
        """

        with self._lock:
            active = self._by_session.get(session_id)
            if active is not None and not self._jobs[active].done:
                raise SessionBusy(active)

            job = Job(id=uuid.uuid4().hex[:12], session_id=session_id, step=step)
            self._jobs[job.id] = job
            self._by_session[session_id] = job.id

        thread = threading.Thread(
            target=self._run, args=(job, work), name=f"job-{step}-{job.id}", daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, work: Callable[[], Any]) -> None:
        try:
            work()
            job.state = JobState.SUCCEEDED
        except Exception as exc:  # noqa: BLE001 - a failed step must be reportable
            # The step failed, not the server. A job that vanished or hung would
            # leave the client polling forever with nothing to show a user.
            job.state = JobState.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            # Set last, and always: a client that saw RUNNING must eventually see
            # a terminal state, including when the step raised.
            job._finished.set()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active_for(self, session_id: str) -> Job | None:
        job_id = self._by_session.get(session_id)
        if job_id is None:
            return None
        job = self._jobs[job_id]
        return None if job.done else job
