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

**One *writing* job per session at a time.** Pipeline steps mutate a shared
blackboard in place, so two running against one session would interleave writes
and corrupt it. A second submission is refused rather than queued: queueing would
hide from the caller that their step has not started, and the pipeline's steps
are ordered anyway -- research before draft, draft before verify -- so a queue
would mostly be a way to run them in the wrong order.

Read-only work is exempt, and the distinction is not a nicety. Multi-chat and
dialectic answer a question without touching the blackboard; today a user can
have two conversations going at once, and putting them under the writers' lock
would take that away to prevent a corruption they cannot cause. A mutual
exclusion that guards nothing is just a queue nobody asked for.
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
    #: A small terminal summary, set by the work itself on success. Deliberately
    #: small: the poll must stay cheap, so the blackboard never goes here. It
    #: exists because `run-all` produces something that is not in the blackboard
    #: at all -- the per-agent step log and the shippable verdict -- and that
    #: would otherwise be lost the moment the run stopped being a request.
    result: dict[str, Any] | None = None
    #: Append-only progress, in the order things actually happened. The panel
    #: runs five models concurrently, so this is completion order and not the
    #: configured order -- which is the point: it is what the user is waiting on.
    events: list[dict[str, Any]] = field(default_factory=list)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)
    #: Guards `events` and wakes followers. The engine emits from pool threads,
    #: so appends are genuinely concurrent.
    _progress: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def done(self) -> bool:
        return self.state is not JobState.RUNNING

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the step finishes. For tests and worker threads.

        Never call this from a request handler: it blocks the thread, and the
        whole design is that request handlers return immediately. The event-
        stream route waits on it in a thread pool instead.
        """

        return self._finished.wait(timeout)

    def emit(self, event: dict[str, Any]) -> None:
        """Record progress and wake anyone following.

        Called from whatever thread the work runs on, including the panel's
        pool threads.
        """

        with self._progress:
            self.events.append(event)
            self._progress.notify_all()

    def follow(self, seen: int, timeout: float) -> list[dict[str, Any]]:
        """Progress beyond index ``seen``, waiting up to ``timeout`` for some.

        Returns an empty list if nothing arrived, which the caller reads as
        "still working" rather than "finished" -- completion is `done`, and the
        two must not be conflated or a quiet job would look like a finished one.
        """

        with self._progress:
            if len(self.events) <= seen and not self.done:
                self._progress.wait(timeout)
            return self.events[seen:]

    def snapshot(self) -> dict[str, Any]:
        """What a client is told about this job. One definition, so the polling
        route and the event stream cannot describe the same job differently."""

        return {
            "job_id": self.id,
            "session_id": self.session_id,
            "step": self.step,
            "state": self.state.value,
            "error": self.error,
            "result": self.result,
            "events": list(self.events),
        }


class JobStore:
    """Runs steps on worker threads and remembers how they ended.

    Threads rather than an async task group because the pipeline is synchronous
    and blocking -- it makes network calls to local model servers with an
    ordinary client. Awaiting it on the event loop would stall every other
    request in the process, including the poll that is supposed to answer in
    milliseconds, which would defeat the entire point of this module.
    """

    #: Finished jobs kept before the oldest are dropped.
    #:
    #: They were kept forever. A job holds its whole result -- a jury exchange
    #: is several KB of prose -- plus every progress event, and this process is
    #: a long-running service, so the store only ever grew. Nothing removed
    #: anything: no TTL, no cap, no delete.
    #:
    #: 200 is chosen to be far past any plausible polling window rather than
    #: tight: a client fetches a result once, moments after the job ends, and
    #: nothing reads a job again after that. Running jobs are never dropped at
    #: any count, so a slow step cannot be evicted out from under the client
    #: waiting on it.
    MAX_FINISHED_JOBS = 200

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._by_session: dict[str, str] = {}
        self._lock = threading.Lock()

    def _prune_locked(self) -> None:
        """Drop the oldest finished jobs. Caller must hold the lock.

        Insertion order is submission order, so the dict is already oldest
        first and no timestamp is needed on ``Job`` -- one that existed only to
        support eviction would end up on the wire the next time somebody
        serialised the job.

        ``_by_session`` is cleaned alongside. It points at the session's most
        recent *mutating* job, and ``submit`` reads ``self._jobs[active]`` to
        decide whether the session is busy; leaving a pointer to a dropped job
        would raise KeyError there and refuse every later submission on that
        session.
        """
        finished = [job_id for job_id, job in self._jobs.items() if job.done]
        excess = len(finished) - self.MAX_FINISHED_JOBS
        if excess <= 0:
            return
        for job_id in finished[:excess]:
            job = self._jobs.pop(job_id)
            if self._by_session.get(job.session_id) == job_id:
                del self._by_session[job.session_id]

    def submit(
        self,
        session_id: str,
        step: str,
        work: Callable[[Job], Any],
        *,
        mutates: bool = True,
    ) -> Job:
        """Start ``work`` on a thread and return its job immediately.

        Whatever ``work`` returns, if it is a ``dict``, becomes the job's
        ``result``. Anything else is discarded -- the pipeline steps return
        agent objects that mean nothing to a client, and serialising them would
        put internals on the wire by accident.

        ``work`` always takes the job as its only argument -- that is how a step
        reports progress, since anything that wants to emit needs a handle on
        the job it belongs to. One signature, always, rather than accepting
        either shape and inferring which was passed: that inference read
        ``threading.Event.wait`` as wanting the job, called it with the job as
        its timeout, and quietly turned the session lock off. A step with
        nothing to report names the argument and ignores it.

        ``mutates=False`` marks work that only reads the session. It neither
        takes the per-session slot nor is blocked by it, so several can run at
        once and a long draft does not stop the user asking a question. Raises
        :class:`SessionBusy` only for mutating work.
        """

        with self._lock:
            if mutates:
                active = self._by_session.get(session_id)
                # `.get` rather than `[...]`: _prune_locked keeps the two in
                # step, but a stale pointer must degrade to "not busy" rather
                # than raising and locking the session out permanently.
                running = self._jobs.get(active) if active is not None else None
                if running is not None and not running.done:
                    raise SessionBusy(active)

            self._prune_locked()

            job = Job(id=uuid.uuid4().hex[:12], session_id=session_id, step=step)
            self._jobs[job.id] = job
            if mutates:
                self._by_session[session_id] = job.id

        thread = threading.Thread(
            target=self._run, args=(job, work), name=f"job-{step}-{job.id}", daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, work: Callable[[Job], Any]) -> None:
        try:
            outcome = work(job)
            if isinstance(outcome, dict):
                job.result = outcome
            job.state = JobState.SUCCEEDED
        except Exception as exc:  # noqa: BLE001 - a failed step must be reportable
            # The step failed, not the server. A job that vanished or hung would
            # leave the client polling forever with nothing to show a user.
            job.state = JobState.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            # Set last, and always: a client that saw RUNNING must eventually see
            # a terminal state, including when the step raised. Followers are
            # woken too, or one parked in `follow` would sit out its full
            # timeout after the job had already ended.
            job._finished.set()
            with job._progress:
                job._progress.notify_all()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active_for(self, session_id: str) -> Job | None:
        job_id = self._by_session.get(session_id)
        if job_id is None:
            return None
        job = self._jobs[job_id]
        return None if job.done else job
