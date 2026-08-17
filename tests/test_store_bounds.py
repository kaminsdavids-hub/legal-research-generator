"""The two in-memory stores are bounded, and bounded safely.

Both grew forever. ``JobStore._jobs`` and ``app._sessions`` were plain dicts
with no TTL, no cap and no delete anywhere, in a process that runs as a service
for days. A job holds its whole result — a jury exchange is several KB of prose
— plus every progress event; a session holds an entire blackboard.

Bounding them is easy. Bounding them without throwing away work somebody is
still doing is the part worth testing:

* a *running* job must never be evicted, however many have piled up behind it;
* ``_by_session`` must not be left pointing at a job that no longer exists, or
  the busy check raises and locks that session out of every later submission;
* a session under active use must never be the one dropped, which is why the
  bound is least-recently-*used* rather than oldest-created.
"""

from __future__ import annotations

import importlib
import threading

import pytest


@pytest.fixture
def JobStore(monkeypatch):
    """Import behind the mock backend.

    legal_research/api/__init__.py does `from .app import app`, so importing
    anything under that package constructs the whole FastAPI application --
    43s, and it starts the Ollama warm-up thread. At module scope that
    happens during pytest *collection*, before a single test runs, on every
    invocation of the suite.
    """
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_LLM_WARMUP_ENABLED", "false")
    return importlib.import_module("legal_research.api.jobs").JobStore


def _finished(store, n: int, session_prefix: str = "s") -> None:
    """Submit n jobs that complete immediately, each on its own session."""
    for i in range(n):
        job = store.submit(f"{session_prefix}-{i}", "multi-chat", lambda _job: {"ok": True})
        _wait(store, job.id)


def _wait(store, job_id: str, timeout: float = 5.0) -> None:
    deadline = threading.Event()
    for _ in range(int(timeout * 100)):
        job = store.get(job_id) if hasattr(store, "get") else store._jobs.get(job_id)
        if job is None or job.done:
            return
        deadline.wait(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_finished_jobs_stop_accumulating(JobStore) -> None:
    store = JobStore()
    store.MAX_FINISHED_JOBS = 5

    _finished(store, 12)

    assert len(store._jobs) <= 6, "the store must not grow with every submission"


def test_a_running_job_is_never_evicted(JobStore) -> None:
    """The eviction that would matter: dropping the job a client is waiting on."""
    store = JobStore()
    store.MAX_FINISHED_JOBS = 2

    release = threading.Event()
    slow = store.submit("slow-session", "dialectic", lambda _job: release.wait(5) or {})

    # Bury it under far more finished jobs than the cap allows.
    _finished(store, 10, session_prefix="filler")

    assert slow.id in store._jobs, "a running job was evicted out from under its client"
    release.set()
    _wait(store, slow.id)


def test_eviction_does_not_lock_a_session_out(JobStore) -> None:
    """The subtle one: a stale _by_session pointer breaks all later submissions.

    ``submit`` consults ``_by_session`` to decide whether a session already has
    mutating work in flight. If pruning removes the job but leaves the pointer,
    that lookup finds an id with no job behind it — and every subsequent
    submission on that session fails.
    """
    store = JobStore()
    store.MAX_FINISHED_JOBS = 1

    first = store.submit("reused", "draft", lambda _job: {"ok": True})
    _wait(store, first.id)

    # Push the first job out of the store.
    _finished(store, 5, session_prefix="filler")

    # The same session must still accept work.
    again = store.submit("reused", "draft", lambda _job: {"ok": True})
    _wait(store, again.id)
    assert again.state.value == "succeeded"


def test_a_busy_session_is_still_refused(JobStore) -> None:
    """Bounding the store must not weaken the one-writer-per-session rule."""
    SessionBusy = importlib.import_module("legal_research.api.jobs").SessionBusy

    store = JobStore()
    release = threading.Event()
    store.submit("busy", "draft", lambda _job: release.wait(5) or {})

    with pytest.raises(SessionBusy):
        store.submit("busy", "verify", lambda _job: {"ok": True})

    release.set()


# --- sessions ----------------------------------------------------------------


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_LLM_WARMUP_ENABLED", "false")
    # importlib, not `from ... import app`: the package re-exports the
    # FastAPI instance under that name and it shadows the submodule.
    module = importlib.import_module("legal_research.api.app")

    module._sessions.clear()
    return module


def _bb(module, session_id: str):
    from legal_research.blackboard import Blackboard

    bb = Blackboard(session_id=session_id, title="t")
    module._remember(bb)
    return bb


def test_sessions_stop_accumulating(app_module) -> None:
    app_module.MAX_SESSIONS = 5
    for i in range(12):
        _bb(app_module, f"session-{i}")

    assert len(app_module._sessions) == 5


def test_a_session_in_use_is_never_the_one_dropped(app_module) -> None:
    """Least-recently-used, not oldest-created: this is the whole safety claim.

    The first session is created before all the others and would be the first
    to go under an insertion-order bound. Reading it keeps it alive, which is
    what makes an LRU cap safe for a paper somebody is still working on.
    """
    app_module.MAX_SESSIONS = 5
    _bb(app_module, "the-paper-in-progress")

    for i in range(4):
        _bb(app_module, f"other-{i}")
        app_module._get("the-paper-in-progress")  # still being worked on

    for i in range(4, 10):
        _bb(app_module, f"other-{i}")
        app_module._get("the-paper-in-progress")

    assert "the-paper-in-progress" in app_module._sessions
    assert len(app_module._sessions) == 5


def test_an_untouched_session_is_the_one_dropped(app_module) -> None:
    app_module.MAX_SESSIONS = 3
    _bb(app_module, "abandoned")
    for i in range(3):
        _bb(app_module, f"newer-{i}")

    assert "abandoned" not in app_module._sessions
