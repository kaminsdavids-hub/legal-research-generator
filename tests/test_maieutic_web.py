"""The loop's HTTP surface. Offline: no models, no network."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modules.maieutic.web import SESSION_ID, create_app

THESIS = "Publishing open model weights is not a deemed export."
ANSWER = "The strongest objection is that weights are functional rather than published."
REPLY = "Functionality has never defeated the exclusion for open technical material."


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _begin(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"thesis": THESIS})
    assert response.status_code == 200
    return response.json()["session_id"]


# --------------------------------------------------------------------------- #
# Session ids are validated, not trusted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "..%2F..%2Fsecret", "a/b", "", "x" * 64, "NOTHEX123456"],
)
def test_a_crafted_session_id_never_reaches_the_filesystem(
    client: TestClient, bad: str
) -> None:
    """Each session is a file named from a client-supplied id. A local-first
    tool is still a tool with an HTTP server in it.
    """
    response = client.get(f"/api/sessions/{bad}")
    # 400 from the id pattern, 404 when routing normalises it to a miss, 405
    # when it collapses onto the POST-only collection route. What matters is
    # that none of them is a 200 and none reaches the filesystem.
    assert response.status_code in (400, 404, 405), response.text
    assert "passwd" not in response.text


def test_a_traversal_id_cannot_write_outside_the_session_directory(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "sessions"))
    client.post("/api/sessions/..%2F..%2Fescaped/answer", json={"text": "x"})
    assert not (tmp_path / "escaped.json").exists()


def test_generated_ids_match_the_pattern_that_guards_the_path(
    client: TestClient,
) -> None:
    assert SESSION_ID.match(_begin(client))


def test_an_unknown_but_well_formed_id_is_a_404(client: TestClient) -> None:
    assert client.get("/api/sessions/0123456789ab").status_code == 404


# --------------------------------------------------------------------------- #
# The author's channel is the only way in
# --------------------------------------------------------------------------- #
def test_there_is_no_endpoint_that_accepts_a_node_or_a_provenance(
    client: TestClient,
) -> None:
    """A client that could post machine-drafted text as human-authored would
    defeat the record the whole system keeps.
    """
    paths = {route.path for route in client.app.routes}  # type: ignore[attr-defined]
    assert not any("node" in p or "patch" in p or "provenance" in p for p in paths)


def test_an_answer_becomes_a_human_node(client: TestClient, tmp_path: Path) -> None:
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})

    from modules.maieutic.loop import Session

    session = Session.load(tmp_path / f"{session_id}.json")
    authored = [n for n in session.graph.nodes.values() if n.text == ANSWER]
    assert authored and authored[0].provenance.value == "human"


def test_an_empty_answer_is_rejected_by_the_schema(client: TestClient) -> None:
    session_id = _begin(client)
    assert client.post(f"/api/sessions/{session_id}/answer", json={"text": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# Refusals are part of the response, not an error
# --------------------------------------------------------------------------- #
def test_a_refused_patch_returns_200_with_its_reasons(client: TestClient) -> None:
    """The author asked a question and got an answer about their work, which is
    a successful interaction whatever the merge decided.
    """
    session_id = _begin(client)
    response = client.post(
        f"/api/sessions/{session_id}/answer",
        json={"text": "It may perhaps arguably possibly seem so."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["merged"] is False
    assert any("banality" in r for r in body["refusals"])


def test_a_refused_answer_leaves_the_question_pending(client: TestClient) -> None:
    session_id = _begin(client)
    before = client.get(f"/api/sessions/{session_id}").json()["pending"]
    client.post(
        f"/api/sessions/{session_id}/answer",
        json={"text": "It may perhaps arguably possibly seem so."},
    )
    after = client.get(f"/api/sessions/{session_id}").json()["pending"]
    assert after == before


def test_a_merged_answer_returns_the_next_question(client: TestClient) -> None:
    session_id = _begin(client)
    body = client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER}).json()
    assert body["merged"] is True
    assert body["next_question"]["gap_kind"] == "unanswered_attack"


# --------------------------------------------------------------------------- #
# The loop over HTTP
# --------------------------------------------------------------------------- #
def test_beginning_returns_the_first_question(client: TestClient) -> None:
    body = client.post("/api/sessions", json={"thesis": THESIS}).json()
    assert body["pending"]["gap_kind"] == "uncontested_claim"
    assert body["nodes"] == 1


def test_an_empty_thesis_is_rejected(client: TestClient) -> None:
    assert client.post("/api/sessions", json={"thesis": "   "}).status_code in (400, 422)


def test_the_manuscript_carries_the_authors_words(client: TestClient) -> None:
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    body = client.get(f"/api/sessions/{session_id}/manuscript").json()
    assert THESIS in body["text"]
    assert ANSWER in body["text"]


def test_the_review_view_annotates_provenance(client: TestClient) -> None:
    session_id = _begin(client)
    body = client.get(f"/api/sessions/{session_id}/manuscript?review=true").json()
    assert "`[human]`" in body["text"]


def test_unanswered_objections_reach_the_client(client: TestClient) -> None:
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    body = client.get(f"/api/sessions/{session_id}").json()
    assert body["open_problems"] == [ANSWER]


def test_skipping_records_a_decline_and_offers_the_next(client: TestClient) -> None:
    session_id = _begin(client)
    body = client.post(f"/api/sessions/{session_id}/skip").json()
    assert body["pending"] is None or body["pending"]["gap_kind"] != "uncontested_claim"

    policy = client.get("/api/policy").json()
    assert policy["summary"]["uncontested_claim"]["answered"] == 0


def test_skipping_with_nothing_pending_is_a_conflict(client: TestClient) -> None:
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/skip")
    while client.post(f"/api/sessions/{session_id}/skip").status_code == 200:
        pass
    assert client.post(f"/api/sessions/{session_id}/skip").status_code == 409


def test_two_sessions_do_not_share_a_manuscript(client: TestClient) -> None:
    first, second = _begin(client), _begin(client)
    assert first != second
    client.post(f"/api/sessions/{first}/answer", json={"text": ANSWER})
    assert client.get(f"/api/sessions/{second}").json()["nodes"] == 1


def test_the_journal_is_shared_across_sessions(client: TestClient) -> None:
    """What it records is a fact about the author, not about one manuscript."""
    for _ in range(3):
        session_id = _begin(client)
        client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    summary = client.get("/api/policy").json()["summary"]
    assert summary["uncontested_claim"]["asked"] == 3


def test_the_policy_names_the_signal_it_learns_from(client: TestClient) -> None:
    body = client.get("/api/policy").json()
    assert "never from whether the gates accepted it" in body["signal"]


def test_the_page_is_self_contained(client: TestClient) -> None:
    """A client that needs a toolchain to try is a client nobody tries."""
    body = client.get("/").text
    assert "<title>maieutic</title>" in body
    assert "src=" not in body, "no external scripts"
    assert "http://" not in body and "https://" not in body


# --------------------------------------------------------------------------- #
# A live answer is a job, not a request
# --------------------------------------------------------------------------- #
def _slow_turn(delay: float = 0.3):  # type: ignore[no-untyped-def]
    """Stands in for a real exchange, which measured 236-294s (§13)."""
    import time

    from modules.dialectic.models import CitationSlot, DialecticTurn, Position

    def provide(question: str, answer: str) -> DialecticTurn:
        time.sleep(delay)
        return DialecticTurn(
            question=question,
            thesis=Position(
                side="thesis", model="a", family="fa",
                propositions=[CitationSlot(proposition="A supporting point.")],
            ),
            antithesis=Position(
                side="antithesis", model="b", family="fb",
                propositions=[CitationSlot(proposition="A counter-point.")],
            ),
        )

    return provide


def _live_client(tmp_path: Path, provider=None) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(create_app(tmp_path, turns=provider or _slow_turn()))


def _await_job(client: TestClient, session_id: str) -> dict:  # type: ignore[type-arg]
    import time

    for _ in range(100):
        body = client.get(f"/api/sessions/{session_id}/answer").json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_a_live_answer_returns_202_rather_than_blocking(tmp_path: Path) -> None:
    """A synchronous endpoint that blocks for five minutes is not something a
    browser or a proxy will hold.
    """
    client = _live_client(tmp_path)
    session_id = _begin(client)
    response = client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    assert response.status_code == 202
    assert response.json()["state"] == "running"


def test_the_job_carries_the_same_result_a_sync_call_would(tmp_path: Path) -> None:
    client = _live_client(tmp_path)
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    body = _await_job(client, session_id)
    assert body["state"] == "done"
    assert body["result"]["merged"] is True
    assert body["result"]["added"] == 3, "the author's answer plus both sides"
    assert body["result"]["next_question"]["gap_kind"] == "unanswered_attack"


def test_two_exchanges_cannot_run_at_once_for_one_session(tmp_path: Path) -> None:
    """They would race on the session file and on which answer the author meant."""
    client = _live_client(tmp_path, _slow_turn(delay=1.0))
    session_id = _begin(client)
    first = client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    second = client.post(f"/api/sessions/{session_id}/answer", json={"text": REPLY})
    assert first.status_code == 202
    assert second.status_code == 409
    _await_job(client, session_id)


def test_a_failed_exchange_still_keeps_the_authors_answer(tmp_path: Path) -> None:
    """The author's work is not hostage to a model being down (M42)."""

    def _down(question: str, answer: str):  # type: ignore[no-untyped-def]
        raise RuntimeError("connection refused")

    client = _live_client(tmp_path, _down)
    session_id = _begin(client)
    client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    body = _await_job(client, session_id)
    assert body["state"] == "done"
    assert body["result"]["merged"] is True
    assert "connection refused" in body["result"]["turn_error"]


def test_polling_a_session_with_no_exchange_is_a_404(tmp_path: Path) -> None:
    client = _live_client(tmp_path)
    assert client.get(f"/api/sessions/{_begin(client)}/answer").status_code == 404


def test_offline_stays_synchronous(client: TestClient) -> None:
    """Nothing to poll for when the exchange is instant, and the client is told
    which it got rather than having to guess.
    """
    session_id = _begin(client)
    response = client.post(f"/api/sessions/{session_id}/answer", json={"text": ANSWER})
    assert response.status_code == 200
    assert "merged" in response.json()


def test_a_bad_session_id_is_refused_before_a_job_is_started(tmp_path: Path) -> None:
    client = _live_client(tmp_path)
    assert client.post("/api/sessions/notavalidid/answer", json={"text": ANSWER}).status_code == 400
