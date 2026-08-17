"""What survives the trip from a dialectic turn to the wire.

``_dialectic_response`` maps slots field by field. A field added to
``CitationSlot`` and to the response schema therefore still arrives as its
default until it is also listed in that mapping, and nothing fails — the type
checks pass, the suite passes, the request returns 200, and the value is simply
gone. That is how ``verified_by`` shipped invisible: the engine set it, the
schema declared it, the panel was ready to render it, and the browser received
"" on every slot. It was caught by looking at a screenshot.

So the guard here is not "verified_by is carried" but "every field is carried",
because the next field added will fail the same way and a test naming one field
would not notice.
"""

from __future__ import annotations

import pytest

from modules.dialectic.models import (
    CitationSlot,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)


@pytest.fixture
def build_response(monkeypatch):
    """Import the app only after the environment is pinned to the mock backend.

    Importing ``legal_research.api.app`` constructs the FastAPI application at
    module scope, and that constructor builds ``MultiModelChat``, whose
    ``__init__`` starts the warm-up thread when ``llm_warmup_enabled``. A
    module-level import therefore warms seven Ollama models the moment pytest
    *collects* this file — before a single test runs, against the real .env,
    on every invocation of the suite. Done that way once here: the run reached
    two tests in six minutes while the GPU loaded models nothing was going to
    use.

    ``tests/test_api.py`` already solves this by importing inside a fixture
    after setting the mode, which is the pattern followed here.
    """
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_LLM_WARMUP_ENABLED", "false")
    from legal_research.api.app import _dialectic_response

    return _dialectic_response


def _turn() -> DialecticTurn:
    """A turn whose slots carry a non-default value in every field.

    Defaults are useless here: a mapping that dropped a field would still
    produce the default, and the assertion would pass against the bug.
    """
    corpus_slot = CitationSlot(
        proposition="The published-information exception applies.",
        court_hint="Interpretation of statutory language in export control regulations",
        weight=Weight.CONTROLLING,
        status=SlotStatus.VERIFIED,
        cluster_id="cluster-1",
        normalized_cite="15 C.F.R. 734.7",
        note="confirmed against the local corpus, not by citation lookup",
        verified_by="corpus",
    )
    lookup_slot = CitationSlot(
        proposition="Source code is protected expression.",
        court_hint="Supreme Court precedent on the First Amendment",
        weight=Weight.PERSUASIVE,
        status=SlotStatus.VERIFIED,
        cluster_id="cluster-2",
        normalized_cite="176 F.3d 1132",
        note="verified",
        verified_by="courtlistener",
    )
    return DialecticTurn(
        question="Does the exception apply?",
        thesis=Position(side="thesis", model="saul:7b-instruct-v1", family="saul",
                        propositions=[corpus_slot]),
        antithesis=Position(side="antithesis", model="llama3.1:8b", family="llama",
                            propositions=[lookup_slot]),
        synthesis="a synthesis",
        calls_spent=2,
        regenerated=1,
    )


def test_every_slot_field_reaches_the_wire(build_response) -> None:
    """The general guard: no field may be dropped by the hand-written mapping."""
    response = build_response(_turn())
    wire = response.thesis.propositions[0]

    missing = [
        name
        for name in CitationSlot.model_fields
        if not hasattr(wire, name)
    ]
    assert not missing, f"schema does not declare: {missing}"

    source = _turn().thesis.propositions[0]
    dropped = [
        name
        for name in CitationSlot.model_fields
        if str(getattr(wire, name)) != str(getattr(source, name))
    ]
    assert not dropped, f"mapping dropped: {dropped}"


def test_the_two_warrants_stay_distinguishable(build_response) -> None:
    """The specific case, because collapsing these is the harm the field prevents.

    A corpus record and a citation lookup are different warrants. If both arrive
    as "verified" with no provenance, the panel renders one badge and the weaker
    one borrows the stronger one's credibility.
    """
    response = build_response(_turn())

    assert response.thesis.propositions[0].verified_by == "corpus"
    assert response.antithesis.propositions[0].verified_by == "courtlistener"


def test_an_unverified_slot_claims_no_warrant(build_response) -> None:
    turn = _turn()
    turn.thesis.propositions[0] = CitationSlot(
        proposition="unsupported",
        status=SlotStatus.PROPOSED,
        normalized_cite="15 C.F.R. 999.1",
    )
    response = build_response(turn)

    assert response.thesis.propositions[0].verified_by == ""
