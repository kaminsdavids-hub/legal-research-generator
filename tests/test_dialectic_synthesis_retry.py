"""What a rejected model is told before it is asked again.

Both retry loops re-roll on temperature and seed. That is enough when the model
was merely unlucky, and useless when it was wrong in a way it cannot guess at:
the citation channel forbids naming authority, "Bernstein v. U.S. Dep" reads to
a model like ordinary legal prose, and being asked the identical question at a
higher temperature produces the identical mistake.

Observed on every dialectic run in the shipped configuration:
``citation string(s) rejected: ['Bernstein v. U.S. Dep']; after 3 attempts`` —
the whole regeneration budget spent, no synthesis returned, and the panel
showing the failure text where the answer should be.

Each loop had branches that fed their reason back and branches that did not,
which is the state these tests pin. Every case here asserts that the retry
prompt differs from the attempt that failed by more than a seed, and each one
fails if its feedback template is emptied — so they hold the fix rather than
describe the code.
"""

from __future__ import annotations

from modules.dialectic.engine import DialecticChat
from modules.dialectic.models import CitationSlot, DialecticTurn, Position


class _ScriptedClient:
    """Returns queued answers and records the prompt it was given each time.

    Carries a ``name`` because the constructor refuses a lineup whose roles come
    from one model family — same-family debaters have correlated errors and
    produce agreement dressed as debate.
    """

    def __init__(self, answers: list[str], name: str = "stub") -> None:
        self._answers = list(answers)
        self.name = name
        self.prompts: list[str] = []

    def chat(self, messages, config=None):  # noqa: ANN001 - test double
        self.prompts.append(messages[-1]["content"])
        return self._answers.pop(0) if self._answers else "a clean synthesis"


def _turn() -> DialecticTurn:
    return DialecticTurn(
        question="Does the exception apply?",
        thesis=Position(
            side="thesis",
            model="m",
            family="f",
            propositions=[CitationSlot(proposition="it applies")],
        ),
        antithesis=Position(
            side="antithesis",
            model="n",
            family="g",
            propositions=[CitationSlot(proposition="it does not apply")],
        ),
    )


def _chat(answers: list[str]) -> tuple[DialecticChat, _ScriptedClient]:
    client = _ScriptedClient(answers, name="gemma3:4b")
    chat = DialecticChat(
        thesis_client=_ScriptedClient([], name="saul:7b-instruct-v1"),
        antithesis_client=_ScriptedClient([], name="llama3.1:8b"),
        synthesis_client=client,
        max_regenerations=3,
    )
    return chat, client


def test_a_rejected_synthesis_is_told_what_tripped() -> None:
    """The retry must differ from the attempt that failed by more than a seed."""
    chat, client = _chat(["The rule from Bernstein v. Doe controls.", "clean prose"])

    synthesis, attempts = chat._generate_synthesis("q", _turn())

    assert len(client.prompts) == 2, "should have retried once"
    retry = client.prompts[1]
    assert "REJECTED" in retry, "the retry prompt must say the last answer was rejected"
    assert "Bernstein v. Doe" in retry, "and quote back what tripped the channel"
    assert synthesis == "clean prose"
    assert attempts == 1


def test_the_first_attempt_carries_no_feedback() -> None:
    """Nothing has been rejected yet; telling it otherwise would be a lie."""
    chat, client = _chat(["clean prose"])

    chat._generate_synthesis("q", _turn())

    assert len(client.prompts) == 1
    assert "REJECTED" not in client.prompts[0]


def test_the_budget_is_not_spent_repeating_one_unexplained_rejection() -> None:
    """The regression: three identical prompts, three identical failures.

    Every attempt naming authority still fails — the channel is not negotiable —
    but each retry must at least have been told why, so a model capable of
    complying gets the chance to.
    """
    named = "As held in Bernstein v. Doe, the exception applies."
    chat, client = _chat([named, named, named])

    synthesis, _ = chat._generate_synthesis("q", _turn())

    assert len(client.prompts) == 3
    assert "REJECTED" not in client.prompts[0]
    for prompt in client.prompts[1:]:
        assert "Bernstein v. Doe" in prompt
    assert "rejected" in synthesis.lower(), "exhausting the budget still reports why"


# --- positions: the same gap, one function up ---------------------------------
#
# `_generate_position` carries a comment saying feedback is "accumulated from a
# rejected attempt, fed back on the re-roll", and only the mirror branch did it.
# A position rejected for citations or for unreadable JSON was re-asked the
# identical question at a higher temperature.

_GOOD_JSON = (
    '{"propositions": [{"proposition": "The exception applies.", '
    '"court_hint": "circuit precedent on encryption source code", '
    '"weight": "controlling"}]}'
)
_NAMES_A_CASE = (
    '{"propositions": [{"proposition": "Bernstein v. Doe controls here.", '
    '"court_hint": "circuit precedent", "weight": "controlling"}]}'
)


def _position_chat(answers: list[str]) -> tuple[DialecticChat, _ScriptedClient]:
    client = _ScriptedClient(answers, name="saul:7b-instruct-v1")
    chat = DialecticChat(
        thesis_client=client,
        antithesis_client=_ScriptedClient([], name="llama3.1:8b"),
        synthesis_client=_ScriptedClient([], name="gemma3:4b"),
        max_regenerations=3,
    )
    return chat, client


def test_a_position_rejected_for_citations_is_told_which_string() -> None:
    chat, client = _position_chat([_NAMES_A_CASE, _GOOD_JSON])

    position, attempts = chat._generate_position("thesis", "q", client)

    assert len(client.prompts) == 2
    retry = client.prompts[1]
    assert "REJECTED" in retry
    assert "Bernstein v. Doe" in retry
    assert "court_hint" in retry, "and where the description belongs instead"
    assert attempts == 1
    assert position.propositions[0].proposition == "The exception applies."


def test_a_position_that_would_not_parse_is_told_why() -> None:
    """The branch that re-rolled most silently: no hits to quote, just a retry."""
    chat, client = _position_chat(["not json at all", _GOOD_JSON])

    chat._generate_position("thesis", "q", client)

    assert len(client.prompts) == 2
    retry = client.prompts[1]
    assert "REJECTED" in retry
    assert "propositions" in retry, "the expected shape must be restated"


def test_the_first_position_attempt_carries_no_feedback() -> None:
    chat, client = _position_chat([_GOOD_JSON])

    chat._generate_position("thesis", "q", client)

    assert len(client.prompts) == 1
    assert "REJECTED" not in client.prompts[0]
