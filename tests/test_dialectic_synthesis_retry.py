"""What the synthesis is told when its answer is rejected.

The loop re-rolls on temperature and seed. That is enough when the model was
merely unlucky, and useless when it was wrong in a way it cannot guess at: the
citation channel forbids naming authority, "Bernstein v. U.S. Dep" reads to a
model like ordinary legal prose, and being asked the identical question at a
higher temperature produces the identical mistake.

Observed on every dialectic run in the shipped configuration:
``citation string(s) rejected: ['Bernstein v. U.S. Dep']; after 3 attempts`` —
the whole regeneration budget spent, no synthesis returned, and the panel
showing the failure text where the answer should be.
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
