"""Eval harness for the dialectic module.

Runs a question set through :class:`DialecticChat`, applies the hard gates,
scores the surviving responses with an LLM judge, and aggregates per cluster.

**How the citation gate is measured here.** The dialectic module is forbidden
from emitting citations in prose — a proposition naming a case is discarded and
regenerated (OBSERVABLES rule 1). Authority lives in the *slot*: retrieval
proposes a candidate, CourtListener verification confirms it. So
"every authority must come through the legal research tool call" is enforced
against the resolved slots and a retrieval log, not against the text. A slot
carrying a citation that never appeared in the retrieval log is a gate failure,
which is exactly the fabrication the gate exists to catch.

Transport-agnostic: this module never imports ``legal_research.*``. The runner
script wires the clients and the corpus.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from modules.dialectic.copy import copy_position
from modules.dialectic.models import DialecticTurn, SlotStatus

# --------------------------------------------------------------------------- #
# Eval set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    cluster: str
    question: str
    frame: str
    must_engage: tuple[str, ...]
    holdout: bool
    central_crux: bool = False

    def prompt(self) -> str:
        """The text handed to the module: question plus its forum frame."""
        if not self.frame:
            return self.question
        return f"{self.question}\n\nFrame: {self.frame}"


@dataclass(frozen=True)
class EvalSet:
    name: str
    clusters: dict[str, str]
    questions: tuple[EvalQuestion, ...]

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        declared = set(data.get("holdout_ids", []))
        questions = tuple(
            EvalQuestion(
                id=q["id"],
                cluster=q["cluster"],
                question=q["question"],
                frame=q.get("frame", ""),
                must_engage=tuple(q["must_engage"]),
                holdout=bool(q.get("holdout")),
                central_crux=bool(q.get("central_crux")),
            )
            for q in data["questions"]
        )
        flagged = {q.id for q in questions if q.holdout}
        if flagged != declared:
            # The holdout list is the whole point of holding out; a silent
            # mismatch would leak a holdout into an optimization loop.
            raise ValueError(
                f"holdout mismatch: flagged={sorted(flagged)} declared={sorted(declared)}"
            )
        return cls(name=data["name"], clusters=data["clusters"], questions=questions)

    def select(self, *, include_holdout: bool = False) -> tuple[EvalQuestion, ...]:
        """Questions to run. Holdouts are excluded unless explicitly requested."""
        if include_holdout:
            return self.questions
        return tuple(q for q in self.questions if not q.holdout)


# --------------------------------------------------------------------------- #
# Retrieval log
# --------------------------------------------------------------------------- #


class LoggingRetriever:
    """Wrap a ``CiteRetriever`` and record every candidate it proposed.

    The gate needs to know what retrieval actually returned. Wrapping is enough
    — the engine calls ``propose`` and nothing else — so this needs no change to
    the module.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.log: list[dict[str, Any]] = []

    def annotate(self, cite: str) -> str:
        """Forward the optional annotator capability to the wrapped retriever.

        A decorator that silently drops an optional capability is worse than one
        that never had it: the engine probes with ``getattr(retriever,
        "annotate", None)``, so wrapping turned annotation off and every
        non-operative authority reached the synthesis unflagged. That is how the
        temporal gate failed three runs in a row while the module itself was
        correct.
        """
        inner = getattr(self._inner, "annotate", None)
        return str(inner(cite) or "") if callable(inner) else ""

    def propose(self, court_hint: str, proposition: str) -> list[str]:
        candidates = self._inner.propose(court_hint, proposition)
        self.log.append(
            {
                "court_hint": court_hint,
                "proposition": proposition,
                "candidates": list(candidates),
            }
        )
        return list(candidates)

    @property
    def retrieved_cites(self) -> set[str]:
        return {c for entry in self.log for c in entry["candidates"]}

    def reset(self) -> None:
        self.log.clear()


# --------------------------------------------------------------------------- #
# Hard gates
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    id: str
    passed: bool
    detail: str


def _slots(turn: DialecticTurn) -> list[Any]:
    return [*turn.thesis.propositions, *turn.antithesis.propositions]


def gate_citation_integrity(
    turn: DialecticTurn,
    retrieved: set[str],
    corpus_verified: set[str] | None = None,
) -> GateResult:
    """Every citation on a slot must have come through retrieval and verified.

    Two distinct failures are reported separately because they mean different
    things: a cite that never appeared in the retrieval log is fabricated, while
    a cite that was retrieved but not verified is merely unconfirmed.

    ``corpus_verified`` holds citations whose authority CourtListener cannot
    adjudicate — statutes, regulations, Federal Register documents. Requiring a
    CourtListener cluster for ``15 C.F.R. 734.7`` would fail every response that
    correctly relies on it, so for those the human-checked corpus is the
    verifier and retrieval-log membership is the integrity test.
    """
    corpus_verified = corpus_verified or set()
    fabricated: list[str] = []
    unverified: list[str] = []
    for slot in _slots(turn):
        cite = slot.normalized_cite
        if not cite:
            continue
        if cite not in retrieved:
            fabricated.append(cite)
        elif cite in corpus_verified:
            continue  # verified by the corpus; CourtListener has no jurisdiction
        elif slot.status != SlotStatus.VERIFIED:
            unverified.append(f"{cite} ({slot.status.value})")

    if fabricated:
        return GateResult(
            "citation_integrity",
            False,
            f"cite(s) not in the retrieval log: {fabricated}",
        )
    if unverified:
        return GateResult(
            "citation_integrity",
            False,
            f"retrieved but not verified: {unverified}",
        )
    return GateResult(
        "citation_integrity",
        True,
        f"{sum(1 for s in _slots(turn) if s.normalized_cite)} cite(s), all retrieved and verified",
    )


#: Phrases that count as acknowledging an authority is no longer operative.
_NON_OPERATIVE_MARKERS = (
    "rescinded",
    "superseded",
    "no longer operative",
    "no longer in force",
    "not currently operative",
    "not operative",
    "repealed",
    "withdrawn",
    "never took effect",
    "no longer good law",
)


def gate_temporal_validity(
    turn: DialecticTurn, status_by_cite: dict[str, tuple[str, str]]
) -> GateResult:
    """A non-operative authority must not be presented as operative law.

    ``status_by_cite`` maps a normalized cite to ``(status, note)`` from the
    corpus.

    **This reads model-authored prose only** — the propositions and the
    synthesis — and deliberately ignores ``slot.note``. The retrieval stage
    writes the status into that note, so a gate that read it would be checking
    that our own machinery wrote a string, not that the response understood the
    rule was repealed. It would pass every time and measure nothing.

    In practice this tests the synthesis, because propositions are generated
    before retrieval runs and cannot know which authority they were given. That
    is the honest scope of the check: the synthesis is the only role that sees
    the status and can act on it.
    """
    offenders: list[str] = []
    checked = 0
    # Model-authored text only. `slot.note` is machinery-authored and excluded.
    authored = " ".join(
        [*(slot.proposition for slot in _slots(turn)), turn.synthesis]
    ).lower()

    for slot in _slots(turn):
        cite = slot.normalized_cite
        if not cite or cite not in status_by_cite:
            continue
        status, _ = status_by_cite[cite]
        if status == "in_force":
            continue
        checked += 1
        if not any(marker in authored for marker in _NON_OPERATIVE_MARKERS):
            offenders.append(
                f"{cite} is {status} but no model-authored text says so"
            )

    if offenders:
        return GateResult("temporal_validity", False, "; ".join(dict.fromkeys(offenders)))
    return GateResult(
        "temporal_validity",
        True,
        f"{checked} non-operative cite(s) acknowledged in model-authored text"
        if checked
        else "no non-operative authority cited",
    )


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

CRITERIA = (
    "steelmanning",
    "authority_hierarchy",
    "counterargument_anticipation",
    "crux_identification",
    "synthesis_discipline",
)

_JUDGE_PROMPT = (
    "You are a strict evaluator of legal dialectic responses.\n"
    "Score the response on each criterion from 1 to 10.\n"
    "  steelmanning                 - both sides argued at the strength competent counsel would bring\n"
    "  authority_hierarchy          - binding vs persuasive, correct forum, good-law status\n"
    "  counterargument_anticipation - including procedural, standing, and ripeness attacks\n"
    "  crux_identification          - isolates the genuinely dispositive question\n"
    "  synthesis_discipline         - conclusion follows from the weighing; honest uncertainty where authority is split\n"
    "Be harsh. A response that merely restates the question scores 1-2. Reserve 9-10 "
    "for work a practising appellate lawyer would sign.\n"
    "Respond ONLY with a JSON object:\n"
    '{"scores": {"steelmanning": N, "authority_hierarchy": N, '
    '"counterargument_anticipation": N, "crux_identification": N, '
    '"synthesis_discipline": N}, "rationale": "..."}\n'
    "Emit no prose outside the JSON object."
)


class JudgeClient(Protocol):
    name: str

    def chat(self, messages: list[Any], config: Any | None = None) -> str: ...


@dataclass
class JudgeVerdict:
    scores: dict[str, int]
    rationale: str
    parsed: bool

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores.values()) if self.scores else 0.0


_JUDGE_RETRY = (
    "\n\nYour previous reply was REJECTED: it did not contain the required JSON "
    "object. Do not describe, transcribe, or restructure the response above. "
    "Emit only:\n"
    '{"scores": {"steelmanning": N, "authority_hierarchy": N, '
    '"counterargument_anticipation": N, "crux_identification": N, '
    '"synthesis_discipline": N}, "rationale": "..."}'
)


def render_for_judge(question: EvalQuestion, turn: DialecticTurn) -> str:
    """Render a turn for scoring, without inviting the judge to copy its shape.

    `copy_exchange` is written for a human reader: it numbers each crux and
    renders it as ``7. resolvable by authority (winner: antithesis; nli: model)``,
    which is close enough to JSON that a judge asked for JSON transcribes it
    instead of scoring. Question D1, whose seven cruxes make the largest table in
    the set, failed identically in three consecutive runs by echoing exactly
    those field names back.

    So the crux table is given as prose here, it appears once rather than twice
    (`copy_exchange` already contains it, and it was being appended again), and
    the ledger trailers are dropped — call counts have no bearing on the score.
    """
    cruxes = "no direct contradiction was identified between the two positions"
    if turn.cruxes:
        lines = []
        for crux in turn.cruxes:
            rank = "outcome-bearing" if crux.outcome_bearing else "not outcome-bearing"
            winner = "neither side outranks" if crux.winner == "none" else f"favours the {crux.winner}"
            lines.append(
                f"  Thesis argues {crux.thesis_prop.proposition!r} while the antithesis "
                f"argues {crux.antithesis_prop.proposition!r}. This is {crux.partition}, "
                f"{winner}, and is {rank}."
            )
        cruxes = f"{len(turn.cruxes)} contradiction(s) were identified:\n" + "\n".join(lines)

    return (
        f"QUESTION: {question.prompt()}\n\n"
        "DOCTRINAL ANCHORS A COMPETENT ANSWER SHOULD ENGAGE:\n"
        + "\n".join(f"  - {m}" for m in question.must_engage)
        + f"\n\nTHESIS:\n{copy_position(turn, 'thesis')}"
        + f"\n\nANTITHESIS:\n{copy_position(turn, 'antithesis')}"
        + f"\n\nSYNTHESIS:\n{turn.synthesis}"
        + f"\n\nCRUXES: {cruxes}"
    )


def judge_response(
    client: JudgeClient,
    question: EvalQuestion,
    turn: DialecticTurn,
    *,
    retries: int = 2,
) -> JudgeVerdict:
    """Score one response, retrying with a correction when the reply will not parse.

    The debaters already get a retry when their output is rejected; the judge
    got one shot, so a single malformed reply discarded the question entirely.
    """
    rendered = render_for_judge(question, turn)
    last = JudgeVerdict({}, "judge was never called", parsed=False)

    for attempt in range(max(1, retries)):
        content = rendered if attempt == 0 else rendered + _JUDGE_RETRY
        try:
            raw = client.chat(
                [
                    {"role": "system", "content": _JUDGE_PROMPT},
                    {"role": "user", "content": content},
                ],
                # Re-roll rather than re-run: at temperature 0 a rejected reply
                # reproduces verbatim, which is how D1 failed three times.
                config={"temperature": 0.0 if attempt == 0 else 0.5, "seed": 7 + attempt},
            )
        except Exception as exc:  # noqa: BLE001 - a broken judge is a result, not a crash
            last = JudgeVerdict({}, f"judge call failed: {exc}", parsed=False)
            continue
        last = parse_verdict(raw)
        if last.parsed:
            return last
    return last


def parse_verdict(raw: str) -> JudgeVerdict:
    """Strictly parse the judge's JSON. Anything unexpected is a parse failure."""
    text = raw.strip()
    if text.startswith("```"):
        import re

        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return JudgeVerdict({}, f"unparseable judge output: {raw[:160]!r}", parsed=False)
    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        return JudgeVerdict({}, f"missing scores object: {raw[:160]!r}", parsed=False)

    scores: dict[str, int] = {}
    for name in CRITERIA:
        value = data["scores"].get(name)
        if not isinstance(value, int | float) or isinstance(value, bool):
            return JudgeVerdict({}, f"criterion {name!r} missing or non-numeric", parsed=False)
        if not 1 <= value <= 10:
            return JudgeVerdict({}, f"criterion {name!r} out of range: {value}", parsed=False)
        scores[name] = int(value)
    return JudgeVerdict(scores, str(data.get("rationale", "")), parsed=True)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class QuestionResult:
    id: str
    cluster: str
    holdout: bool
    gates: list[GateResult]
    verdict: JudgeVerdict
    #: Set when the question could not be run at all (timeout, transport error).
    #: Distinct from a gate failure: nothing was measured, so nothing is known.
    error: str = ""
    #: Module diagnostics. Recorded because a scored result alone cannot explain
    #: a change in cost: enforcing disclosure of non-operative authority doubled
    #: run time, and nothing in the output said which questions were retrying.
    regenerated: int = 0
    crux_count: int = 0
    synthesis_note: str = ""

    @property
    def gates_passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    @property
    def scored(self) -> bool:
        """False when nothing usable came back and the gates did not fail.

        That combination is *missing data*, not a zero: nothing is known about
        the response's quality. Only a gate failure is a real zero.
        """
        if self.error:
            return False
        return self.gates_passed is False or self.verdict.parsed

    @property
    def score(self) -> float | None:
        """0 for a failed hard gate, ``None`` when nothing was measured.

        Collapsing a broken judge or a crashed question to 0.0 silently reports
        it as a terrible response. On the first full run that conflation moved
        cluster D from 7.30 to 5.84 and turned the best cluster into the worst.
        """
        if self.error:
            return None
        if not self.gates_passed:
            return 0.0
        if not self.verdict.parsed:
            return None
        return self.verdict.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cluster": self.cluster,
            "holdout": self.holdout,
            "scored": self.scored,
            "error": self.error,
            "regenerated": self.regenerated,
            "crux_count": self.crux_count,
            "synthesis_note": self.synthesis_note,
            "gates": {
                "pass": self.gates_passed,
                "detail": {g.id: {"pass": g.passed, "detail": g.detail} for g in self.gates},
            },
            "scores": [self.verdict.scores.get(c, 0) for c in CRITERIA],
            "score_names": list(CRITERIA),
            "mean": None if self.score is None else round(self.score, 3),
            "judge_parsed": self.verdict.parsed,
            "judge_rationale": self.verdict.rationale,
        }


@dataclass
class RunReport:
    eval_set: str
    results: list[QuestionResult] = field(default_factory=list)
    #: Runs sharing a round id are averaged before the plateau rule is applied.
    round_id: str = ""

    @property
    def _scored(self) -> list[QuestionResult]:
        """Results carrying a usable score. Judge failures are excluded."""
        return [r for r in self.results if r.score is not None]

    @property
    def overall_mean(self) -> float:
        vals = [r.score for r in self._scored if r.score is not None]
        return statistics.fmean(vals) if vals else 0.0

    def cluster_means(self) -> dict[str, float]:
        by: dict[str, list[float]] = {}
        for r in self._scored:
            if r.score is not None:
                by.setdefault(r.cluster, []).append(r.score)
        return {k: round(statistics.fmean(v), 3) for k, v in sorted(by.items())}

    def gate_failures(self) -> list[str]:
        return [r.id for r in self.results if not r.gates_passed]

    def retry_cost(self) -> dict[str, Any]:
        """How much regeneration this run spent, and where.

        A retry is a full generation, so this is the difference between a fast
        run and a slow one. `undisclosed` counts questions kept despite never
        disclosing a non-operative authority, which is the enforcement giving up.
        """
        regen = [r.regenerated for r in self.results]
        return {
            "total": sum(regen),
            "questions_with_retries": sum(1 for r in regen if r),
            "max": max(regen, default=0),
            "undisclosed": [r.id for r in self.results if r.synthesis_note],
        }

    def unscored(self) -> list[str]:
        """Questions with no usable score: missing data, never a zero."""
        return [r.id for r in self.results if r.score is None]

    def errors(self) -> dict[str, str]:
        """Questions that could not be run at all, with the reason."""
        return {r.id: r.error for r in self.results if r.error}

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_set": self.eval_set,
            "round_id": self.round_id,
            "overall_mean": round(self.overall_mean, 3),
            "scored_count": len(self._scored),
            "total_count": len(self.results),
            "cluster_means": self.cluster_means(),
            "gate_failures": self.gate_failures(),
            "unscored_judge_failed": self.unscored(),
            "retry_cost": self.retry_cost(),
            "run_errors": self.errors(),
            "questions": [r.to_dict() for r in self.results],
        }


#: Threshold for the plateau rule when a round is the average of several runs.
#: Averaging is what makes the originally specified 0.2 usable: a citation gate
#: flip moves a single run's mean by about 0.24, but only 0.24/N on a round of N
#: runs, and ordinary drift falls as 1/sqrt(N). At N=3 that is roughly 0.08 plus
#: 0.05, comfortably inside 0.2. Measuring rather than assuming this needs two
#: rounds on identical code -- six runs -- which has not been done yet.
ROUND_PLATEAU_DELTA = 0.2

#: Threshold when comparing single runs. Raised from the specified 0.2 after measuring
#: the noise floor: three runs of unchanged code moved by 0.087 in one set and
#: 0.219 in another, the difference being whether a question's citation gate
#: happened to flip. A flip costs about 0.24 on a 32-question mean and two would
#: cost 0.48, so a threshold under 0.5 only holds on runs where none lands, and
#: would report convergence that never happened. See REMEDIATION 11.11.
PLATEAU_DELTA = 0.5
PLATEAU_ROUNDS = 3


def round_means(runs: Sequence[tuple[str, float]], *, per_round: int = 3) -> list[float]:
    """Collapse per-run means into per-round means.

    ``runs`` is ``(round_id, mean)`` in run order. Runs carrying a round id are
    grouped by it; runs without one are chunked by *per_round* in order, which
    is what a set produced before round ids existed looks like.
    """
    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    unlabelled: list[float] = []
    for round_id, mean in runs:
        if not round_id:
            unlabelled.append(mean)
            continue
        if round_id not in grouped:
            grouped[round_id] = []
            order.append(round_id)
        grouped[round_id].append(mean)

    means = [statistics.fmean(grouped[r]) for r in order]
    for i in range(0, len(unlabelled), per_round):
        chunk = unlabelled[i : i + per_round]
        if chunk:
            means.append(statistics.fmean(chunk))
    return means


def has_plateaued(
    means: Sequence[float], *, delta: float = PLATEAU_DELTA, rounds: int = PLATEAU_ROUNDS
) -> bool:
    """True when the overall mean moved < *delta* across *rounds* consecutive runs.

    Needs ``rounds + 1`` observations to see ``rounds`` changes.
    """
    if len(means) < rounds + 1:
        return False
    recent = means[-(rounds + 1) :]
    # Pairing a sequence with its own tail is deliberately offset by one, so
    # strict= must stay False here.
    pairs = zip(recent, recent[1:], strict=False)
    return all(abs(b - a) < delta for a, b in pairs)
