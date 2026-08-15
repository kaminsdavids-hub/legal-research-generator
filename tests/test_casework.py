"""Divergence, Portfolio, and the coupling between them.

The governing property both modules rest on is that their fitness functions are
*executed, not judged*. Evolutionary search finds whatever an objective actually
rewards, so a judged objective would be optimised by rewording rather than by
finding hard cases. Several tests below exist only to hold that line: no model
in a fitness loop, no unverified authority in a pool, no infeasible individual
ever scored.
"""

from __future__ import annotations

import random

import pytest

from modules.casework.corpus import CaseCorpus, CodedCase, Coding
from modules.casework.schema import (
    WILDCARDS,
    Axis,
    AxisKind,
    EvaluatorStamp,
    FeatureSchema,
    Outcome,
    Scrutiny,
    Verdict,
    load_schema,
)
from modules.casework.search import ArchiveMismatch, Elite, EliteArchive
from modules.coupling.events import EpochManager, EventKind, EventLog, regions_from_priors
from modules.divergence.genotype import FactVector, FeasibilityMask, decode, encode
from modules.divergence.objectives import brittleness, disagreement, realism_prior, score
from modules.divergence.precedent import match_precedents
from modules.divergence.rules import load_rules
from modules.portfolio.model import (
    Authority,
    CitatorFlag,
    PortfolioProblem,
    Precedential,
    Proposition,
    UnverifiedAuthority,
)
from modules.portfolio.nsga import named_strategies, select
from modules.portfolio.objectives import score_selection, thin_coverage

SCHEMA_PATH = "data/feature_schema.yaml"
RULES_PATH = "data/rules"


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return load_schema(SCHEMA_PATH)


@pytest.fixture(scope="module")
def rules():
    return load_rules(RULES_PATH)


def _mask(schema: FeatureSchema) -> FeasibilityMask:
    mask = FeasibilityMask(schema)
    mask.require(
        "alignment removal presupposes weights or code",
        lambda f: f["post_release_modification"] != "alignment_removal"
        or f["distribution_modality"] in ("weights", "training_code"),
    )
    mask.require(
        "a paper carries no capability tier above 3",
        lambda f: f["distribution_modality"] != "paper" or int(f["capability_tier"]) <= 3,
    )
    return mask


def _facts(schema: FeatureSchema, **overrides: str) -> FactVector:
    base = {axis.name: axis.values[0] for axis in schema.axes}
    return FactVector({**base, **overrides})


# --------------------------------------------------------------------------- #
# The feasibility mask: a fuzzer, because the property is universal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_sampling_never_produces_an_infeasible_vector(schema: FeatureSchema, seed: int) -> None:
    """Infeasible individuals are never created — not created and repaired, and
    not created and penalised, which would let the search trade realism for
    disagreement."""

    mask = _mask(schema)
    rng = random.Random(seed)

    for _ in range(200):
        vector = mask.sample(rng)
        assert mask.violations(vector) == []


@pytest.mark.parametrize("seed", range(8))
def test_every_neighbour_is_feasible(schema: FeatureSchema, seed: int) -> None:
    """Brittleness walks neighbours, so an infeasible neighbour would put an
    incoherent fact pattern inside the objective."""

    mask = _mask(schema)
    rng = random.Random(seed)
    vector = mask.sample(rng)

    for neighbour in mask.neighbours(vector):
        assert mask.violations(neighbour) == []


def test_an_over_constrained_mask_raises_rather_than_returning_junk(schema: FeatureSchema) -> None:
    mask = FeasibilityMask(schema).require("impossible", lambda f: False)

    with pytest.raises(RuntimeError, match="over-constrained"):
        mask.sample(random.Random(1), attempts=50)


def test_neighbours_are_one_ordinal_step(schema: FeatureSchema) -> None:
    """"tier 2 -> 3" is a neighbour; "tier 2 -> 5" is not, or brittleness would
    call almost everything knife-edge."""

    mask = _mask(schema)
    vector = _facts(schema, capability_tier="3")
    tiers = {
        n["capability_tier"]
        for n in mask.neighbours(vector)
        if all(n[a.name] == vector[a.name] for a in schema.axes if a.name != "capability_tier")
    }

    assert tiers == {"2", "4"}


def test_encode_decode_round_trips(schema: FeatureSchema) -> None:
    vector = _mask(schema).sample(random.Random(3))

    assert decode(encode(vector, schema), schema) == vector


# --------------------------------------------------------------------------- #
# Objectives are computations
# --------------------------------------------------------------------------- #
def test_entropy_rewards_a_genuine_split_over_a_lopsided_one() -> None:
    """The reason for entropy rather than counting disagreeing pairs."""

    even = disagreement(["permitted", "permitted", "restricted", "restricted"])
    lopsided = disagreement(["permitted", "permitted", "permitted", "restricted"])
    unanimous = disagreement(["permitted"] * 4)

    assert even > lopsided > unanimous
    assert unanimous == 0.0


def test_scrutiny_is_part_of_the_verdict_identity() -> None:
    """Two readings that both permit, on different tiers, disagree about
    something a court would spend an opinion on."""

    strict = Verdict(Outcome.PERMITTED, Scrutiny.STRICT)
    rational = Verdict(Outcome.PERMITTED, Scrutiny.RATIONAL_BASIS)

    assert strict.label != rational.label
    assert disagreement([strict.label, rational.label]) > 0


def test_brittleness_finds_a_one_step_flip(schema: FeatureSchema, rules) -> None:
    mask = _mask(schema)
    # A paper at tier 3 is permitted by functional_artifact; weights are not.
    facts = _facts(schema, distribution_modality="paper", capability_tier="3")

    assert brittleness(facts, rules, mask, schema, max_radius=2) <= 2.0


def test_a_broken_rule_is_a_distinct_verdict_not_silent_agreement(schema: FeatureSchema) -> None:
    """A rule that raises must not look like consensus — that is the failure
    mode with the worst consequences here."""

    from modules.divergence.rules import LoadedRule, RuleSet

    def exploding(facts):  # noqa: ANN001, ANN202
        raise ValueError("bad predicate")

    ruleset = RuleSet(
        rules=[
            LoadedRule("ok", lambda f: Verdict(Outcome.PERMITTED), __file__, "r"),  # type: ignore[arg-type]
            LoadedRule("broken", exploding, __file__, "r"),  # type: ignore[arg-type]
        ],
        digest="test",
    )
    verdicts = ruleset.verdicts(_facts(schema))

    assert verdicts["broken"].outcome is Outcome.UNCERTAIN
    assert "ValueError" in verdicts["broken"].rationale
    assert disagreement(list(ruleset.labels(_facts(schema)))) > 0


def test_realism_uses_only_verified_codings(schema: FeatureSchema) -> None:
    """A prior built from guessed codings is a guess with a number on it."""

    facts = _facts(schema)
    draft = CaseCorpus([CodedCase("a", "1 F.3d 1", "A", facts.as_dict(), "permitted",
                                  Coding.MODEL_DRAFT)])
    verified = CaseCorpus([CodedCase("a", "1 F.3d 1", "A", facts.as_dict(), "permitted",
                                     Coding.HUMAN)])

    assert realism_prior(facts, draft, schema) == 0.0
    assert realism_prior(facts, verified, schema) == 1.0


# --------------------------------------------------------------------------- #
# Precedent collision: the module's most valuable output
# --------------------------------------------------------------------------- #
def test_an_unverified_coding_cannot_produce_a_collision(schema: FeatureSchema, rules) -> None:
    """The force of the finding is "your framework misclassifies a real
    decision"; that is worth nothing if the coding was itself guessed."""

    facts = _facts(schema, distribution_modality="weights", capability_tier="5")
    corpus = CaseCorpus(
        [CodedCase("draft", "1 F.3d 1", "Draft", facts.as_dict(), "permitted", Coding.MODEL_DRAFT)]
    )
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(schema, corpus, [(facts, scores, rules.verdicts(facts))])

    assert collision.precedents == []
    assert collision.unverified_candidates == ["draft"]


def test_a_verified_coding_produces_a_collision_and_names_the_contradiction(
    schema: FeatureSchema, rules
) -> None:
    facts = _facts(schema, distribution_modality="weights", capability_tier="5",
                   forum="public_repository")
    corpus = CaseCorpus(
        [
            CodedCase("real", "9 F.3d 9", "Real v. Case", facts.as_dict(), "permitted",
                      Coding.HUMAN, coded_by="author")
        ]
    )
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(schema, corpus, [(facts, scores, rules.verdicts(facts))])

    assert len(collision.precedents) == 1
    precedent = collision.precedents[0]
    assert precedent.case_id == "real"
    assert precedent.held == "permitted"
    # At least one reading says restricted about a case that came out permitted.
    assert precedent.contradicting


# --------------------------------------------------------------------------- #
# Portfolio: the invariant on the pool
# --------------------------------------------------------------------------- #
def _authority(aid: str, **kw) -> Authority:
    base = dict(
        record_id=kw.pop("record_id", aid),
        verified=kw.pop("verified", True),
        court_level=kw.pop("court_level", "appellate"),
        jurisdiction=kw.pop("jurisdiction", "us_federal"),
        year=kw.pop("year", 2020),
        precedential=kw.pop("precedential", Precedential.BINDING),
        citator=kw.pop("citator", CitatorFlag.GOOD),
    )
    return Authority(id=aid, **base, **kw)


def test_an_unverified_authority_can_never_enter_the_pool() -> None:
    """A hard invariant, not a preference: no weighting can produce a selection
    containing a citation the grounding gate did not accept."""

    with pytest.raises(UnverifiedAuthority, match="grounding gate"):
        PortfolioProblem(
            propositions=[Proposition("p1", "claim")],
            candidates={"p1": [_authority("a1"), _authority("a2", verified=False)]},
        )


def test_a_negative_citator_flag_excludes_rather_than_penalises() -> None:
    """An overruled case is not a worse citation; it is not a citation."""

    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={
            "p1": [_authority("good"), _authority("bad", citator=CitatorFlag.OVERRULED)]
        },
    )

    assert [a.id for a in problem.selectable("p1")] == ["good"]
    assert problem.excluded() == {"p1": ["bad"]}

    front = select(problem, front_size=4, population=12, generations=4)
    assert all("bad" not in [aid for _, aid in p.selection] for p in front)


def test_missing_citator_data_is_flagged_not_assumed_good() -> None:
    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={"p1": [_authority("unknown", citator=CitatorFlag.UNKNOWN)]},
    )

    front = select(problem, front_size=2, population=8, generations=3)

    assert front[0].manual_citator_checks == ["unknown"]


def test_every_proposition_is_covered_and_contested_ones_twice() -> None:
    problem = PortfolioProblem(
        propositions=[
            Proposition("p1", "ordinary"),
            Proposition("p2", "contested", contested=True),
        ],
        candidates={
            "p1": [_authority("a1")],
            "p2": [_authority("b1", record_id="rec-b"), _authority("b2", record_id="rec-c")],
        },
    )

    for portfolio in select(problem, front_size=6, population=16, generations=5):
        covered = {pid for pid, _ in portfolio.selection}
        assert covered == {"p1", "p2"}
        sources = {aid for pid, aid in portfolio.selection if pid == "p2"}
        assert len(sources) >= 2


def test_an_uncoverable_proposition_is_reported_not_worked_around() -> None:
    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={"p1": [_authority("only", citator=CitatorFlag.ABROGATED)]},
    )

    assert problem.infeasible() == [
        "p1: no verified authority survives citator exclusion"
    ]


def test_concentration_notices_a_portfolio_resting_on_one_case() -> None:
    problem = PortfolioProblem(
        propositions=[Proposition("p1", "a"), Proposition("p2", "b")],
        candidates={
            "p1": [_authority("x1", record_id="same")],
            "p2": [_authority("x2", record_id="same"), _authority("y", record_id="other")],
        },
    )

    one_source = score_selection(problem, [("p1", "x1"), ("p2", "x2")])
    two_sources = score_selection(problem, [("p1", "x1"), ("p2", "y")])

    assert one_source.concentration > two_sources.concentration


def test_named_strategies_do_not_present_one_portfolio_three_times() -> None:
    problem = PortfolioProblem(
        propositions=[Proposition("p1", "a")],
        candidates={"p1": [_authority("a1")]},
    )
    front = select(problem, front_size=4, population=8, generations=3)

    names = named_strategies(front)

    assert len({tuple(s.portfolio.selection) for s in names}) == len(names)


# --------------------------------------------------------------------------- #
# Coupling: the loop must not oscillate
# --------------------------------------------------------------------------- #
def test_a_signal_raised_this_epoch_is_invisible_to_this_epoch() -> None:
    """The damping, in one assertion. Divergence cannot chase a gap its own
    findings created moments earlier."""

    epochs = EpochManager(log=EventLog(), epoch=4)
    epochs.record_thin_coverage([{"proposition": "p1", "features": {"forum": "journal"}}])

    assert epochs.priors_for(4) == []
    assert len(epochs.priors_for(5)) == 1


def test_precedents_reach_portfolio_only_in_a_later_epoch() -> None:
    epochs = EpochManager(log=EventLog(), epoch=2)
    epochs.record_precedents([{"case_id": "real", "features": {}}])

    assert epochs.propositions_for(2) == []
    assert len(epochs.propositions_for(3)) == 1


def test_the_feedback_loop_converges_rather_than_oscillating() -> None:
    """Regression for the failure the damping exists to prevent: two modules
    each responding to the other's last output, elaborating one corner of the
    space forever.

    Simulated over ten epochs — the signal set must reach a fixed point rather
    than growing without bound.
    """

    epochs = EpochManager(log=EventLog(), epoch=0)
    sizes: list[int] = []

    for _ in range(10):
        epoch = epochs.open_epoch()
        priors = epochs.priors_for(epoch)
        # Divergence acts only on what earlier epochs raised, and emits one
        # finding per prior region (plus one of its own).
        epochs.record_precedents([{"case_id": f"c{epoch}", "features": {"forum": "journal"}}])
        # Portfolio reports the same region thin every time — the worst case.
        epochs.record_thin_coverage([{"proposition": "p1", "features": {"forum": "journal"}}])
        sizes.append(len(regions_from_priors(priors)))

    # Deduplication by region means the actionable prior set is bounded at one,
    # however many times the same signal is raised.
    assert max(sizes) <= 1
    assert sizes[-1] == sizes[-2]


def test_an_epoch_boundary_is_recorded_when_the_evaluator_changes() -> None:
    log = EventLog()
    epochs = EpochManager(log=log, epoch=3)

    new_epoch = epochs.evaluator_changed("schema-2", "rules-2")

    assert new_epoch == 4
    assert log.of_kind(EventKind.EVALUATOR_CHANGED)
    assert log.of_kind(EventKind.EPOCH_OPENED, epoch=4)


def test_thin_coverage_names_its_reason() -> None:
    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={
            "p1": [
                _authority(
                    "old", precedential=Precedential.PERSUASIVE, year=1950, relation="rule_support"
                )
            ]
        },
        current_year=2026,
        half_life_years=12.0,
    )

    findings = thin_coverage(problem, [("p1", "old")])

    assert findings[0]["proposition"] == "p1"
    assert "single source" in findings[0]["reasons"]
    assert "nothing binding" in findings[0]["reasons"]
    assert "all stale" in findings[0]["reasons"]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_an_archive_refuses_to_load_under_a_different_evaluator(tmp_path) -> None:
    """Two archives built by different rules describe different questions;
    merging them would average two experiments."""

    stamp = EvaluatorStamp("schema-1", "rules-1", epoch=1, seed=7)
    archive = EliteArchive(stamp=stamp, elites=[Elite({"a": "b"}, {"disagreement": 1.0})])
    path = tmp_path / "archive.json"
    archive.save(path)

    assert EliteArchive.load(path, expect=stamp).elites[0].genome == {"a": "b"}
    with pytest.raises(ArchiveMismatch, match="different evaluator"):
        EliteArchive.load(path, expect=EvaluatorStamp("schema-1", "rules-2", 1, 7))


def test_the_schema_digest_changes_when_the_schema_does(tmp_path) -> None:
    first = tmp_path / "a.yaml"
    first.write_text("axes:\n  - name: x\n    kind: categorical\n    values: [a, b]\n")
    second = tmp_path / "b.yaml"
    second.write_text("axes:\n  - name: x\n    kind: categorical\n    values: [a, b, c]\n")

    assert load_schema(first).digest != load_schema(second).digest


def test_an_ordinal_axis_measures_distance_in_steps() -> None:
    ordinal = Axis("tier", AxisKind.ORDINAL, ("1", "2", "3", "4", "5"))
    categorical = Axis("forum", AxisKind.CATEGORICAL, ("a", "b", "c"))

    assert ordinal.distance("1", "4") == 3
    assert categorical.distance("a", "c") == 1


# --------------------------------------------------------------------------- #
# Rendering: downstream of scoring, and provably so
# --------------------------------------------------------------------------- #
def test_no_scoring_module_imports_the_renderer() -> None:
    """The structural guarantee. If prose could influence a score, the search
    would optimise the prose — so the fitness path must not be able to reach
    this module at all."""

    from pathlib import Path

    for name in ("objectives.py", "mapelites.py", "rules.py", "genotype.py"):
        source = Path("modules/divergence") / name
        text = source.read_text(encoding="utf-8")
        assert "render" not in text.replace("rendered", "").replace("render_", ""), (
            f"{name} references the renderer; scoring must not reach prose"
        )


def test_the_renderer_imports_no_scorer() -> None:
    from pathlib import Path

    text = Path("modules/divergence/render.py").read_text(encoding="utf-8")

    assert "from .objectives" not in text
    assert "from .rules" not in text
    assert "from .mapelites" not in text


def test_without_a_model_the_template_ships_and_says_so() -> None:
    from modules.divergence.render import render_hypothetical

    result = render_hypothetical({"forum": "public_repository", "actor_type": "academic"})

    assert result.source == "template"
    assert "public repository" in result.prose
    assert any("no model configured" in w for w in result.warnings)


def test_a_rendered_citation_is_refused() -> None:
    """Nothing in a genotype carries authority, so a reporter cite in the output
    came from the model's weights."""

    from modules.divergence.render import RenderError, render_hypothetical

    class _Cites:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202
            return "A researcher posts weights, as in Junger v. Daley, 209 F.3d 481."

    with pytest.raises(RenderError, match="cites authority"):
        render_hypothetical({"forum": "public_repository"}, client=_Cites())


def test_prose_that_answers_the_question_is_refused() -> None:
    """The hypothetical poses the question; the frozen predicates answer it."""

    from modules.divergence.render import RenderError, render_hypothetical

    class _Concludes:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202
            return (
                "An academic posts model weights to a public repository. "
                "Therefore the publication is permitted under the exclusion."
            )

    with pytest.raises(RenderError, match="states an outcome"):
        render_hypothetical({"forum": "public_repository"}, client=_Concludes())


def test_an_unmentioned_axis_warns_rather_than_failing() -> None:
    """A fluent paragraph may render tier 5 as "frontier-scale" and be better
    prose for it — but the author is told which axes went unmentioned."""

    from modules.divergence.render import render_hypothetical

    class _Fluent:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202
            return "A university lab publishes a frontier-scale model openly online."

    result = render_hypothetical(
        {"capability_tier": "5", "actor_type": "academic"}, client=_Fluent()
    )

    assert result.source == "model"
    assert any("not visibly mentioned" in w for w in result.warnings)


def test_a_model_failure_does_not_cost_the_finding() -> None:
    """The fact pattern and its scores are the result; the paragraph is
    presentation."""

    from modules.divergence.render import render_hypothetical

    class _Broken:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202
            raise RuntimeError("endpoint down")

    result = render_hypothetical(
        {"forum": "journal"}, scores={"disagreement": 1.0}, client=_Broken()
    )

    assert result.source == "template"
    assert result.scores["disagreement"] == 1.0
    assert any("model unavailable" in w for w in result.warnings)


def test_verdicts_sit_beside_the_prose_not_inside_it() -> None:
    from modules.divergence.render import render_hypothetical

    result = render_hypothetical(
        {"forum": "journal"},
        scores={"disagreement": 1.0, "brittleness": 1.0},
        verdicts=("permitted", "restricted"),
    )
    rendered = result.render()

    assert "Readings split: permitted, restricted" in rendered
    assert "permitted" not in result.prose


# --------------------------------------------------------------------------- #
# The report package
# --------------------------------------------------------------------------- #
def _archive(elites, stamp=None):
    return EliteArchive(
        stamp=stamp or EvaluatorStamp("s1", "r1", epoch=2, seed=7, recipe="divergence search ..."),
        elites=elites,
    )


def test_every_report_carries_a_reproduction_block() -> None:
    """The first question about a surprising finding is whether it came from the
    rules you are looking at now."""

    from modules.casework.report import divergence_report

    report = divergence_report(_archive([Elite({"forum": "journal"}, {"disagreement": 1.0,
                                                                     "brittleness": 1.0})]))
    text = report.render()

    assert "reproduction" in text
    assert "s1" in text and "r1" in text
    assert "epoch   2" in text and "seed    7" in text
    assert "divergence search" in text


def test_an_unverified_corpus_is_a_read_first_caveat() -> None:
    """The module's headline finding is switched off, and the report says so at
    the top rather than leaving an author to notice zero collisions."""

    from modules.casework.report import divergence_report

    report = divergence_report(_archive([]), corpus_size=4, verified_codings=0)

    assert any("0 verified" in c for c in report.caveats)
    assert "READ FIRST" in report.render()


def test_a_verified_corpus_raises_no_such_caveat() -> None:
    from modules.casework.report import divergence_report

    report = divergence_report(_archive([]), corpus_size=40, verified_codings=40)

    assert not any("verified" in c for c in report.caveats)


def test_a_unanimous_archive_is_flagged_as_suspicious() -> None:
    """If the readings never split anywhere the search reached, either the
    predicates encode the same reading or the mask excludes the interesting
    region — both are the author's problem to know about."""

    from modules.casework.report import divergence_report

    flat = [Elite({"forum": "journal"}, {"disagreement": 0.0, "brittleness": 3.0})]
    report = divergence_report(_archive(flat))

    assert any("unanimous" in c for c in report.caveats)


def test_the_portfolio_report_flags_missing_citator_data() -> None:
    from modules.casework.report import portfolio_report

    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={"p1": [_authority("unknown", citator=CitatorFlag.UNKNOWN)]},
    )
    front = select(problem, front_size=2, population=8, generations=3)
    report = portfolio_report(
        front,
        named_strategies(front),
        stamp=EvaluatorStamp("n/a", "n/a", 1, 7),
        thin=[],
    )

    assert any("NOT assumed good" in c for c in report.caveats)
    assert "MANUAL_CITATOR_CHECK" in report.render()


def test_the_portfolio_report_says_the_choice_is_editorial() -> None:
    from modules.casework.report import portfolio_report

    problem = PortfolioProblem(
        propositions=[Proposition("p1", "claim")],
        candidates={"p1": [_authority("a1")]},
    )
    front = select(problem, front_size=2, population=8, generations=3)

    text = portfolio_report(
        front, named_strategies(front), stamp=EvaluatorStamp("n/a", "n/a", 1, 7)
    ).render()

    assert "[stated]" in text
    assert "not ranked" in text


def test_a_report_round_trips_to_json(tmp_path) -> None:
    from modules.casework.report import divergence_report

    report = divergence_report(
        _archive([Elite({"forum": "journal"}, {"disagreement": 1.0, "brittleness": 1.0})])
    )
    path = tmp_path / "report.json"
    report.save(path)

    import json

    data = json.loads(path.read_text())
    assert data["stamp"]["rules_digest"] == "r1"
    assert data["sections"]


def test_holdout_notice_warns_about_an_unvalidated_tuned_weight() -> None:
    """A weight fitted on this manuscript and reported on this manuscript is a
    weight fitted to noise."""

    from modules.casework.report import holdout_notice

    assert "nothing to hold out" in holdout_notice({}, [])
    assert "WARNING" in holdout_notice({"brittleness_weight": 0.25}, [])
    assert "validated on 5" in holdout_notice({"brittleness_weight": 0.25}, [1, 2, 3, 4, 5])


# --------------------------------------------------------------------------- #
# graph_adapter: where a finding becomes something the manuscript must answer
# --------------------------------------------------------------------------- #
def _precedent(case_id="real", coded_by="author", contradicting=("expressive_code",)):
    from modules.divergence.precedent import MishandledPrecedent

    return MishandledPrecedent(
        case_id=case_id,
        citation="209 F.3d 481",
        name="Junger v. Daley",
        distance=0,
        facts={"forum": "public_repository"},
        held="permitted",
        verdicts=("permitted", "restricted"),
        scores={"disagreement": 1.0},
        coded_by=coded_by,
        contradicting=tuple(contradicting),
    )


def _graph_with_thesis():
    from modules.maieutic.graph import ArgumentGraph, Node, NodeType, Provenance

    thesis = Node(
        id="thesis",
        type=NodeType.THESIS,
        text="Publishing open model weights is not a deemed export.",
        provenance=Provenance.HUMAN,
    )
    return ArgumentGraph(nodes={thesis.id: thesis}, edges=[])


def test_a_precedent_becomes_an_objection_the_paper_must_answer() -> None:
    """Filed as an OBJECTION so it lands under the graph's existing machinery:
    an objection with no reply is an UNANSWERED_ATTACK, which the Socratic
    engine ranks second only to a self-grounding cycle."""

    from modules.casework.graph_adapter import emit_precedents
    from modules.maieutic.graph import EdgeType, NodeType, Provenance
    from modules.maieutic.socratic import GapKind, analyse

    graph = _graph_with_thesis()
    emission = emit_precedents(graph, [_precedent()], epoch=3, attach_to="thesis")

    assert len(emission.added) == 1
    node = graph.nodes[emission.added[0]]
    assert node.type is NodeType.OBJECTION
    assert "Junger v. Daley" in node.text
    assert any(e.type is EdgeType.ATTACKS and e.source == node.id for e in graph.edges)
    # And the loop now asks about it.
    assert GapKind.UNANSWERED_ATTACK in {gap.kind for gap in analyse(graph)}
    assert node.provenance is Provenance.SYSTEM


def test_an_emitted_node_is_never_the_authors() -> None:
    """The graph reserves HUMAN for the answer-capture and edit paths; a search
    result wearing it would corrupt the integrity record."""

    from modules.casework.graph_adapter import emit_precedents, emitted_nodes
    from modules.maieutic.graph import Provenance

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent()], epoch=1)

    assert all(n.provenance is Provenance.SYSTEM for n in emitted_nodes(graph))


def test_an_unverified_coding_emits_nothing() -> None:
    """Enforced again at the layer that writes: this is the one that would put
    a guess into the manuscript."""

    from modules.casework.graph_adapter import emit_precedents

    graph = _graph_with_thesis()
    emission = emit_precedents(graph, [_precedent(coded_by="")], epoch=1)

    assert emission.added == []
    assert emission.refused == ["real"]
    assert len(graph.nodes) == 1


def test_emission_is_idempotent_across_epochs() -> None:
    """A loop that appended a node per epoch would grow the manuscript on every
    run."""

    from modules.casework.graph_adapter import emit_precedents

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent()], epoch=1, attach_to="thesis")
    second = emit_precedents(graph, [_precedent()], epoch=2, attach_to="thesis")

    assert second.added == []
    assert len(second.already_present) == 1
    assert len(graph.nodes) == 2


def test_an_unattached_emission_is_visible_rather_than_guessed() -> None:
    """Guessing which claim a case undermines would be the machine making an
    argument. Unattached, it surfaces as an orphan instead."""

    from modules.casework.graph_adapter import emit_precedents
    from modules.maieutic.socratic import GapKind, analyse

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent()], epoch=1, attach_to=None)

    assert GapKind.ORPHANED_NODE in {gap.kind for gap in analyse(graph)}


def test_portfolio_reads_its_requirements_from_the_graph() -> None:
    """The second half of the coupling: neither module imports the other, and
    the manuscript is the only thing they share."""

    from modules.casework.graph_adapter import emit_precedents, propositions_for_portfolio

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent()], epoch=1, attach_to="thesis")

    propositions = propositions_for_portfolio(graph)
    from_search = [p for p in propositions if p["from_divergence"]]

    assert len(from_search) == 1
    # An objection built from a decided case inherits the two-source rule.
    assert from_search[0]["contested"] is True
    assert all(p["contested"] is False for p in propositions if not p["from_divergence"])


def test_the_propositions_load_into_a_portfolio_problem() -> None:
    """End of the edge: a Divergence finding is a coverage requirement."""

    from modules.casework.graph_adapter import emit_precedents, propositions_for_portfolio

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent()], epoch=1, attach_to="thesis")
    specs = propositions_for_portfolio(graph)

    problem = PortfolioProblem(
        propositions=[
            Proposition(p["id"], p["text"], contested=p["contested"]) for p in specs
        ],
        candidates={
            p["id"]: [
                _authority(f"{p['id']}-a", record_id="rec-1"),
                _authority(f"{p['id']}-b", record_id="rec-2"),
            ]
            for p in specs
        },
    )
    front = select(problem, front_size=4, population=16, generations=5)

    covered = {pid for pid, _ in front[0].selection}
    assert covered == {p["id"] for p in specs}


def test_unanswered_emissions_are_countable() -> None:
    """The measure of whether the coupling did any good."""

    from modules.casework.graph_adapter import emit_precedents, unanswered_emissions

    graph = _graph_with_thesis()
    emit_precedents(graph, [_precedent("a"), _precedent("b")], epoch=1, attach_to="thesis")

    assert len(unanswered_emissions(graph)) == 2


# --------------------------------------------------------------------------- #
# The shipped case corpus
# --------------------------------------------------------------------------- #
def test_the_shipped_corpus_fits_the_shipped_schema(schema: FeatureSchema) -> None:
    from modules.casework.corpus import load_cases

    corpus = load_cases("data/cases.yaml")

    assert len(corpus.cases) >= 15
    assert corpus.validate_against(schema) == []


def test_no_shipped_coding_claims_to_be_verified() -> None:
    """Every record is a draft. Marking one verified without a human reading the
    case would defeat the only guard the precedent path has."""

    from modules.casework.corpus import load_cases

    corpus = load_cases("data/cases.yaml")

    assert corpus.verified() == []
    assert all(case.coded_by == "" for case in corpus.cases)


def test_citation_provenance_is_recorded_separately_from_coding() -> None:
    """A citation confirmed against a primary source is a fact; the coding is a
    judgement. A record must not let the first stand in for the second."""

    from modules.casework.corpus import load_cases

    corpus = load_cases("data/cases.yaml")
    confirmed = [c for c in corpus.cases if c.citation_source.startswith("cap")]
    unconfirmed = [c for c in corpus.cases if not c.citation_source.startswith("cap")]

    assert len(confirmed) >= 14
    # The records whose citations could not be machine-confirmed say so.
    assert all("not-in-cap" in c.citation_source for c in unconfirmed)
    assert all("machine-confirmed" in c.note.lower() for c in unconfirmed)


def test_the_withdrawn_opinion_is_flagged_in_its_note() -> None:
    """Bernstein's panel opinion was withdrawn; a corpus that coded it silently
    would feed a collision built on an un-issued decision."""

    from modules.casework.corpus import load_cases

    bernstein = next(
        c for c in load_cases("data/cases.yaml").cases if c.id == "bernstein-9th-1999"
    )

    assert "WITHDRAWN" in bernstein.note
    assert "192 F.3d 1308" in bernstein.note


def test_the_corpus_still_produces_no_collisions(schema: FeatureSchema, rules) -> None:
    """The end-to-end consequence of an all-draft corpus, asserted rather than
    assumed: 17 coded cases and the precedent path is still dark."""

    from modules.casework.corpus import load_cases
    from modules.divergence.precedent import match_precedents

    corpus = load_cases("data/cases.yaml")
    junger = next(c for c in corpus.cases if c.id == "junger-6th-2000")
    # Junger's own answered axes, exactly. Since it leaves two blank it gets no
    # tolerance, so nothing looser would reach it — which is the point of the
    # per-case allowance, and means this asserts the draft gate rather than a
    # near-miss that happened to land.
    facts = _facts(schema, **{k: v for k, v in junger.features.items() if v not in WILDCARDS})
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(schema, corpus, [(facts, scores, rules.verdicts(facts))])

    assert collision.precedents == []
    assert "junger-6th-2000" in collision.unverified_candidates  # for a human, not a finding


def test_verifying_one_record_switches_the_precedent_path_on(schema: FeatureSchema, rules) -> None:
    """What the author's work buys: a single verified coding is enough to make
    the module's headline output start working."""

    from modules.casework.corpus import CaseCorpus, load_cases
    from modules.divergence.precedent import match_precedents

    corpus = load_cases("data/cases.yaml")
    junger = next(c for c in corpus.cases if c.id == "junger-6th-2000")
    verified = CaseCorpus(
        [
            CodedCase(
                id=junger.id, citation=junger.citation, name=junger.name,
                features=junger.features, outcome=junger.outcome,
                coding=Coding.MODEL_VERIFIED, coded_by="author",
            )
        ]
    )
    facts = FactVector(dict(junger.features))
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(schema, verified, [(facts, scores, rules.verdicts(facts))])

    assert len(collision.precedents) == 1
    assert collision.precedents[0].citation == "209 F.3d 481"


def test_junger_is_coded_as_the_opinion_reads() -> None:
    """The two corrections the opinion forced, pinned so a later edit that
    reinstates either has to argue with the text.

    `outcome` is empty because the Sixth Circuit reversed and remanded: it held
    source code is protected speech and left validity for the district court.
    `restriction_type` is incidental because the panel framed the analysis it
    did not reach under O'Brien intermediate scrutiny.
    """

    from pathlib import Path

    from modules.casework.corpus import load_cases

    junger = next(c for c in load_cases("data/cases.yaml").cases if c.id == "junger-6th-2000")

    assert junger.outcome == ""
    assert junger.features["restriction_type"] == "incidental"
    assert "data/reviews/junger-6th-2000.md" in junger.note
    assert Path("data/reviews/junger-6th-2000.md").exists()
    # Reading the opinion is not the human confirmation the flag asserts.
    assert junger.coding is Coding.MODEL_DRAFT


def test_a_case_with_no_recorded_holding_contradicts_nothing(
    schema: FeatureSchema, rules
) -> None:
    """A threshold holding locates a point in the fact space and adjudicates
    nothing there. Coding it `uncertain` instead would mark every reading that
    reaches a confident outcome as refuted by a case that refuted nothing."""

    from modules.casework.corpus import load_cases

    junger = next(c for c in load_cases("data/cases.yaml").cases if c.id == "junger-6th-2000")
    verified = CaseCorpus(
        [
            CodedCase(
                id=junger.id, citation=junger.citation, name=junger.name,
                features=junger.features, outcome=junger.outcome,
                coding=Coding.MODEL_VERIFIED, coded_by="test",
            )
        ]
    )
    facts = FactVector(dict(junger.features))
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(schema, verified, [(facts, scores, rules.verdicts(facts))])

    assert len(collision.precedents) == 1
    assert collision.precedents[0].contradicting == ()


def test_a_precedent_nothing_contradicts_is_not_written_as_an_objection() -> None:
    """The framework getting a case right is not an objection the author owes a
    reply to. Emitting one would file a false sentence and leave a permanent
    UNANSWERED_ATTACK against it."""

    from modules.casework.graph_adapter import emit_precedents
    from modules.divergence.precedent import MishandledPrecedent
    from modules.maieutic.graph import ArgumentGraph

    graph = ArgumentGraph()
    agreed = MishandledPrecedent(
        case_id="junger-6th-2000", citation="209 F.3d 481", name="Junger v. Daley",
        distance=0, facts={}, held="", verdicts=("uncertain",), scores={},
        coded_by="author", contradicting=(),
    )
    conflicting = MishandledPrecedent(
        case_id="other-case", citation="1 F.3d 1", name="Other v. Case",
        distance=0, facts={}, held="restricted", verdicts=("permitted",), scores={},
        coded_by="author", contradicting=("expressive_code",),
    )

    emission = emit_precedents(graph, [agreed, conflicting], epoch=1)

    assert emission.no_conflict == ["junger-6th-2000"]
    assert len(emission.added) == 1
    assert "no reading contradicts" in emission.summary()
    # And the one node written names the reading it indicts.
    (node_id,) = emission.added
    assert "expressive_code" in graph.nodes[node_id].text


# --------------------------------------------------------------------------- #
# UNDETERMINED: what a coding says when the source does not
# --------------------------------------------------------------------------- #
def test_undetermined_is_a_coding_value_and_never_a_search_value(
    schema: FeatureSchema,
) -> None:
    """The asymmetry the whole design rests on. A decided case can be silent
    about a fact; a fact pattern the optimizer invented cannot be."""

    from pathlib import Path

    from modules.casework.schema import UNDETERMINED

    coding = {axis.name: axis.values[0] for axis in schema.axes}
    coding["capability_tier"] = UNDETERMINED

    schema.validate_coding(coding)  # allowed
    with pytest.raises(ValueError, match="not a value of axis"):
        schema.validate(coding)  # a fact pattern may not be undetermined

    # It is absent from every axis, so the sampler and the genome cannot reach it.
    assert all(UNDETERMINED not in axis.values for axis in schema.axes)
    rng = random.Random(7)
    mask = _mask(schema)
    for _ in range(60):
        assert UNDETERMINED not in mask.sample(rng).as_dict().values()
    # And absent from the frozen file, so it cost no digest change.
    assert UNDETERMINED not in Path("data/feature_schema.yaml").read_text()


def test_an_undetermined_axis_is_a_wildcard_in_distance(schema: FeatureSchema) -> None:
    """A coding that admits what it does not know must not thereby be pushed out
    of matching range — that would make the honest coding count for less than
    the invented one."""

    from modules.casework.schema import UNDETERMINED

    tier = schema.axis("capability_tier")

    assert tier.distance(UNDETERMINED, "5") == 0
    assert tier.distance("1", UNDETERMINED) == 0
    # Without the wildcard this pair is four ordinal steps apart.
    assert tier.distance("1", "5") == 4


def test_a_reading_that_needs_a_silent_axis_declines_to_answer(rules) -> None:
    """`deemed_export` branches on capability_tier and cannot speak to a case
    whose record omits it. `published_information` never looks, so it still
    answers — a blanket "any silence makes every rule uncertain" would erase
    that."""

    from modules.casework.schema import UNDETERMINED, Outcome

    facts = FactVector(
        {
            "distribution_modality": "training_code",
            "actor_type": "academic",
            "capability_tier": UNDETERMINED,
            "post_release_modification": "none",
            "harm_proximity": UNDETERMINED,
            "jurisdiction": "us_federal",
            "restriction_type": "incidental",
            "forum": "public_repository",
        }
    )

    verdicts = rules.verdicts(facts)

    assert verdicts["deemed_export"].outcome is Outcome.UNCERTAIN
    assert "capability_tier" in verdicts["deemed_export"].rationale
    assert "does not settle" in verdicts["deemed_export"].rationale
    # Reads forum and distribution_modality only: unaffected, and still answers.
    assert verdicts["published_information"].outcome is Outcome.PERMITTED
    assert "does not settle" not in verdicts["published_information"].rationale


def test_a_rule_crashing_on_a_silent_axis_reads_as_silence_not_breakage(rules) -> None:
    """`functional_artifact` does int(capability_tier). The ValueError is not a
    bug in the rule; it is the rule being asked a question the source cannot
    answer, and the rationale has to say the useful one of those two things."""

    from modules.casework.schema import UNDETERMINED, Outcome

    facts = FactVector(
        {
            "distribution_modality": "weights",
            "actor_type": "firm",
            "capability_tier": UNDETERMINED,
            "post_release_modification": "none",
            "harm_proximity": "remote",
            "jurisdiction": "us_federal",
            "restriction_type": "incidental",
            "forum": "public_repository",
        }
    )

    verdict = rules.verdicts(facts)["functional_artifact"]

    assert verdict.outcome is Outcome.UNCERTAIN
    assert "not applicable" in verdict.rationale
    assert "ValueError" not in verdict.rationale


def test_a_coding_too_silent_to_govern_is_named_rather_than_matched(
    schema: FeatureSchema, rules
) -> None:
    """The price of the wildcard, paid explicitly. Silence costs nothing in
    distance, so a case that waives half the schema sits within tolerance of a
    huge region and would be cited as the precedent behind findings it has no
    bearing on."""

    from modules.casework.schema import UNDETERMINED

    silent = CodedCase(
        id="mostly-silent", citation="1 F.3d 1", name="Silent v. Record",
        features={
            **{axis.name: axis.values[0] for axis in schema.axes},
            "capability_tier": UNDETERMINED,
            "harm_proximity": UNDETERMINED,
            "forum": UNDETERMINED,
            "actor_type": UNDETERMINED,
        },
        outcome="restricted", coding=Coding.MODEL_VERIFIED, coded_by="test",
    )
    assert silent.determinacy(schema) == 0.5

    facts = FactVector({axis.name: axis.values[0] for axis in schema.axes})
    scores = score(facts, rules, _mask(schema), schema)
    collision = match_precedents(
        schema, CaseCorpus([silent]), [(facts, scores, rules.verdicts(facts))]
    )

    assert collision.precedents == []
    assert collision.too_undetermined == ["mostly-silent"]
    assert "too undetermined" in collision.summary()


def test_coverage_counts_silence_as_its_own_answer(schema: FeatureSchema) -> None:
    """An axis where most cases are undetermined is unanswered, not thin, and
    those need different work: more cases versus a source that speaks to it."""

    from modules.casework.schema import UNDETERMINED

    case = CodedCase(
        id="c", citation="1 F.3d 1", name="C", outcome="permitted",
        coding=Coding.MODEL_VERIFIED, coded_by="test",
        features={
            **{axis.name: axis.values[0] for axis in schema.axes},
            "harm_proximity": UNDETERMINED,
        },
    )

    counts = CaseCorpus([case]).coverage(schema)

    assert counts["harm_proximity"][UNDETERMINED] == 1
    assert sum(counts["harm_proximity"].values()) == 1
    assert counts["jurisdiction"][UNDETERMINED] == 0


def test_junger_records_the_two_axes_its_opinion_does_not_supply() -> None:
    """The case the wildcard was added for: no key length appears anywhere in
    the opinion, and the panel held the record did not resolve harm proximity.
    Six of eight axes answered still clears the floor to anchor a finding."""

    from modules.casework.corpus import load_cases
    from modules.casework.schema import load_schema

    schema = load_schema("data/feature_schema.yaml")
    junger = next(c for c in load_cases("data/cases.yaml").cases if c.id == "junger-6th-2000")

    assert junger.undetermined == ("capability_tier", "harm_proximity")
    assert junger.determinacy(schema) == 0.75


def test_no_judicial_record_claims_a_capability_tier() -> None:
    """The sweep's largest result. No opinion in the corpus grades an artifact on
    a five-point capability scale, because the scale postdates every one of them —
    so the tiers previously coded here were the coder's analogy wearing the
    court's authority. The axis is otherwise carried only by the two regulatory
    instruments, which set numeric thresholds.

    Which blank each judicial record gets is the substantive claim: a case about
    a *means of doing something* has a capability the court did not quantify; a
    case about *information* has none to quantify."""

    from modules.casework.corpus import load_cases
    from modules.casework.schema import INAPPLICABLE, UNDETERMINED, WILDCARDS

    cases = load_cases("data/cases.yaml").cases
    judicial = [c for c in cases if c.citation_source.startswith("cap")]
    instruments = [c for c in cases if not c.citation_source.startswith("cap")]

    assert len(judicial) == 15
    assert all(c.features["capability_tier"] in WILDCARDS for c in judicial)
    assert all(c.features["capability_tier"] not in WILDCARDS for c in instruments)

    means = {c.id for c in judicial if c.features["capability_tier"] == UNDETERMINED}
    information = {c.id for c in judicial if c.features["capability_tier"] == INAPPLICABLE}

    # Encryption code, DeCSS, an H-bomb design, a murder manual, terrorist training.
    assert means == {
        "junger-6th-2000", "bernstein-9th-1999", "karn-ddc-1996", "corley-2d-2001",
        "progressive-wdwis-1979", "rice-4th-1997", "hlp-2010",
    }
    # A policy history, a scandal sheet, two memoirs, a phone call, a prescriber
    # database, a video game, a depiction of conduct.
    assert information == {
        "pentagon-papers-1971", "near-1931", "marchetti-4th-1972", "snepp-1980",
        "bartnicki-2001", "sorrell-2011", "brown-ema-2011", "stevens-2010",
    }


def test_one_reading_cannot_be_tested_against_any_decided_case(rules) -> None:
    """`functional_artifact` opens with int(capability_tier), so once no opinion
    supplies a tier it has nothing to say about a single real decision in the
    corpus. That is a fact about the reading, not a bug: a formalisation whose
    first move is a fact no court records cannot be checked against courts."""

    from modules.casework.corpus import load_cases

    corpus = load_cases("data/cases.yaml")
    judicial = [c for c in corpus.cases if c.citation_source.startswith("cap")]

    silent = [
        c.id
        for c in judicial
        if "not applicable" in rules.verdicts(FactVector(dict(c.features)))[
            "functional_artifact"
        ].rationale
    ]

    assert silent == [c.id for c in judicial]


def test_two_records_are_too_narrow_a_holding_to_anchor_anything() -> None:
    """Bernstein limited itself to prior restraint and procedural safeguards; the
    Pentagon Papers per curiam allocated a burden in three paragraphs. Both fall
    under the determinacy floor, which is the right result — a court that decided
    one narrow thing has not told you where the case sits."""

    from modules.casework.corpus import load_cases
    from modules.casework.schema import load_schema

    schema = load_schema("data/feature_schema.yaml")
    corpus = load_cases("data/cases.yaml")

    below = sorted(c.id for c in corpus.cases if c.determinacy(schema) < 0.75)

    assert below == ["bernstein-9th-1999", "pentagon-papers-1971"]


def test_the_corrections_the_opinions_forced() -> None:
    """Three codings the sweep found stating the opposite of the holding. Pinned
    with the passage each rests on, so reverting one has to argue with the text."""

    from modules.casework.corpus import load_cases

    by_id = {c.id: c for c in load_cases("data/cases.yaml").cases}

    # "§ 2511(1)(c) ... is in fact a content-neutral law of general applicability"
    assert by_id["bartnicki-2001"].features["restriction_type"] == "incidental"
    # "dissemination itself carries very substantial risk of imminent harm"
    assert by_id["corley-2d-2001"].features["harm_proximity"] == "imminent"
    # liability "regulated incidentally to the ... enforcement of generally
    # applicable statutes"
    assert by_id["rice-4th-1997"].features["restriction_type"] == "incidental"
    # 15 C.F.R. 734.13 defines when a release counts as an export, so what is
    # done with the technology afterwards does not arise — inapplicable rather
    # than silent, since no further reading of it would supply a value.
    assert by_id["deemed-export-rule"].features["post_release_modification"] == "inapplicable"


# --------------------------------------------------------------------------- #
# INAPPLICABLE: what a coding says when the question does not arise
# --------------------------------------------------------------------------- #
def test_the_two_blanks_are_wildcards_alike_and_differ_everywhere_else(
    schema: FeatureSchema,
) -> None:
    """Same behaviour in distance, opposite behaviour in the measure that decides
    whether a case can govern — because one blank is a hole someone can fill and
    the other never will be."""

    from modules.casework.schema import INAPPLICABLE, UNDETERMINED

    tier = schema.axis("capability_tier")
    assert tier.distance(INAPPLICABLE, "5") == 0
    assert tier.distance(UNDETERMINED, "5") == 0

    base = {axis.name: axis.values[0] for axis in schema.axes}
    silent = CodedCase(id="s", citation="", name="", outcome="",
                       features={**base, "capability_tier": UNDETERMINED})
    absent = CodedCase(id="a", citation="", name="", outcome="",
                       features={**base, "capability_tier": INAPPLICABLE})

    # The silence is charged for; the axis that never applied leaves the
    # denominator, so the coding is complete over what there was to find.
    assert silent.determinacy(schema) == 7 / 8
    assert absent.determinacy(schema) == 1.0
    assert silent.applicability(schema) == 1.0
    assert absent.applicability(schema) == 7 / 8


def test_a_case_the_schema_does_not_describe_is_not_filed_as_a_lead(
    schema: FeatureSchema, rules
) -> None:
    """The guard against the obvious abuse: mark enough axes inapplicable and a
    case scores perfect determinacy while matching half the space. Applicability
    catches it, and it lands in `off_schema` rather than `too_undetermined` —
    the remedy is removing the case, not reading harder."""

    from modules.casework.schema import INAPPLICABLE

    base = {axis.name: axis.values[0] for axis in schema.axes}
    stranger = CodedCase(
        id="wrong-subject", citation="1 F.3d 1", name="Stranger v. Schema",
        features={**base, "capability_tier": INAPPLICABLE, "harm_proximity": INAPPLICABLE,
                  "forum": INAPPLICABLE, "actor_type": INAPPLICABLE},
        outcome="restricted", coding=Coding.MODEL_VERIFIED, coded_by="test",
    )
    # Perfect over what applies, which is exactly why determinacy alone is not enough.
    assert stranger.determinacy(schema) == 1.0
    assert stranger.applicability(schema) == 0.5

    facts = FactVector(base)
    scores = score(facts, rules, _mask(schema), schema)
    collision = match_precedents(
        schema, CaseCorpus([stranger]), [(facts, scores, rules.verdicts(facts))]
    )

    assert collision.precedents == []
    assert collision.off_schema == ["wrong-subject"]
    assert collision.too_undetermined == []
    assert "off-schema" in collision.summary()


def test_a_reading_says_which_kind_of_blank_stopped_it(rules) -> None:
    """Same verdict, different diagnosis, and an author needs both: one blank is
    worth chasing and the other never will be. `deemed_export` reads actor_type,
    then forum, then the tier, so a case blank on two of those in different ways
    gets a reason for each — and only for the axes it actually consulted."""

    from modules.casework.schema import INAPPLICABLE, UNDETERMINED, Outcome

    facts = FactVector(
        {
            "distribution_modality": "weights", "actor_type": UNDETERMINED,
            "capability_tier": INAPPLICABLE, "post_release_modification": "none",
            # Never read by this rule, and so never mentioned.
            "harm_proximity": INAPPLICABLE, "jurisdiction": "us_federal",
            "restriction_type": "incidental", "forum": "public_repository",
        }
    )

    verdict = rules.verdicts(facts)["deemed_export"]

    assert verdict.outcome is Outcome.UNCERTAIN
    assert "actor_type, which the source does not settle" in verdict.rationale
    assert "capability_tier, which does not arise in this case" in verdict.rationale
    assert "harm_proximity" not in verdict.rationale


def test_no_record_in_the_corpus_is_off_schema() -> None:
    """The corpus already drops off-schema cases in a comment; this is the same
    judgement as a number, so it can be checked rather than trusted."""

    from modules.casework.corpus import load_cases
    from modules.casework.schema import load_schema

    schema = load_schema("data/feature_schema.yaml")

    assert all(c.applicability(schema) >= 0.625 for c in load_cases("data/cases.yaml").cases)


def test_a_coding_that_predates_an_axis_does_not_crash_the_matcher(
    schema: FeatureSchema, rules
) -> None:
    """The unverified-candidate path fills missing axes so a coding can outlive a
    schema change — but it filled them with "", which is not a value of any axis,
    so `Axis.distance` fell through to tuple.index("") and raised ValueError on
    the first ordinal axis. The guard there catches KeyError only, so the exact
    scenario the default exists for was the one that crashed."""

    stale = CodedCase(
        id="predates-the-schema", citation="1 F.3d 1", name="Old v. Coding",
        features={a.name: a.values[0] for a in schema.axes if a.name != "capability_tier"},
        outcome="restricted",
    )
    facts = FactVector({a.name: a.values[0] for a in schema.axes})
    scores = score(facts, rules, _mask(schema), schema)

    collision = match_precedents(
        schema, CaseCorpus([stale]), [(facts, scores, rules.verdicts(facts))]
    )

    assert collision.unverified_candidates == ["predates-the-schema"]


def test_a_blank_axis_does_not_buy_a_wider_net(schema: FeatureSchema, rules) -> None:
    """The defect the first end-to-end run exposed. Blanks are free in distance,
    so without this they compound with the tolerance: Stevens, blank on two axes,
    matched a capability-tier-5 pattern at a reported distance of 1 while
    actually differing on three, and was emitted as an objection about "its
    facts" — one of which was a capability the case has no position on."""

    from modules.casework.schema import INAPPLICABLE, UNDETERMINED

    base = {axis.name: axis.values[0] for axis in schema.axes}
    blank = CodedCase(
        id="two-blanks", citation="1 F.3d 1", name="Blank v. Coding",
        features={**base, "capability_tier": INAPPLICABLE, "harm_proximity": UNDETERMINED},
        outcome="permitted", coding=Coding.MODEL_VERIFIED, coded_by="test",
    )
    full = CodedCase(
        id="fully-coded", citation="2 F.3d 2", name="Full v. Coding",
        features=dict(base), outcome="permitted",
        coding=Coding.MODEL_VERIFIED, coded_by="test",
    )

    # One step away on an axis both codings actually answer.
    other = schema.axis("jurisdiction").values[1]
    facts = FactVector({**base, "jurisdiction": other})
    scores = score(facts, rules, _mask(schema), schema)

    def matched(case: CodedCase) -> list[str]:
        collision = match_precedents(
            schema, CaseCorpus([case]), [(facts, scores, rules.verdicts(facts))]
        )
        return [p.case_id for p in collision.precedents]

    # The fully-coded case spends its one step and matches.
    assert matched(full) == ["fully-coded"]
    # The blank one already spent it — twice over — so it must agree exactly.
    assert matched(blank) == []
    assert matched(blank) == [] and blank.determinacy(schema) == 6 / 7  # still above the floor


def test_a_coding_blank_on_everything_is_not_excluded_outright(
    schema: FeatureSchema, rules
) -> None:
    """Floored at zero, not negative. A coding is never penalised for admitting
    what it does not know — only stopped from being rewarded for it — so it can
    still match, on the strongest terms left to it: exact agreement everywhere
    it did answer."""

    from modules.casework.schema import UNDETERMINED

    base = {axis.name: axis.values[0] for axis in schema.axes}
    blank = CodedCase(
        id="mostly-blank", citation="1 F.3d 1", name="Blank v. Coding",
        features={**base, "capability_tier": UNDETERMINED, "harm_proximity": UNDETERMINED},
        outcome="permitted", coding=Coding.MODEL_VERIFIED, coded_by="test",
    )

    exact = FactVector({**base, "capability_tier": "5", "harm_proximity": "imminent"})
    scores = score(exact, rules, _mask(schema), schema)
    collision = match_precedents(
        schema, CaseCorpus([blank]), [(exact, scores, rules.verdicts(exact))]
    )

    assert [p.case_id for p in collision.precedents] == ["mostly-blank"]
    assert collision.precedents[0].distance == 0


# --------------------------------------------------------------------------- #
# Seeding the search from the coded cases
# --------------------------------------------------------------------------- #
def test_a_coded_case_is_expanded_before_it_can_seed_anything(schema: FeatureSchema) -> None:
    """A coded case is not a fact pattern: its blank axes carry no value the
    search may hold. Completing it over those axes is the exact meaning of an
    undetermined one — the case is at one of these points and the record does
    not say which — so the whole region it might occupy gets seeded."""

    from modules.casework.schema import INAPPLICABLE, UNDETERMINED, WILDCARDS
    from modules.divergence.seeding import seeds_from_cases

    case = CodedCase(
        id="two-blanks", citation="1 F.3d 1", name="Blank v. Coding",
        features={**{a.name: a.values[0] for a in schema.axes},
                  "capability_tier": UNDETERMINED, "harm_proximity": INAPPLICABLE},
        outcome="permitted",
    )

    plan = seeds_from_cases(schema, CaseCorpus([case]), _mask(schema))

    # 5 tiers x 3 proximities, and every one a vector the search may actually hold.
    assert plan.per_case == {"two-blanks": 15}
    assert len(plan.seeds) == 15
    assert all(v not in WILDCARDS for seed in plan.seeds for v in seed.as_dict().values())
    for seed in plan.seeds:
        schema.validate(seed.as_dict())
    # The answered axes are held fixed; only the blanks vary.
    assert {s["distribution_modality"] for s in plan.seeds} == {schema.axes[0].values[0]}
    assert {s["capability_tier"] for s in plan.seeds} == set(schema.axis("capability_tier").values)


def test_seeding_spends_its_budget_on_the_better_coded_cases(schema: FeatureSchema) -> None:
    """Cheapest first, so one record blank on four axes cannot crowd out six
    well-coded ones — and whatever is dropped is named rather than truncated in
    silence, which would read as full coverage of the corpus."""

    from modules.casework.schema import UNDETERMINED
    from modules.divergence.seeding import seeds_from_cases

    base = {a.name: a.values[0] for a in schema.axes}
    tidy = CodedCase(id="tidy", citation="", name="", outcome="", features=dict(base))
    sprawling = CodedCase(
        id="sprawling", citation="", name="", outcome="",
        features={**base, "capability_tier": UNDETERMINED, "harm_proximity": UNDETERMINED},
    )

    plan = seeds_from_cases(schema, CaseCorpus([sprawling, tidy]), _mask(schema), budget=5)

    assert plan.per_case == {"tidy": 1}
    assert plan.truncated == ["sprawling"]
    assert plan.dropped == 15
    assert "dropped past the budget" in plan.summary()


def test_a_coded_decision_the_mask_calls_impossible_is_named(schema: FeatureSchema) -> None:
    """A real case with no feasible completion means the mask and the corpus
    disagree about what can exist. That is a bug in one of them, and it must not
    pass in silence."""

    from modules.divergence.seeding import seeds_from_cases

    case = CodedCase(id="cannot-exist", citation="", name="", outcome="",
                     features={a.name: a.values[0] for a in schema.axes})
    mask = _mask(schema).require("nothing is allowed", lambda v: False)

    plan = seeds_from_cases(schema, CaseCorpus([case]), mask)

    assert plan.seeds == []
    assert plan.infeasible == ["cannot-exist"]
    assert "infeasible under the mask" in plan.summary()


def test_the_stamped_recipe_names_everything_that_moves_the_archive() -> None:
    """The stamp is a command, not a summary. Descriptors, the realism weight and
    seeding each change the result, and a recipe omitting them reproduces a
    different run — which is what it was quietly doing."""

    from modules.divergence.cli import build_parser

    args = build_parser().parse_args(
        ["search", "--schema", "s.yaml", "--rules", "r", "--epoch", "1",
         "--descriptors", "actor_type", "--realism-weight", "0.25", "--seed-from-cases"]
    )

    assert args.seed_from_cases is True
    assert args.realism_weight == 0.25
    assert args.descriptors == ["actor_type"]


def test_a_caller_that_supplies_no_recipe_is_told_the_stamp_is_incomplete(
    schema: FeatureSchema, rules
) -> None:
    """`run_map_elites` sees a list of seed vectors, not the switch that made
    them, so it cannot write a command that reproduces them. It says so instead
    of printing one that silently drops the seeds."""

    from modules.divergence.mapelites import DivergenceConfig, run_map_elites

    mask = _mask(schema)
    priors = [FactVector({a.name: a.values[0] for a in schema.axes})]
    config = DivergenceConfig(descriptors=("actor_type", "capability_tier"), iterations=40)

    seeded = run_map_elites(schema, rules, mask, config=config, epoch=1, priors=priors)
    plain = run_map_elites(schema, rules, mask, config=config, epoch=1)

    assert "INCOMPLETE" in seeded.stamp.recipe
    assert "1 seed vector(s)" in seeded.stamp.recipe
    assert "INCOMPLETE" not in plain.stamp.recipe
    # And what it can name, it does.
    assert "--realism-weight" in plain.stamp.recipe
    assert "--descriptors actor_type capability_tier" in plain.stamp.recipe
