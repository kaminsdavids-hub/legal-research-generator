# Dialectic and Maieutic Modules — Observable Rules

This file records the rules governing `modules/dialectic/` and
`modules/maieutic/`, split into two sections:

* **Enforced** — the code implements the rule and a named test fails if it stops
  holding.
* **Designed, not yet enforced** — the rule is the intended behaviour, but
  nothing in the test suite would catch a regression.

The split is the whole value of this file. A document that reads as a
specification of enforced behaviour while describing intent is worse than no
document, because it invites exactly the assumption the reader should not make.
Every rule below names the test that enforces it, or says plainly that none does.

---

## Enforced

### 1. No model-emitted citation strings

Models emit **propositions carrying citation slots**, not citations. A
proposition is plain prose; the slot metadata (`court_hint`, `weight`, `status`)
is separate. The `CitationChannel` scans the proposition text (and any raw
assistant/system text) for:

* reporter-pattern citations (`123 F.2d 456`, `5 U.S. 137`, ...) — the reporter
  list is hand-maintained and duplicate-free
  (`test_reporter_table_has_no_duplicates`), and
* "X v. Y" / "X vs. Y" case-name patterns.

If either pattern matches, the turn is void and regenerated. User-provided text
is exempt because the invariant targets generation models, not the human.

**Every model-authored field is scanned, not just `proposition`.** A model
blocked from citing in one field will otherwise route the citation through
another. `court_hint` is scanned; `normalized_cite` is not accepted from a model
at all (see rule 2).

> Enforced by `test_citation_channel_reports_zero_false_negatives`,
> `test_citation_channel_scans_assistant_but_not_user`,
> `test_citation_channel_detects_split_cite`,
> `test_canary_c1_model_output_with_full_cite_is_rejected`,
> `test_canary_c2_model_output_with_split_newline_cite_is_rejected`,
> `test_citation_in_court_hint_is_rejected_like_one_in_the_proposition`,
> `test_end_to_end_turn_rejects_citation_in_proposition`.

### 2. `normalized_cite` is an output, never an input

A citation slot's `normalized_cite` is produced by the retrieval stage and
confirmed by verification. It is **never** read from model JSON. A model that
supplies one is treated as having emitted a citation: the turn is discarded and
regenerated, exactly as a citation in `proposition` would be.

> Enforced by `test_citation_in_normalized_cite_is_rejected_like_one_in_the_proposition`.

### 3. The retrieval stage is what makes verification reachable

The pipeline is:

```
propositions ──► retrieval(court_hint, proposition) ──► candidate cites
                                                             │
                                            CourtListener verification
                                                             │
                                     VERIFIED / NOT_FOUND per slot
```

Retrieval is the only thing that turns a plain-English `court_hint` into a
candidate citation. Without a configured retriever, no slot ever carries a
candidate, `verify_position` finds nothing to look up, and the exchange spends
zero verification calls — which is correct, and must be **stated** rather than
left as a bare `pending`.

A retrieved candidate is not authority. It moves the slot to `proposed`, and
only a CourtListener lookup returning HTTP 200 with a cluster can move it to
`verified`.

> Enforced by `test_retrieval_stage_feeds_courtlistener_verification`,
> `test_spec_compliant_exchange_spends_no_verification_calls`,
> `test_retrieved_candidate_is_never_clean_until_verified`,
> `test_retrieval_miss_leaves_the_slot_pending_and_says_so`,
> `test_offline_retrievers_satisfy_the_protocol`,
> `test_stub_retriever_prefers_the_more_specific_hint`.

### 4. CourtListener v4 rate budget

`CourtListenerClient` enforces a token-bucket budget over three rolling windows,
the Free Law Project authenticated-user defaults:

* 5 requests / minute
* 50 requests / hour
* 125 requests / day

Verification performs **one POST** to `/api/rest/v4/citation-lookup/` per
position, never per citation. When the budget is exhausted, `BudgetExhausted` is
raised; the module never falls back to unverified output. On a forced `429` or
timeout, every affected slot is marked `NOT_FOUND`. No slot is ever marked
`VERIFIED` unless the lookup returned HTTP 200 with a cluster.

The budget is **process-local**: it uses `time.monotonic()`, so the day window
cannot survive a restart. See REMEDIATION §9.7 for the decision record.

> Enforced by `test_rate_budget_enforces_all_three_windows`,
> `test_rate_budget_prunes_outside_the_day_window`,
> `test_rate_budget_cache_hit_does_not_spend`,
> `test_forced_429_leaves_slots_not_found_not_verified`,
> `test_forced_timeout_leaves_slots_not_found_not_verified`,
> `test_canary_c3_429_mid_verification_no_slot_verified`.
>
> A slot that falls to `NOT_FOUND` because no returned key matched says so
> explicitly rather than reporting an opaque status —
> `test_verification_note_names_the_key_mismatch`.

### 5. Crux extraction is a separate NLI pass, and weight does not gate it

The `CruxExtractor` runs an NLI pass over the full cross-product of thesis ×
antithesis propositions; the debaters themselves never assign `negates`.

**Weight does not gate extraction.** The weight is self-declared by each debater
about its own argument, so gating on it let a modest model disable the whole
crux engine by calling its own argument `supporting`. A contradiction between
two `supporting` propositions is still a contradiction; it simply is not
authority-resolvable, and `PrecedenceRule.classify` returns `open` for it.
`Crux.outcome_bearing` preserves the distinction for ranking.

When the crux table is empty, `DialecticTurn.crux_note` states why, and the
serializers render it.

> Enforced by `test_crux_extraction_is_not_gated_by_self_declared_weight`,
> `test_crux_extractor_finds_contradictions_only_for_outcome_bearing_slots`,
> `test_empty_crux_table_states_the_reason`.

### 5a. Weight-aware partition (the precedence rule)

`PrecedenceRule` classifies each crux using this precedence order:

1. **Weight always outranks verification.** The weight order is
   `controlling > persuasive > supporting > contra`.
2. **Verification only breaks ties within the same weight.**

Consequence (the rule the canaries protect): **a verified persuasive cite from
another circuit does NOT outrank an unverified but correct reading of
controlling precedent.**

After authority comparison, remaining cruxes are partitioned as `resolvable by
fact` (both sides dispute a factual predicate) or `open` (everything else).

> Enforced by `test_precedence_rule_weight_gates_authority_classification`,
> `test_canary_c6_persuasive_verified_does_not_outrank_controlling_unverified`,
> and `probe.py` item 8.

### 5b. Weights and statuses are validated

`Weight` and `SlotStatus` are `StrEnum`s and `CitationSlot` is typed with them,
so an out-of-vocabulary value fails validation rather than being silently
stored. Model-supplied weights are case-normalized; a weight that has to be
coerced to the default is recorded in the slot's `note`, never rewritten
silently.

> Enforced by `test_citation_slot_rejects_a_weight_outside_the_enum`,
> `test_weight_parsing_is_case_insensitive_and_notes_coercions`,
> `test_unrecognised_weight_is_coerced_but_recorded_in_the_note`.

### 6. The NLI evaluator reports which path answered

When an NLI client is configured it is consulted, and its answer is parsed
strictly: exactly `{"label": "entailment"|"contradiction"|"neutral"}`. Anything
else falls back to the offline heuristic — and the fallback is **recorded** on
the resulting `Crux` as `nli_source="heuristic"`, so a silent downgrade is
visible in the output rather than invisible.

> Enforced by `test_nli_uses_the_model_when_one_is_configured`,
> `test_nli_falls_back_to_heuristic_and_records_the_downgrade`,
> `test_nli_source_is_recorded_on_the_crux`,
> `test_nli_client_reaches_the_evaluator`.

### 6a. The offline heuristic does not manufacture contradictions

The heuristic is the fallback path, so its failure mode matters: it must not
report a contradiction between two propositions that merely share stopwords and
differ in polarity. Contradiction requires content-word overlap above a
threshold **and** a negation that reaches a shared term. Negation is detected on
raw text, so contractions (`doesn't`, `won't`) count.

Morphological antonyms (`constitutional` / `unconstitutional`) carry no negation
word, so they are caught by a word-boundary antonym check that runs *before* the
overlap-based entailment shortcut.

> Enforced by `test_nli_derives_negates_not_model`,
> `test_nli_does_not_flag_unrelated_legal_prose_as_contradiction`,
> `test_nli_detects_negation_written_as_a_contraction`,
> `test_nli_handles_morphological_antonyms`, and `probe.py` items 4 and 5.

### 7. Copy parity and unsupported markers

Every serializer — `copy_exchange`, `copy_position`, and `copy_crux_table` —
preserves `[UNSUPPORTED: <weight> authority for <proposition>]` markers
verbatim. Unfilled slots stay visible and flagged; a copy path that silently
drops the marker is treated as the single most dangerous defect.

**The only state that renders clean is a `verified` slot carrying a citation.**
A `verified` slot with an empty `normalized_cite` is a partially-populated
record, not a resolved one, and keeps its marker.

> Enforced by `test_copy_exchange_preserves_unsupported_marker`,
> `test_copy_position_preserves_unsupported_marker`,
> `test_copy_crux_table_preserves_unsupported_marker`,
> `test_canary_c4_serializer_preserves_unsupported_marker`,
> `test_partially_populated_verified_slot_still_renders_a_marker`.

### 8. Distinct model families

Thesis, antithesis, synthesis, **and the NLI model** must use distinct base
model families. The engine raises `FamilyCollision` if any two configured roles
share one. The NLI model is included because it adjudicates the debate, so
sharing a family with a debater reintroduces the correlated error the guard
exists to prevent. `detect_family()` maps model strings to families (`gpt`,
`gemma`, `llama`, `nemotron`, `hermes`, `saul`, `apertus`, `unknown`).

> Enforced by `test_detect_family_maps_models_to_families`,
> `test_family_collision_guard_raises_when_debaters_share_family`,
> `test_canary_c5_same_family_guard_fires`,
> `test_family_guard_covers_the_nli_model`.
>
> **The guard is enforced; its empirical justification is not established.**
> See "Designed, not yet enforced" §D3 and REMEDIATION §5.

### 9. Synthesis runs last and sees the crux table

`chat()` orders the pipeline retrieve → verify → extract cruxes → synthesise. The
synthesis prompt asks the model to name the decisive crux, so the crux table is
passed into it via `copy_crux_table(turn)`. The prompt carries each position
once: the thesis is not also rendered into the antithesis slot, and
`_SYNTHESIS_PROMPT` is the system message only, not additionally appended to the
user message.

> Enforced by `test_synthesis_receives_the_crux_table_and_no_duplicated_thesis`.
>
> Whether the synthesis model's *output* actually reflects the crux is not
> enforced — see D1.

### 10. Rejected turns are re-rolled, and failures keep their diagnostics

A rejected turn is retried with a changed decoding policy, not re-run: attempt 0
is deterministic, retries use temperature 0.7 and a per-attempt seed. This
applies to positions **and to synthesis**, which previously got a single shot.

A position or synthesis that exhausts its budget records *why* — parse failure
or citation rejection, including the offending hits and the attempt index — and
`DialecticTurn.regenerated` carries the real retry count.

> Enforced by `test_regeneration_varies_decoding_so_retries_are_not_identical`,
> `test_synthesis_retries_on_citation_instead_of_failing_once`,
> `test_synthesis_failure_records_the_reason`, and `probe.py` item 7.

---

### 11. The two sides are actually opposed

A dialectic whose sides agree produces no cruxes by construction. Each role is
therefore assigned an explicit stance — the thesis argues the affirmative, the
antithesis the negative — and the antithesis is shown the thesis's propositions
and required to contradict them directly, addressing the same predicate.

Generating both sides blind from the same question let them converge on the same
answer: 44% of all cross-product pairs came back `entailment` in a live
diagnostic, and on two of four questions the two sides asserted the same claim in
near-identical words (REMEDIATION §10.2).

A position that failed generation is **not** fed back as an argument: the
`(generation failed ...)` placeholder is excluded from the rebuttal block.

The propositions quoted into the antithesis prompt have already passed the
citation channel (rule 1), so echoing them cannot introduce a citation the
antithesis did not author.

> Enforced by `test_each_side_is_told_which_way_to_argue`,
> `test_antithesis_is_shown_the_thesis_to_contradict`,
> `test_failed_thesis_is_not_fed_back_as_an_argument`.
>
> **Consequence:** the antithesis now conditions on the thesis, so the two
> debaters are no longer independent draws. The §9.8 correlation experiment
> compared independent draws; a rerun would measure a different quantity.

### 11a. The antithesis argues independently, it does not mirror

Telling the antithesis to contradict the thesis produced contradictions of a
degenerate kind: the thesis's own sentence with "not" inserted. A mirror
concedes the thesis's framing, predicate and choice of authority and disputes
only the sign, so it carries no adversarial information.

`IndependenceGuard` rejects a proposition that differs from an opposing one in
polarity while carrying essentially the same vocabulary. **Both** conditions are
required — shared vocabulary alone is expected, and a polarity difference alone
is what real disagreement looks like. Rejected drafts are re-rolled with
feedback naming the offending propositions.

Unlike the citation channel this is a **quality** property, not a safety one, so
it degrades visibly instead of failing closed: if every attempt mirrors, the
least-bad draft is kept with each mirroring proposition flagged in its `note`,
which reaches the copy payloads and the API. The thesis is not subject to the
guard — it is generated first and has nothing to mirror.

Thresholds are calibrated against observed data (real mirrors jaccard
0.55–1.00, independent counter-theories 0.00–0.12) and sit in the gap between
the two populations. Both populations are test fixtures, so a change that
collapses the separation fails the suite.

> Enforced by `test_independence_guard_catches_polarity_flip_mirrors`,
> `test_independence_guard_allows_a_real_counter_theory`,
> `test_independence_guard_needs_both_shared_words_and_flipped_polarity`,
> `test_independence_guard_ignores_propositions_too_short_to_judge`,
> `test_independence_guard_scan_raises_and_names_every_mirror`,
> `test_independence_guard_is_inert_without_an_opposing_side`,
> `test_antithesis_that_mirrors_is_regenerated_with_feedback`,
> `test_persistent_mirroring_degrades_visibly_rather_than_failing_closed`,
> `test_thesis_is_not_subject_to_the_independence_guard`.

### 12. Derived structure reaches the API as typed fields

`regenerated`, `crux_note`, `outcome_bearing` and `nli_source` are fields on
`DialecticResponse` / `DialecticCrux`, not values a client must recover by
parsing the pre-rendered `copy_*` strings. The `Weight` and `SlotStatus` enums
serialize as their values (`"supporting"`, `"pending"`).

> Enforced by `test_dialectic_endpoint_exposes_derived_fields` and
> `test_dialectic_endpoint_reports_an_empty_crux_table_reason` in
> `tests/test_api.py`.

## Designed, not yet enforced

These are intended behaviours with no test that would catch a regression. Do not
read them as guarantees.

### D1. The synthesis output actually reflects the crux

Rule 9 enforces that the crux table *reaches* the synthesis prompt. Whether the
model then names the decisive crux in its answer — the thing the prompt asks for
— is unenforced, because asserting it needs a live model and a judgement about
free prose. A fake client returns whatever string the test gave it.

### D2. The NLI model path against a real model

`NLIEvaluator._ask_model` is exercised only against fakes. Its behaviour with a
real constrained-classification model — how often a small instruct model returns
parseable JSON, and therefore how often production silently runs on the
heuristic — is unmeasured.

### D3. Same-family models have correlated errors — **unsupported**

The premise behind rule 8: same-family instances produce correlated output, so
distinct families are required for the debate to surface real contradictions.

**This claim is not supported by any evidence in this repository.** The
originally reported collapse (1.0 → 0.0) was an artifact of a harness that ran
the same client on both sides over byte-identical text. A corrected live re-run
through the production path (8 questions, 2026-08-07) found **3 cruxes vs 2**,
Fisher exact two-sided **p = 1.000**, with the two arms disagreeing in *both*
directions. That is null, not confirmation — and it is also underpowered enough
not to refute the claim. See REMEDIATION §5 and §9.8.

Rule 8's guard is **enforced in code and tested**; what is unestablished is the
empirical premise offered as its justification. The guard is kept as a cheap,
low-risk default for an adversarial design, not because a collapse was measured.

`test_correlation_guard_runs_both_arms_through_the_production_path` enforces the
*harness mechanics* only and deliberately asserts no empirical claim: a unit test
over canned strings cannot answer this question, and the old one only appeared to
because it was rigged. `CorrelationGuardReport.separates` is a direction
comparison, not a significance test, and says so.

### D3a. The crux yield rate

The §9.8 base rate (a crux on 3 of 8 and 2 of 8 questions) was **diagnosed and
its main cause fixed** — the two debaters were arguing the same side. Measured
before/after on 4 questions: entailment between the sides fell from 44% to 3%,
and questions yielding at least one crux went from 2/4 to 4/4. See rule 11 for
what is enforced and REMEDIATION §10.2 for the full numbers.

What remains unenforced is the *rate itself*. Nothing asserts a minimum crux
yield against live models, because that needs a live run and a question set with
known-contested answers.

### D3c. Crux count is not a quality metric

Independence and direct contradiction pull against each other: the more
genuinely independent the antithesis's theory, the less likely it addresses the
same predicate as any thesis proposition, and the fewer pairs the NLI can call
contradictions. Measured, eliminating mirrors moved contradiction from 75% to
31% and neutral from 22% to 64% while *increasing* the number of distinct
disagreements (REMEDIATION §10.3).

Nothing enforces this, and nothing can: it is a warning, not a rule. **Do not
optimise for crux count.** A configuration that maximises cruxes is one that
maximises mirrors, because flipping the sign of a sentence guarantees a
contradiction.

### D3b. The crux table contains near-duplicates

Each side emits 3 propositions that are often near-paraphrases of one another,
and the extractor compares the full cross-product, so a single disagreement can
be reported many times — 8 cruxes for one disputed predicate on one measured
question. The crux count therefore means "how many contradicting pairs", not
"how many issues are in dispute".

The independence guard (rule 11a) removed most of this, since mirrors were the
main source of the redundancy: the worst case fell from 8 cruxes to 3. Redundancy
*within the thesis's own* propositions is untouched, so the caveat stands.

Nothing deduplicates the table or enforces a bound on it.
`test_empty_crux_table_states_the_reason` covers the empty case; there is no
coverage of the over-full one. Collapsing near-duplicate cruxes would change
what a crux means in the output, so it is left as a deliberate design decision
rather than folded into a correctness pass. See REMEDIATION §10.2.

### D4. The corpus retrieval adapter

`service._CorpusCiteRetriever` maps corpus hits to reporter citations. Only the
offline `StubCiteRetriever` is covered by tests; the corpus-backed adapter has no
test, because exercising it needs a corpus fixture with complete
volume/reporter/page records.

Still open. The maieutic module's own corpus adapter *is* now covered (rule M8),
and the duck-typed record fixture in `tests/test_maieutic_service.py` is the
fixture this rule says is missing — but it covers `CorpusCitationVerifier`, a
different class. `_CorpusCiteRetriever` remains untested.

### D5. Daily rate budget across restarts

Rule 4's 125/day cap is process-local by decision, not accident. If it ever has
to hold across restarts it needs a persistent store keyed on wall time. Nothing
tests or enforces the durable interpretation.

---

## Maieutic — Enforced

Rules governing `modules/maieutic/`. Tests live in `tests/test_maieutic_graph.py`,
`tests/test_maieutic_novelty.py`, `tests/test_maieutic_grounding.py` and
`tests/test_maieutic_service.py`.

### M1. Provenance is carried, not inferred

Every node records whether a model proposed it, a human wrote it, or a human
edited a proposal. `Node.propose` cannot produce a human-provenance node and
`as_verified` is the grounding gate's call alone. Enforced by
`test_provenance_cannot_be_forged` and `test_a_generator_cannot_hand_over_a_pre_verified_authority`.

### M2. A patch must add something

`GraphPatch` refuses construction with no nodes. An insertion asserting nothing
new is a restatement, and the cheapest place to say so is the constructor.
Enforced by `test_an_empty_patch_is_refused`.

### M3. Novelty has three bands, and the middle one is adjudicated

Above `HARD` (0.92) a candidate is a restatement outright; below `SOFT_LOW`
(0.70) it is novel on its face; between them an entailment critic decides,
because different words are not a contribution if the graph already entails
them. Enforced by `test_in_the_soft_band_entailment_decides_not_wording`.

**Both thresholds are PROVISIONAL** — see D6 below.

### M4. A degraded novelty gate is legible

Without real embeddings the gate falls back to lexical overlap, which cannot see
a paraphrase — the exact failure it exists to catch. Every verdict records the
`Method` that produced it, and the soft band without a critic admits the
question was unadjudicated rather than guessing. Enforced by
`test_every_verdict_records_the_method_that_produced_it` and
`test_the_soft_band_without_a_critic_admits_but_says_so`.

### M5. Novelty passes on partial contribution; grounding does not

A patch survives novelty if **anything** in it is new, but fails grounding if
**anything** in it is ungrounded. The asymmetry is deliberate: a restatement
beside a real contribution is merely redundant, whereas one fabricated claim
beside verified material is where an unsupported claim does the most damage.
Enforced by `test_a_patch_cannot_smuggle_a_restatement_beside_something_new` and
`test_grounding_is_all_or_nothing_across_a_patch`.

**Amended after the first live run.** The gate is still all-or-nothing and the
test above still holds — but the *loop* no longer refuses the whole patch when
the only ungrounded nodes are the machine's. Atomicity had a consequence nobody
chose: the party that cites badly is the machine and the party that loses their
work is the author, and every live exchange failed that way. What actually
matters is that nothing ungrounded reaches the manuscript, and dropping the
offending nodes secures that just as well. Atomicity therefore gives way, and
only across parties: if a node the **author** wrote fails grounding, the patch
still fails whole. See M41 and REMEDIATION §12.3.

### M6. Every node is verified-cited or explicitly argued — there is no third category

AUTHORITY nodes fabricate, so a citation must resolve through retrieval *and*
support the claim made of it; resolving is necessary and not sufficient. ORIGINAL
nodes float, so they need no citation but do need an argument subgraph. Requiring
citations of ORIGINAL nodes would push the author toward saying only what someone
else has already said, which is the opposite of what the loop is for. Enforced by
`test_a_real_case_cited_for_something_it_does_not_say_fails`,
`test_an_original_node_asserted_alone_cannot_merge` and
`test_an_original_node_needs_no_citation`.

### M7. The grounding gate fails closed

With no verifier configured, an AUTHORITY node does not merge. This is the
fabrication wall; a gate that waves authority through when its checker is absent
is worse than no gate, because it reports a pass. Enforced by
`test_without_a_verifier_an_authority_fails_closed` and
`test_a_broken_scorer_is_a_failure_not_a_pass`.

### M8. Support is scored per passage, against the scorer's own threshold

`SupportScorer.score` takes one passage at a time, and the best-supporting
passage decides. The threshold comes from the scorer, never from the gate:
lexical overlap and NLI entailment probabilities are not on comparable scales,
so a cut chosen here would be wrong for at least one of them. Enforced by
`test_the_scorer_is_called_with_one_passage_at_a_time` and
`test_the_threshold_comes_from_the_scorer_not_from_us`.

### M9. Non-case authority resolves without CourtListener

CourtListener indexes case law and cannot adjudicate a C.F.R. section however
correct it is. Sending one there poisoned the all-or-nothing cite cache and
invalidated three eval runs (REMEDIATION §11.9a). The corpus alone resolves
non-case authority. Enforced by `test_non_case_authority_resolves_without_courtlistener`.

### M10. Nothing passes by vacuous truth

`all([])` is True, and that reading once reported crashed eval questions as
passing both gates (REMEDIATION §11.9). A patch that assessed nothing does not
read as assessed. Enforced by
`test_a_patch_that_grounds_nothing_does_not_pass_by_vacuous_truth`.

### M11. No gate reads text the machinery wrote

Both gates read node text only — never notes, annotations or synthesis prose. A
critic reading its own system's output validates itself and reports a pass
forever; that happened three times in the dialectic work (REMEDIATION §5, §11.5,
§11.11a). This is a structural property of what the gates are passed, not a
separately testable assertion; it is listed here so a future change that hands a
gate a rendered string is recognisable as a regression.

### M12. Gap analysis reads structure, not prose

Every gap in `socratic.analyse` is a property of the graph's shape: an objection
with no reply, a claim nothing attacks, a node with no edges, a `DEPENDS_ON`
cycle. This is the module's main defence. A critic that reads text to judge
whether an argument is complete can be satisfied by confident writing — that is
how three self-satisfying gates reached the dialectic work (REMEDIATION §5,
§11.5, §11.11a). A missing reply edge cannot be written around.

The engine does read node text, to quote a claim back to the author when
phrasing a question. That is not the same failure: a question asserts nothing,
and the author answers it. The failure is a *gate* whose pass depends on
machine-authored prose, and there is none here.

### M13. Asking is not answering

A gap that has been asked about is skipped when choosing the next question, so
the loop moves on instead of nagging — but it stays in `analyse` and in
`AskedLog.outstanding`. A hole that vanishes from the report because it was
mentioned once is the report lying. The log keys on `Gap.key`, not question
text, so rewording a question does not make it new. Enforced by
`test_an_asked_but_unanswered_gap_stays_visible`,
`test_rewording_a_question_does_not_make_it_new` and
`test_a_gaps_key_is_stable_across_passes`.

### M14. The engine supplies pressure, not content

It produces questions and never proposes a node. An optional `QuestionWriter`
may reword a templated question, but a "question" that does not ask one is
refused and the template stands — an assertion arriving through the question
channel is content entering where only the author's may. A writer that raises
costs the wording, not the question. Enforced by
`test_a_writer_that_asserts_instead_of_asking_is_refused` and
`test_a_broken_writer_costs_wording_not_the_question`.

### M15. No question is manufactured to fill a quota

With no unasked gaps, `ask` returns nothing. Inventing one would spend the
author's attention on whatever the engine could think of rather than on a hole.
Enforced by `test_no_gaps_means_no_question_rather_than_an_invented_one`.

### M16. The worst hole is asked about first

`PRIORITY` runs self-grounding → unanswered attack → unsupported claim →
uncontested claim → orphan → unverified authority. A self-grounding argument is
a defect no further material fixes; an unverified citation is last because it is
the one gap the machine can sometimes close without asking anyone. A section
whose novelty delta has collapsed is *deprioritised, not abandoned* — a cycle in
a mined-out section still matters more than a stray citation somewhere fresh.
Enforced by `test_the_worst_hole_is_asked_about_first` and
`test_a_mined_out_section_is_deprioritised_not_abandoned`.

### M17. Every gap the engine can find, it can ask about

A gap kind with no question template is a silent dead end: found, reported,
never surfaced. Enforced by `test_every_gap_kind_has_a_question`, which iterates
`GapKind` rather than a hand-written list, so a new kind fails until it has one.

### M18. The author's answer is the only human node

It is captured verbatim; everything the exchange contributed carries
`Provenance.DIALECTIC`. `Node.propose` refuses HUMAN outright, so this is a
guarantee rather than a convention. An empty answer is the author declining the
question and raises `AnswerRefused` — it is recorded as unanswered, not merged as
nothing. Enforced by `test_the_answer_is_captured_verbatim_and_is_human`,
`test_nothing_the_machine_contributed_claims_human_provenance` and
`test_an_empty_answer_is_declining_the_question_not_an_insertion`.

### M19. The synthesis never enters the manuscript

Thesis and antithesis material is pressure — support the author may accept,
objections they must answer. The synthesis is the machine writing the paper's
conclusion, which is the one thing this system exists not to do. It is returned
alongside the patch as material for choosing the next question, never merged.
Enforced by `test_the_synthesis_never_enters_the_manuscript`.

### M20. Authority is not created at the adapter

Only a slot the dialectic module actually verified becomes an AUTHORITY node,
and it arrives with `verified=False`: CourtListener answered whether the citation
*resolves*, and the grounding gate asks whether the source supports *this claim*,
which is a different question. A `proposed` candidate enters as a plain premise
carrying no citation, so an unconfirmed cite cannot be read off the manuscript as
a real one. Enforced by
`test_an_authority_arrives_unverified_for_the_grounding_gate_to_judge` and
`test_a_retrieved_candidate_is_not_authority`.

### M21. Refusals at the boundary are recorded, never silent

A proposition carrying a citation string is refused (defence in depth — the
engine already voids such a turn), and so is a verified but `NOT CURRENTLY
OPERATIVE` authority, because `Node` has no note field and the warning cannot
travel with it. Merging that one would let a repealed rule render as clean
authority, which is precisely what the marker exists to prevent. A proposition
that vanished without a record is indistinguishable from one the model never
produced. Enforced by `test_a_proposition_carrying_a_citation_string_is_refused`,
`test_a_rescinded_authority_is_refused_rather_than_rendered_clean` and
`test_a_refusal_is_recorded_not_silent`.

### M22. A reply attaches to the objection, not to what it attacked

`unanswered_attacks` looks for a REPLY pointing at the OBJECTION. Attaching
anywhere else leaves every answered attack reading as open forever — the same
wrong-edge-end error the graph work already made once. Enforced by
`test_a_reply_attaches_to_the_objection_not_to_what_it_attacked`, which asserts
on `unanswered_attacks()` going empty rather than on edge shape.

### M23. Every gap kind can absorb an answer, and one that cannot be closed says so

`_ROLES` is complete over `GapKind`, so no question goes nowhere; the test
iterates the enum rather than a hand-written list. `SELF_GROUNDING` and
`UNVERIFIED_AUTHORITY` are marked `unresolved`, because closing them means
removing an edge or re-running retrieval and a patch only adds. The gap is still
open after the answer, and the report says so rather than treating a question as
resolved by having been answered. Enforced by
`test_every_gap_kind_can_absorb_an_answer` and
`test_a_gap_needing_an_edit_says_the_answer_did_not_close_it`.

### M24. Coherence blocks on structure, and only advises on semantics

The findings carry two severities and `passed` reads only the blocking ones.

**Blocking** — `SELF_GROUNDING`, `DANGLING_EDGE`, `CONTRADICTORY_RELATION`,
`DISCONNECTED`, `REPLY_TO_NOTHING`. Each is decidable from the graph's shape
alone. No model is consulted and none can be talked around, which is what makes
them safe to merge on.

**Advisory** — `SUPPORT_THAT_CONTRADICTS`, `ATTACK_THAT_AGREES`. Whether a node
marked SUPPORTS in fact contradicts what it supports is a question only an
entailment critic can answer, and this repository has been burned twice by
trusting one uncalibrated: the NLI heuristic scored `constitutional` against
`unconstitutional` as entailment (REMEDIATION §9.5), and the judge rated pure
failure placeholders 8.00/10 (§11.2). A model's opinion does not veto the
author's work. Enforced by `test_support_that_contradicts_is_reported_but_does_not_block`,
`test_every_incoherence_has_a_severity` (iterates the enum, so a new kind with no
severity fails at review time rather than at report time) and
`test_a_semantic_finding_never_flips_a_patch_to_failing`.

### M25. The gate does not merge what it is judging

Cycle detection needs the merged result, so it builds a trial view rather than
mutating the graph. A gate with a side effect is a gate that has already merged
whatever it was asked to judge. Enforced by
`test_the_gate_does_not_merge_what_it_is_judging`.

### M26. A merge is only answerable for what it introduces

A support/attack contradiction already sitting in the graph does not fail the
next patch; blocking on it would make the graph unmergeable forever. Likewise the
opening node of an empty graph is exempt from `DISCONNECTED` — there is nothing
yet to connect to — but a *first patch of several* nodes must still connect them.
Enforced by `test_a_contradiction_already_in_the_graph_is_not_this_merges_fault`,
`test_the_opening_node_of_an_empty_graph_is_exempt` and
`test_a_first_patch_of_several_nodes_must_still_connect_them`.

### M27. An antithesis proposition enters as an objection

The adapter types it `OBJECTION`, not `PREMISE`. A premise carrying an ATTACKS
edge leaves the graph describing an attack as support — exactly the mismatch M24
exists to catch, and the adapter must not be manufacturing them. Cited authority
stays `AUTHORITY` whichever side deploys it, so the grounding gate can still
check the citation. Enforced by
`test_an_antithesis_proposition_becomes_an_objection` and
`test_cited_authority_stays_an_authority_whichever_side_deploys_it`.

### M28. Banality compares against the field, novelty against the manuscript

A proposition the sources already state is not a contribution however absent it
is from the paper. The novelty gate compares a node to the graph; this one
compares it to the corpus. Enforced by
`test_a_claim_the_sources_already_state_is_commonplace`.

Without a corpus the limb does not run and `method` stays `None` — reporting a
claim as original because nothing was checked would be a stronger statement than
the gate is entitled to. Enforced by
`test_without_a_corpus_no_claim_about_the_field_is_made` and
`test_a_commonplace_verdict_records_how_it_was_reached`.

### M29. Only vacuity blocks

A claim whose content words are *entirely* hedges asserts nothing — that is
decidable by counting, not by asking a model. Everything else reports, for two
different reasons that both matter: legal writing hedges as a professional norm,
so blocking "this may be the better view" would harm the manuscript the gate
protects; and corpus similarity is method-dependent and uncalibrated. `BLOCKING`
is a table the test iterates, so widening it is a deliberate edit. Enforced by
`test_a_claim_that_is_all_hedge_asserts_nothing_and_is_blocked`,
`test_a_mostly_hedged_claim_is_reported_but_merges` and
`test_vacuity_is_the_only_blocking_verdict`.

### M30. Only what claims to be the contribution is judged for banality

A paper needs commonplace material: background, setup, statements of existing
doctrine. A machine-proposed premise being unoriginal is correct, not a defect,
and assessing it would make the manuscript unwritable.

`_is_a_contribution` qualifies a node two ways: **what the author wrote**
(`Provenance.HUMAN`, whatever structural role it plays) or **a THESIS/ORIGINAL
node** (a contribution by type). The first limb was added after wiring the loop
up — the adapter never produces a THESIS or an ORIGINAL, so a type-only rule made
this gate unreachable in the running system and a vacuous answer merged
unlooked-at. See REMEDIATION §12.1. Enforced by
`test_what_the_author_wrote_is_judged_whatever_role_it_plays`,
`test_the_same_words_from_the_machine_are_not_judged` and
`test_background_material_is_supposed_to_be_unoriginal`.

### M31. The gate feeds the question policy

`PatchBanality.barren_sections()` names sections where a patch landed no
substantive claim, which is what `SocraticEngine.ask(avoid_sections=...)`
consumes: somewhere the author is producing only throat-clearing and received
wisdom is somewhere to stop asking. Per-patch and therefore noisy — a caller
should accumulate across cycles rather than act on one. Enforced by
`test_a_section_producing_only_hedge_is_reported_as_barren` and
`test_a_section_with_one_real_claim_is_not_barren`.

### M32. Every node renders exactly once

This is the payoff the graph representation exists for: a claim reachable by two
paths is written once, not restated because the writer forgot it had been said.
Non-repetition is structural, not a matter of care. Enforced by
`test_a_node_reachable_by_two_paths_is_written_once` and
`test_every_node_is_rendered_exactly_once`.

### M33. Rendering is deterministic and involves no model

The author's own sentences are the prose; this module contributes ordering,
structural labels and disclosure, and never composes a claim. A model writing the
prose would reintroduce every fabrication risk the gates prevent, at the last
step, where nothing downstream checks it. `DEPENDS_ON` decides what must be read
first; ties break on node type and then on id, so the same graph always renders
identically — a renderer that shuffles equivalent nodes makes every draft diff
unreadable. Enforced by `test_the_same_graph_always_renders_the_same_way`,
`test_what_the_reader_must_accept_first_comes_first` and
`test_the_authors_words_are_not_rewritten`.

### M34. Two disclosures no audience can suppress

**An unverified citation never renders as clean authority** — marked in the
manuscript view, not only in review. A reader must not have to know which view
they are reading to know what has been confirmed. **Unanswered objections are
declared** in their own section: a paper that states its strongest objections and
answers them is the point, and one that states them and quietly moves on is the
failure the graph exists to make visible. Enforced by
`test_an_unverified_citation_never_renders_as_clean_authority` (which iterates
`Audience`) and `test_unanswered_objections_are_declared_not_buried`.

A manuscript in which no node was written by the author says so, because that is
a fact about the paper — `test_a_manuscript_with_no_authored_content_says_so`.

### M35. A self-grounding graph renders, and reports the cycle

The coherence gate blocks patches that create a `DEPENDS_ON` cycle, but a graph
loaded from disk can carry one. No reading order is correct, so the renderer
emits the nodes in a stable order and warns; hanging or silently dropping them
would be worse. Enforced by `test_a_self_grounding_graph_renders_rather_than_hanging`.

### M36. Every gate runs, and every refusal is reported

`GateReport.refusals` lists every reason a patch cannot merge, not just the
first. Reporting one refusal and stopping would make the author fix a problem,
resubmit, and discover the next — and the system's whole premise is that their
attention is the scarce resource. Advisories are carried separately and never
appear as refusals. Enforced by
`test_every_refusal_is_reported_not_just_the_first` and
`test_advisories_are_reported_without_blocking`.

The gate order is coherence → grounding → novelty → banality: a structurally
broken patch makes the other verdicts meaningless, and a fabricated citation
should be reported as fabrication rather than as restatement.

### M37. A refused patch leaves the question pending

A refusal means the author has not answered yet, not that the gap has been dealt
with. The question is not recorded in the `AskedLog` and the loop re-offers it.
Enforced by `test_a_refused_patch_leaves_the_question_pending`.

### M38. Offline, grounding fails closed on any authority

`Gates.offline()` builds a grounding gate with no verifier. Nothing offline can
confirm a citation, and merging one on trust is the failure the gate exists to
prevent, so this is correct rather than inconvenient. Enforced by
`test_offline_grounding_fails_closed_on_an_authority`.

### M39. The CLI is exercised as a CLI

`build_parser` is separate from `main`, and the tests invoke `main()` with real
argv for every subcommand. A CLI whose parser is never exercised can be broken
for every invocation while the suite stays green — which is what happened in
REMEDIATION §11.11c. Enforced by `test_the_parser_accepts_every_advertised_subcommand`,
`test_begin_then_answer_then_render` and
`test_cycle_reads_answers_from_stdin_and_stops_on_a_blank_line`.

### M40. A session survives a crash mid-cycle

State is written with write-then-rename, so an interrupted save cannot truncate a
manuscript, and `cycle` saves after every round rather than only at the end.
Enforced by `test_cycle_saves_every_round_not_only_at_the_end` (which interrupts
mid-loop and reloads from disk), `test_a_save_does_not_leave_a_temp_file_behind`
and `test_a_session_round_trips_through_disk`.

### M41. A live run is gated by gates that can confirm what it produces

`Gates.live(settings)` builds the grounding gate with a corpus verifier and gives
the banality gate the corpus as its view of the literature. Running the live
engine behind `Gates.offline()` refuses every exchange at the fabrication wall
for want of a verifier rather than for want of an authority (REMEDIATION §12.2).
Enforced by `test_live_gates_can_confirm_what_a_live_exchange_produces`.

### M42. The machine's failures do not cost the author their answer

A dialectic exchange that raises is caught in `Session.answer`, and the patch
proceeds carrying what the author wrote — their work is not hostage to a model
being down. An exchange whose authorities cannot be grounded has those nodes
dropped and *named*, not silently discarded, and the author's answer merges
without them. If a node the author wrote fails grounding, the patch still fails
whole. Enforced by `test_a_failed_exchange_costs_the_pressure_never_the_answer`,
`test_the_machines_bad_citation_does_not_cost_the_author_their_answer`,
`test_nothing_ungrounded_reaches_the_manuscript` and
`test_atomicity_gives_way_only_across_parties`.

### M43. The reduced patch is what merges

When nodes are dropped, `graph.apply` receives the patch the gates actually
judged. Merging the original would merge something unjudged — which is what the
code did until the test above caught it.

### M44. The answer is what gets debated, not the question

`debate_prompt` puts the author's answer under test with the Socratic question as
context. Sending the question instead would have the models argue the topic in
general and hand back material about the subject, when what the loop needs is
pressure on the specific thing the author just committed to. Enforced by
`test_the_debate_prompt_puts_the_answer_under_test_not_the_question`.

A thesis proposition that parrots the answer back is compared against it and
refused by novelty — `test_the_machine_cannot_merge_the_authors_answer_back_at_them`.

---

## Maieutic — Designed, not yet enforced

### D6. Novelty thresholds are uncalibrated

`HARD = 0.92` and `SOFT_LOW = 0.70` were chosen by judgement, not measurement.
The dialectic module's independence thresholds were set by scoring two real
populations and putting the cut in the gap between them (REMEDIATION §10.3);
these have had no such calibration, because no corpus of accepted-versus-rejected
manuscript nodes exists yet. Tests pin the *bands' behaviour* using a stub
embedder with exact vectors, so they will keep passing whatever the numbers are.
Calibrate before trusting a merge decision to them.

### D7. Answered: the scorer is not the bottleneck

Run and closed. `evals/compare_support_scorers.py` re-scored the 14 authority
claims from a live session under all three scorers. Every one passed **exactly
one of fourteen** — and no two agreed on *which*. Scores differed on all 14 rows,
so this is three real scorers, not one measured three times. See REMEDIATION §14.

What remains open is narrower and better posed: the support check's verdict on
any individual authority is almost entirely method-dependent (zero overlap in the
positive class across three methods), while the aggregate rate is stable. The
stability is a coincidence of thresholds, not agreement. `SEMANTIC_THRESHOLD =
0.55` sits above almost the whole embedding distribution observed here
(0.11–0.57), which is D6 and D16's problem appearing in a third place.

### D8. `build_grounding_gate` has no test

It swallows corpus-load and scorer-build failures by design, returning a
fail-closed gate. Nothing exercises the settings-driven path, so a
misconfiguration that silently produces a verifier-less gate would be visible
only as every authority failing to merge.

### D9. Question quality is unmeasured

The gaps are structural and therefore real, but nothing establishes that the
questions they produce are *good* ones — that an author answering them writes a
better paper than one answering a generic prompt. The templates are
deterministic and blunt by design; `QuestionWriter` exists so a model can do
better, and no model has been run through it. Judging this needs an author, not
a test.

### D10. Gap coverage is a class, not a list

`GapKind` covers the holes visible in the graph's current shape. An argument can
fail in ways this shape cannot express — a premise that does not in fact support
what it points at, a distinction without a difference, a section that argues the
wrong question well. Those are semantic and would need a critic, with all the
self-satisfaction risk that carries. Nothing here claims the list is complete.

### D11. The adapter has never seen a live dialectic turn

Every `DialecticTurn` in `tests/test_maieutic_adapter.py` is hand-built. The
adapter reads the real pydantic models, so a schema change breaks the tests —
but nothing shows what a real exchange's propositions look like once they are
nodes: how many survive the gates, whether the antithesis produces objections an
author finds worth answering, or how much of a turn is refused at the boundary in
practice. Running one is the next real measurement this module needs.

### D12. The answer's node type comes from the gap, not from the answer

An author asked for the strongest objection to their claim gets whatever they
write typed as an `OBJECTION` attacking it, even if they in fact wrote further
support. The graph then carries a wrong edge, and no gate catches it: grounding
checks that support exists, not that an ATTACKS edge really attacks. Classifying
the answer instead of the question would need a critic reading the author's
prose, with the self-satisfaction risk that carries; the alternative is letting
the author set the type, which is a UI decision not yet made.

### D13. Promoting an advisory finding to blocking needs calibration, not a code change

`_SEVERITY` is a table, so flipping `SUPPORT_THAT_CONTRADICTS` to BLOCKING is one
line. Doing it without first measuring the critic's false-positive rate on real
manuscript nodes would hand an uncalibrated model a veto over the author's work,
which is the failure §9.5 and §11.2 both record. The measurement needed is the
same one the dialectic module's independence thresholds got (§10.3): score two
real populations, and look for a gap.

### D14. Coherence checks relations, not reasoning

The gate can tell that an edge's asserted relation is contradicted by the text.
It cannot tell whether a premise that does not contradict its target actually
*supports* it — most irrelevant material is neutral, not contradictory, and
neutral passes. A premise that is merely beside the point merges cleanly.

### D15. The banality gate is the weakest of the four, deliberately

It blocks on one decidable thing and reports the rest. That is the honest
position given what is uncalibrated here, but it means the gate cannot enforce
the property it is named for: a confident truism passes. "Context matters",
"there are arguments on both sides", "the doctrine is complex" all carry real
content words, have no close corpus twin, and hedge very little. Catching them
needs a critic with a sense of what is worth saying, which is a judgment no
threshold in this file encodes.

### D16. `HEDGED_AT` and `COMMONPLACE_AT` are uncalibrated

0.34 and 0.88, chosen by judgement like the novelty thresholds and with the same
missing prerequisite: no corpus of accepted-versus-rejected manuscript claims
exists to set them against. The tests pin the *verdicts* using worked examples
and a stub embedder, so they will keep passing whatever the numbers are. One
worked example ("Weights may arguably be expressive under the First Amendment")
lands at 0.333 against a 0.34 cut — close enough that a small wording change
flips its verdict, which is what an uncalibrated threshold looks like from the
inside.

### D17. The hedge list is English legal-academic prose only

`HEDGES` is a hand-maintained word set. It has no notion of hedging by
construction ("one might think that some would argue"), and a hedge it does not
list reads as content. Like the reporter table in the dialectic module, it is
maintained rather than derived.

### D18. Section order is sorted, not authored

The graph records which section a node belongs to but not what order the sections
go in, so `render.sections` puts unsectioned material first and sorts the rest
alphabetically. That is deterministic, which is what the tests check, and it is
not an outline. A real manuscript needs an authored section order, and nothing
here supplies one.

### D19. The prose is assembled, not written

Each node becomes its own paragraph. There are no transitions, no subordination,
no argumentative connective tissue beyond the structural labels — a rendered
graph reads as a sequence of assertions rather than as an essay. This is the
honest consequence of refusing a model at this step, and it means the render
output is a faithful view of the argument rather than a draft of the paper.
Closing that gap without reintroducing fabrication risk is an open design
question, not a TODO.

### D20. The live loop has been run once, by hand

Superseded in part. `--live` now runs the real engine and a live exchange has
merged: four nodes from one answer, two ungrounded authorities dropped, ~2m45s
across four local models. What that establishes is that the path works, not that
it works well. One answer on one thesis is an anecdote, not a measurement, and
nothing here records how often a live exchange yields objections worth answering,
how many of its authorities survive grounding, or what the loop feels like over a
whole paper. The eval harness measures the dialectic module in isolation; there
is no equivalent for the loop.

### D21. Nothing prevents the author from being asked in a bad order

The policy asks about the worst structural hole first, which is not the same as
asking in the order that helps an author think. A paper is written by developing
one line of argument, and this engine will happily jump between sections because
a cycle in one outranks an unanswered objection in another. Whether that is
disruptive enough to matter is a question for someone using it, not a test.

### D22. The loop eval's authors are stand-ins

`evals/loop_sessions.json` holds scripted answers written *outside* the loop, so
the machinery faces text it did not produce — which is what stops the harness
being the self-satisfying arrangement that has already cost this repository three
gates. What it establishes is that the loop works on plausible input. It does not
establish that the loop helps a real author: nothing here measures whether the
questions were worth answering, whether the objections sharpened the argument, or
whether the paper got better. Those need a person writing a real paper. The
fixture format carries a `source` field on every session and the reports print
it, so a run over stand-ins and a run over a real transcript are never silently
compared.

### D23. Depth is a function of the machine's pressure — measured both ways

Offline, both fixture sessions run dry after two exchanges, with three of five
and two of four answers unused. The offline gap sequence is fixed: an uncontested
thesis, then the unanswered attack the author's own answer created, then nothing.

Live, neither session ran dry — all four and all five answers were used, because
each exchange's objections create the next round's gaps. The mechanism is
confirmed in both directions. See §13 for the run.

### D24. Two gate weaknesses the first run made concrete

Both were already recorded as design limits; the harness turned them into
observations.

* **A confident truism merges.** "Context matters a great deal and there are
  arguments on both sides" clears the banality gate, because it has real content
  words and hedges very little. D15 predicted this; `S3-thin` now demonstrates it
  on every run.
* **A near-verbatim restatement can squeak under the hard threshold.** "The
  regulatory treatment of open weights is unsettled at present" against a thesis
  of "The regulatory treatment of open weights is unsettled" scores just under
  `HARD = 0.92` on lexical overlap, lands in the soft band, and — with no
  entailment critic configured offline — is admitted as unadjudicated. This is
  D6 and the soft-band rule interacting, and it is the strongest argument yet
  for calibrating those thresholds against real data.

### D25. The gate is working; the generator is not

**1 of 19** in the first live sessions, and four explanations have now been
tested. It is **not the scorer** (§14: lexical, embedding and NLI all pass 1 of
14), **not the prompt framing** (§15: premise-citing gives p = 0.33 and costs
four-fifths of the objections), and **not the corpus** (§16: rebuilt from real
opinion text, end-to-end survival 1 of 10 — unchanged).

What remains is that the retrieval-plus-model pipeline proposes citations that
genuinely do not support their claims. A 5–10% survival rate is a measurement of
how often a 7–8B local model plus vector retrieval over 40 records produces a
citation that survives contact with its own source. The fabrication wall refusing
them is the wall working.

The number worth trying to move is the generator's. The obvious control — the
same loop against a frontier model — has not been run.

### D29. Do not switch to the full-text corpus without switching the scorer

`lexical_support` is recall of the claim's tokens in the passage with no length
penalty, so it is monotonically non-decreasing as passages grow. The rebuilt
corpus has passages at a median of 141 words against 12, which raises every
lexical score mechanically: a passage about statutory severability, padded with
legal vocabulary and saying nothing about the claim, scores 0.429 against a 0.34
threshold. `support_scorer` defaults to `lexical`, so adopting
`openweights_fulltext.jsonl` as the default corpus while leaving that alone would
make the fabrication wall weaker, not stronger. Enforced by
`test_lexical_support_cannot_fall_as_a_passage_grows` and
`test_unrelated_text_can_clear_the_lexical_threshold_on_length_alone`.

### D28. Groundability and informativeness pull against each other

Surfaced by §15 rather than designed. A proposition general enough for an
existing source to support is close to a truism; a proposition specific enough to
advance the argument is an extension no source states. The banality gate and the
grounding gate therefore squeeze AUTHORITY nodes from both sides, and no prompt
resolves it — it is a property of the task. Worth stating because it bounds what
the citation machinery can ever deliver: authority for the *steps*, not for the
conclusion.

### D26. The loop generates open problems faster than it closes them

S1: 13 machine objections over 5 answers, 5 of them answered, **10 open problems**
in the rendered manuscript. Each answer closes one attack and provokes two or
three more. The author falls steadily behind, by construction — the Socratic
policy always offers the oldest unanswered attack, and the exchange always adds
new ones. Whether that is productive pressure or an unfinishable treadmill is a
question for someone using it, and it is the strongest reason to want a real
author's transcript rather than more fixture runs.

### D27. The advisory limbs are dark even in live mode

Every live exchange reported zero advisories, because `Gates.live` wires the
grounding verifier and the banality corpus but **no entailment critic**. So the
novelty soft band stays unadjudicated and coherence's semantic checks never run,
in exactly the mode where a real NLI model is available. Given D24 — a
near-verbatim restatement admitted through the soft band — this is a concrete,
small gap: the dialectic engine already builds an NLI client, and `Gates.live`
could pass it to `NoveltyGate` and `CoherenceGate`. Recording it rather than
fixing it blind, because the change alters what merges and would need
re-measuring against these numbers to mean anything.

### D30. Retrieval leaves headroom, and closing it naively makes the gate decorative

Measured, not designed (REMEDIATION §18). Retrieval selects the best-available
corpus record 1–2 times in 10, and under the embedding scorer roughly 3 of 10
groundable claims are lost to that choice. So D25's "the generator is the
bottleneck" is too strong: retrieval is a separate contributor.

The fix that suggests itself — rank records by the scorer the grounding gate uses
— reaches the ceiling by construction and destroys the gate. Retrieval would
propose only what the gate must accept, and the gate would report a pass forever
while checking nothing. Any retrieval work here has to improve on a signal the
support check does not use, and has to be audited for that alignment; the ceiling
tooling measures the gain but cannot detect the collapse.

### D31. The retrieval headroom is real and still open

§19 tried the one legitimate closure D30 allows — reranking on headnotes, title
and metadata, none of which the support check reads — and it moved five of six
measured cells not at all. Topical aboutness does not predict passage-level
support. The gap between what retrieval picks (1/10) and what is reachable (4/10
under embedding) stands, and no candidate fix has survived measurement.

The independence check is the part worth keeping: `agrees_with_ceiling` reports
how often the reranker picks the gate's own favourite, and ran at 0–1 of n. A
future retrieval change that closes the gap *and* raises that number has not
improved retrieval, it has aligned it with the gate.

### M45. The learned policy reads the author, never the gates

`learn.Episode` carries the gap kind, the outcome, the section and the answer's
length — and nothing about whether the patch merged. Ranking questions by whether
their answers passed the gates would have the machinery grade its own curriculum,
which is §18's collapse one layer up and more tempting here because the verdicts
are sitting in `StepResult`. A refused answer is recorded as ANSWERED: the author
wrote a paragraph and the machinery rejected it, which is a fact about the gates.
Declining is the only negative signal, because only the author can decide a
question was not worth their time. Enforced by
`test_a_refused_answer_still_counts_as_a_question_worth_asking`,
`test_the_journal_records_nothing_about_gate_verdicts` (asserts the field set) and
`test_declining_is_the_only_negative_signal`.

### M46. Evidence nudges within a bound; the declared order dominates at distance

An adjustment is capped at `MAX_SHIFT` positions. Adjacent kinds can swap on
strong evidence — intended, since neighbouring priorities were close to a
judgement call — but nothing travels further, so a `DEPENDS_ON` cycle the author
keeps skipping still outranks an unverified citation they enjoy fixing. No kind
is suppressed outright: one that stopped being asked could never earn its way
back. Below `MIN_EVIDENCE` a kind is *unmeasured*, not unwanted — the §11.8
distinction again. Enforced by `test_adjacent_kinds_can_swap_on_strong_evidence`,
`test_evidence_cannot_travel_further_than_max_shift`,
`test_the_adjustment_is_bounded_by_max_shift`,
`test_no_kind_is_ever_suppressed_outright` and
`test_engagement_is_unmeasured_below_the_evidence_floor`.

### M47. The journal outlives the manuscript

What is learned is a fact about the *author*, so it persists separately from the
session. It also has to: an offline manuscript runs dry after two or three
questions (D23), which is below the evidence floor for any single gap kind, so a
per-manuscript journal could never accumulate enough to act on — a feature that
would have shipped and never once activated. Enforced by
`test_evidence_accumulates_across_manuscripts`, which drives three manuscripts
through the CLI and asserts engagement becomes measurable, and by
`test_the_journal_persists_separately_from_the_manuscript`.

### M48. The author's channel is the only way into the graph

`POST /answer` writes a HUMAN node because the request *is* the author speaking.
There is deliberately no endpoint accepting a node, a patch or a provenance: a
client able to post machine-drafted text as human-authored would defeat the
record the whole system keeps, and no care at the UI layer would restore it.
Enforced by `test_there_is_no_endpoint_that_accepts_a_node_or_a_provenance`,
which inspects the route table rather than a hand-written list, and
`test_an_answer_becomes_a_human_node`.

### M49. Session ids are validated before they reach the filesystem

Each session is a file named from a client-supplied id. Ids are generated
server-side and matched against `^[0-9a-f]{12}$` on the way back in, so a
traversal attempt is refused before any path is built. A local-first tool is
still a tool with an HTTP server in it. Enforced by
`test_a_crafted_session_id_never_reaches_the_filesystem` (parameterised over
traversal, encoded traversal, and malformed ids) and
`test_a_traversal_id_cannot_write_outside_the_session_directory`, which asserts
on the filesystem rather than the status code.

### M50. A refused patch is a 200 with reasons

The author asked a question and got an answer about their work, which is a
successful interaction whatever the merge decided; a 4xx would tell the client
something went wrong when nothing did. The question stays pending, matching M37.
Enforced by `test_a_refused_patch_returns_200_with_its_reasons` and
`test_a_refused_answer_leaves_the_question_pending`.
