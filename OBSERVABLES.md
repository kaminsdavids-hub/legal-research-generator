# Dialectic Module — Observable Rules

This file records the rules governing `modules/dialectic/`, split into two
sections:

* **Enforced** — the code implements the rule and a named test in
  `tests/test_dialectic.py` fails if it stops holding.
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
its main cause fixed** — the two debaters were arguing the same side. See rule 11
for what is now enforced, and REMEDIATION §10.2 for the diagnosis.

What remains unenforced is the *rate itself*. Nothing asserts a minimum crux
yield against live models, because that needs a live run and a question set with
known-contested answers. The residual rate after the stance fix is recorded in
REMEDIATION §10.2 as a measurement, not a guarantee.

### D4. The corpus retrieval adapter

`service._CorpusCiteRetriever` maps corpus hits to reporter citations. Only the
offline `StubCiteRetriever` is covered by tests; the corpus-backed adapter has no
test, because exercising it needs a corpus fixture with complete
volume/reporter/page records.

### D5. Daily rate budget across restarts

Rule 4's 125/day cap is process-local by decision, not accident. If it ever has
to hold across restarts it needs a persistent store keyed on wall time. Nothing
tests or enforces the durable interpretation.
