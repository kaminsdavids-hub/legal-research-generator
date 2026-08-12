# Dialectic Module — Remediation Log

This file records blockers, dead ends, fixes, and deviations encountered while building `modules/dialectic/`.

## 1. Referenced existing modules not present

The task references `modules/socratic/` and `modules/general/` as existing copy-Q&A modules. In this working tree neither directory exists (verified with `find_file_by_name`, `git ls-files`, and `grep`).

* **Impact:** The Stage 4 copy-parity implementation could not mirror the transport/state of those modules directly; it was modeled on the existing `src/legal_research/multi_chat.py` and `src/legal_research/chat_grounding.py` patterns instead. The final Socratic comparison was also blocked.
* **Mitigation:** Implemented `modules/dialectic/` as a self-contained package with its own state, transport, and copy affordances. The absence is noted here per standing rules.

## 2. CourtListener rate limits

Verified against the Free Law Project documentation on 2026-08-05. The authenticated-user default remains:

* 5 requests / minute
* 50 requests / hour
* 125 requests / day

These values are implemented as the defaults in `modules/dialectic/verification.py`. No change from the values supplied in the task.

## 3. Project-wide `make typecheck` failures (pre-existing)

Running `python -m mypy src modules/dialectic` surfaced 14 pre-existing type errors in `src/legal_research/` that are unrelated to the dialectic module:

* `src/legal_research/citations/retriever.py`
* `src/legal_research/citations/support.py`
* `src/legal_research/chat_grounding.py`
* `src/legal_research/multi_chat.py`
* `src/legal_research/agents/interviewer.py`
* `src/legal_research/agents/editor.py`
* `src/legal_research/api/app.py`

Per the task's instruction not to refactor existing modules, the `validate` target was scoped to lint and typecheck only `modules/dialectic`, while leaving the original `typecheck` target unchanged. `modules/dialectic/` itself has zero mypy errors.

> **Amended 2026-08-07.** In the current environment `make typecheck` does not
> reach those 14 errors: `mypy src modules/dialectic` aborts with an
> `INTERNAL ERROR` inside a third-party package
> (`.venv/.../transformers/models/dia/processing_dia.py:85`, mypy 1.14.1) and
> exits 2 having reported nothing about this repository's code. Verified
> pre-existing by stashing all `src/` edits and re-running: byte-identical
> crash. So the "14 errors" figure is inherited from an earlier environment and
> is currently unverifiable — `make typecheck` is not a working gate here.
> `make validate`, which uses the scoped `typecheck-dialectic` target, is
> unaffected and passes with zero errors.

## 4. CitationChannel tuning

Initial `X v. Y` regex was case-insensitive over the whole pattern, which let it match narrative words like "the" and "is" and produced false positives on ordinary sentences. The pattern was narrowed so party names must start with an uppercase letter, and the connector is the only case-insensitive part. After this change the adversarial test reported:

* **False negatives:** 0 / 171 (0.00%)
* **False positives:** 0 / 59 (0.00%)
* Malformed strings that still contained a valid reporter substring were flagged 3 / 8 times, which is the intended behavior.

## 5. Correlation guard results

> **Corrected 2026-08-07 (see §9.8). The numbers below were an artifact of the
> harness and are retained only as a record of what was originally claimed.**

~~Using the mock NLI evaluator and deterministic clients:~~

* ~~Distinct-family mean crux count: **1.0**~~
* ~~Same-family mean crux count: **0.0**~~

~~The collapse confirms that the crux signal is sensitive to the thesis/antithesis family difference; the family guard enforces this in production.~~

**What was actually measured.** The same-family arm was constructed as
`_SameFamilyClient(antithesis_client, thesis_client.name)` — it wrapped the
antithesis client and relabelled it with the thesis's name, so *both roles ran
the same client*. The test then supplied `same_family_json = antithesis_json`,
i.e. byte-identical output for both sides. The reported collapse from 1.0 to 0.0
therefore measured that identical text does not contradict itself. That is true,
and it is not evidence about model families.

Two further problems with the original harness:

* `_run_with_clients` bypassed both the family guard and verification, so the
  cruxes it counted were computed over slots whose `status` was always
  `pending` — a different code path from production, where
  `PrecedenceRule.rank` reads `status`.
* The `_outcome_bearing` weight gate (§9.5) was still in place, so the counts
  were additionally filtered by the debaters' self-declared weights.

The corrected result from a live re-run is recorded in §9.8.

## 6. Final Socratic comparison

Because no `modules/socratic/` copy-Q&A module exists, the comparison used the existing `SocraticInterviewer` from `src/legal_research/agents/interviewer.py` as the closest available proxy. Twenty rounds were run with a pro-thesis seed and twenty with an anti-thesis seed; the concatenated interviewer questions were compared against the dialectic crux propositions.

* **Result:** 0 of 2 dialectic crux propositions were recoverable from the Socratic output (0%).

This low recovery supports keeping the dialectic module; the Socratic interviewer surface does not surface the same adversarial, outcome-bearing contradictions.

## 7. No container/model pulls performed

No Docker images or model weights were pulled during this session, so the `docker manifest inspect` check was not triggered.

> **Superseded 2026-08-07 for the §9 pass.** One model weight *was* pulled:
> `llama3.2:3b` (2.0 GB), with the maintainer's explicit approval, because the
> corrected correlation-guard experiment (§9.8) requires two genuinely different
> checkpoints of one family and only `llama3.1:8b` was present locally. No
> Docker images were pulled. The alternatives were rejected on the record: the
> two `saul` tags resolve to the same blob ID (`20c063b1eec8`), so they are one
> checkpoint under two names; and `qwen2.5-coder:1.5b-base` is a base model that
> would fail JSON parsing, which would have produced a *low same-family crux
> count for the wrong reason* — a spurious confirmation of the very claim under
> test.

## 8. Deployment: wiring the module to the Spark and Netlify

### 8a. `gpt-oss:20b` cannot serve a debate role (blocker, fixed)

The first live exchange returned `(generation failed or contained a citation string)` for the thesis. Direct probing showed `gpt-oss:20b` returns an **empty content string** through the Ollama OpenAI-compatible endpoint — it is reasoning-only and its output goes to a channel the endpoint does not expose. Every turn therefore failed to parse and burned the whole regeneration budget.

* **Fix:** `dialectic_thesis_model` now defaults to `hermes3:8b` (family `hermes`), which returns clean JSON. Roles remain four distinct families: hermes / llama / gemma / nemotron.

### 8b. Citation leak through `court_hint` (defect, fixed)

With `hermes3:8b` live, the model emitted a clean `proposition` but placed **"New York v. Harris"** in `court_hint`. The Stage 0 scan only read `proposition`, so the citation passed. This defeats the invariant: a model blocked from citing in one field simply routes the citation through another.

* **Fix:** `_generate_position` now scans `court_hint` as well as `proposition`. The prompt was tightened to state that no field may contain a citation. Regression test: `test_citation_in_court_hint_is_rejected_like_one_in_the_proposition`.

### 8c. Regeneration re-ran instead of re-rolling (defect, fixed)

Rejected turns were retried at `DecodingPolicy.COLD` (temperature 0, fixed seed 7), so every retry reproduced the offending output byte-for-byte. The regeneration budget was spent without ever changing the input.

* **Fix:** attempt 0 stays deterministic; retries use temperature 0.7 and a per-attempt seed. Regression test: `test_regeneration_varies_decoding_so_retries_are_not_identical`.

### 8d. rsync path error clobbered a source file (self-inflicted, fixed)

`rsync src/legal_research/api/ spark:.../src/legal_research/` (trailing slash on the source) copied the *contents* of `api/` one level too high, overwriting `src/legal_research/__init__.py` with `api/__init__.py` and leaving stray `app.py` / `schemas.py`. The backend then failed with `ModuleNotFoundError: No module named 'legal_research.app'`. Restored `__init__.py` from the local copy and removed the strays. Recorded here because the failure looked like a packaging bug and was not.

### 8e. Live results

* Endpoint: `POST /api/sessions/{id}/dialectic`, exposed in the UI as the **Dialectic** tab.
* Live URL: `https://legal-research-generator-app-932.netlify.app`
* A live exchange produced 1 thesis proposition and 2 antithesis propositions, all `pending`, and **0 cruxes** — ~~correct, because the thesis self-assigned `supporting` weight and only `controlling`/`persuasive` propositions are outcome-bearing.~~

  > **Corrected 2026-08-07.** The observation was right; calling it *correct*
  > was not. A modest model self-assigning a low weight silently disabled the
  > feature the module exists to provide, and the UI rendered an empty table
  > with no explanation. The weight is self-declared by each debater about its
  > own argument, so gating extraction on it hands the model a switch that turns
  > off the analysis of its own output. Fixed in §9.5; re-running the same
  > question live now yields 1 crux where the gate previously yielded 0.
* ~~`calls_spent: 0` and every slot `pending`/`not_found`, because `LRG_COURTLISTENER_TOKEN` is unset on the Spark. This is the specified behavior: no slot is marked `VERIFIED` on a path that did not receive a 200 with a cluster. Set the token to enable verification.~~

  > **Corrected 2026-08-07.** This diagnosis was wrong. The zero verification
  > calls were **structural, not a missing token**. Setting
  > `LRG_COURTLISTENER_TOKEN` changes nothing on its own: `verify_position`
  > selects `filled = [slot for slot in position.propositions if
  > slot.normalized_cite]` and returns early when that list is empty, and
  > *nothing in the module ever populated `normalized_cite`*. There was no step
  > between `court_hint` (a plain-English description, by design) and a
  > candidate citation. For any spec-compliant exchange `filled` was always
  > empty, the network was never touched, and every slot stayed `pending`
  > forever — with or without a token. The missing retrieval stage is built in
  > §9.3.
* The `modules` package is now included in the wheel (`pyproject.toml` → `packages = ["src/legal_research", "modules"]`) so the Spark venv can import it.

## 9. `modules/dialectic/` correctness pass (2026-08-07)

A defect-driven pass over the module. Every item below was reproduced by
executing the shipped code before it was changed.

### 9.0. `probe.py` did not exist

The work order describes `probe.py` at the repo root as re-running all the
defects and printing CONFIRMED / not reproduced per item. **No such file was in
the tree** (checked with `ls`, `find`, and `git ls-files`).

* **Impact:** acceptance criterion 3 could not be evaluated as written.
* **Mitigation:** written from scratch as part of this pass, against the
  *unfixed* code, and verified to report CONFIRMED for all 8 items before any
  fix landed. Items 1–7 are defects (CONFIRMED = still broken); item 8 is a
  control on the documented precedence rule (CONFIRMED = correct behaviour
  intact). Numbering follows the work order's own acceptance criterion rather
  than the W-numbers: items are W1, W3, W4, W5A, W5B, W6, W8, and the control.
  W2, W7, W9 and W10 are covered by tests rather than by the probe — W7's
  ordering and W9's experiment are not expressible as a boolean reproduction
  check.
* Final state: all 7 defects report `not reproduced`; the control still reports
  `CONFIRMED`.

One probe was rewritten mid-pass, which is worth recording because rewriting a
probe to make it pass is exactly the wrong move. Item 5 (W5 Defect B)
originally asserted `_has_negation(_normalize(text))` — an internals-level
symptom. The chosen fix (§9.4) makes `_has_negation` take raw text, so that call
became a misuse of the new contract rather than a test of it. It was retargeted
at the observable consequence instead: a contraction-negated proposition is not
recognised as contradicting its affirmative twin. **The replacement was verified
to reproduce on the unfixed code first** — it returned `entailment` for a flat
contradiction, a worse result than the documented `neutral`.

### 9.1. `normalized_cite` bypass closed (W1)

`_try_parse_slots` read `normalized_cite` straight off the model's JSON, so a
live reporter citation placed there passed the channel intact — the same defect
class as the `court_hint` leak (§8b), in a third field. Confirmed: `573 U.S. 373`
survived while a direct scan of the same string raised.

* `normalized_cite` is no longer accepted from a model at all. A non-empty one
  raises `CitationDetected(raw_cite, [raw_cite])`, so the turn is discarded and
  regenerated exactly as a citation in `proposition` would be.
* `_POSITION_PROMPT`'s `HARD RULE` now names the three keys that are read
  (`proposition`, `court_hint`, `weight`) and states that no other key is read.
* **Note on the call site:** `CitationDetected` is not a `ValueError`/`KeyError`,
  so raising it from the parser would have escaped `_generate_position`'s
  existing `except` clause and propagated out of `chat()`. The retry loop was
  restructured to catch both in one place.
* Regression test:
  `test_citation_in_normalized_cite_is_rejected_like_one_in_the_proposition`.

### 9.2. The end-to-end verification test replaced (W2)

`test_end_to_end_turn_with_clean_clients_and_verification` was the only test
proving verification worked, and it worked by having the fake model emit
`normalized_cite="392 U.S. 1"` — it certified the §9.1 bypass. Replaced by two
tests that split the two claims it was conflating:

* `test_spec_compliant_exchange_spends_no_verification_calls` — the honest
  behaviour with no retriever configured: zero verification calls, every slot
  explicitly unverified with the reason stated.
* `test_retrieval_stage_feeds_courtlistener_verification` — the path that
  legitimately asserts `calls_spent > 0` and `SlotStatus.VERIFIED`, via the new
  retrieval stage.

### 9.3. The missing retrieval stage built (W3)

New `modules/dialectic/retrieval.py`: a `CiteRetriever` Protocol
(`propose(court_hint, proposition) -> list[str]`), a deterministic
`StubCiteRetriever` for offline tests, and a `NullCiteRetriever`. The concrete
corpus-backed adapter (`_CorpusCiteRetriever`) is in `service.py`, the only
module allowed to import `legal_research.*`; it wraps
`citations/retriever.py` and `citations/corpus.py` **without modifying them**.

* `SlotStatus.PROPOSED` added for the state between generation and verification,
  so `pending` (nothing proposed) and `proposed` (candidate awaiting lookup) are
  distinguishable. `copy._slot_marker` renders a `proposed` slot with an
  `[UNSUPPORTED: ...]` marker plus the unconfirmed candidate — an unverified
  candidate never renders clean.
* `DialecticChat.chat` calls retrieval between generation and verification. With
  no retriever configured, slots stay `pending` **and say so**: the note reads
  "unverified by construction: no retrieval stage configured...", surfaced
  through the copy serializers rather than left as a bare `pending` that reads
  as work in progress.

### 9.4. NLI client wired, and the heuristic fixed (W4, W5)

`service.py` constructed `nli_client=make(settings.dialectic_nli_model)` and
`DialecticChat` stored it, but `CruxExtractor()` was built with no arguments, so
`NLIEvaluator(client=None)` ran the heuristic on both branches. Confirmed:
`engine.nli_client='nemotron-mini:4b'` while `crux_extractor.nli.client=None`.

* `DialecticChat.__init__` now builds `CruxExtractor(nli=NLIEvaluator(client=nli_client))`.
* `NLIEvaluator.relate` implements the model path: a constrained classification
  parsed strictly as `{"label": "entailment"|"contradiction"|"neutral"}`.
* On parse failure it falls back to the heuristic **and records that it did**, as
  `Crux.nli_source`. `relation()` is kept as the label-only API so existing
  callers and tests are unaffected.
* `_assert_family_distinct` now covers the NLI model and reports every colliding
  pair by role name.

The heuristic had two defects, both confirmed:

* **A.** `shared_terms = set(p.split()) & set(h.split())` over text whose
  stopwords were never dropped, so the rule reduced to "exactly one side
  contains a negation word ⇒ contradiction". 3/3 unrelated legal-prose pairs
  were flagged. Now: content-word overlap (stopwords and negation tokens
  dropped) must reach a ratio threshold **and** the negation must reach a shared
  term, via a negation-scope window. 0/3 after the fix.
* **B.** `_has_negation` matched `won't`, `n't`, `isnot`, `arenot`, `doesnot`,
  but was only ever called on `_normalize`d text, and `_normalize` destroys
  apostrophes. Of the two options offered, **`_has_negation` is now applied to
  raw text** and the unreachable apostrophe-less alternatives (`isnot`,
  `arenot`, `doesnot`) were deleted. Consistent: every call site passes raw text.
  Worth noting the live severity — the pair "The court applies Miranda..." /
  "The court doesn't apply Miranda..." classified as **`entailment`**, not the
  documented `neutral`.

**A third heuristic defect, found while fixing the first two and not in the work
order.** Morphological antonyms carry no negation *word*, so
"...is constitutional" / "...is unconstitutional" read as a same-polarity
near-paraphrase with high content overlap, and the overlap-based entailment
shortcut returned **`entailment`** for a flat contradiction before the antonym
table was ever consulted. Two changes: the antonym check now runs before the
entailment shortcut, and antonym matching is word-boundary aware rather than a
substring test — `"constitutional" in p` is also true of "unconstitutional", so
the substring form would have called two sentences that *both* say
"unconstitutional" a contradiction. Regression test:
`test_nli_handles_morphological_antonyms`, which covers both directions.

### 9.5. The weight gate no longer disables the crux engine (W6)

* `Weight` and `SlotStatus` are real `StrEnum`s, and `CitationSlot.weight` /
  `.status` are typed with them, so out-of-vocabulary values now fail
  validation. Previously `class X(str)` with class attributes left the fields
  typed as bare `str` and `extra="forbid"` gave no protection.
* Weight parsing is case-normalized. A model emitting `"Controlling"` was
  silently stored as `supporting` and dropped out of crux extraction with no log
  line anywhere. A weight that must be coerced is now recorded in the slot's
  `note`.
* The `_outcome_bearing` filter is removed from `extract`. Cruxes are extracted
  across all proposition pairs and `PrecedenceRule.classify` partitions them —
  it already returned `open` for exactly this case. `Crux.outcome_bearing`
  preserves the distinction for ranking, and results are sorted
  outcome-bearing-first.
* `DialecticTurn.crux_note` explains an empty table (how many pairs were
  compared), and the serializers render it instead of showing nothing.

### 9.6. Synthesis: ordering, prompt, retries (W7)

* `chat()` is reordered to retrieve → verify → extract cruxes → synthesise, and
  `copy_crux_table(turn)` is passed into the synthesis prompt. The prompt asks
  the model to "name the decisive crux", which was previously generated *after*
  the synthesis ran.
* Removed the duplicated-thesis construction. `_generate_synthesis` built
  `copy_exchange(turn.model_copy(update={"antithesis": turn.thesis, ...}))`,
  rendering the thesis into the antithesis slot and feeding that alongside the
  real JSON for both sides — the model saw the thesis three times and the
  antithesis once.
* Removed the doubled system prompt (`_SYNTHESIS_PROMPT` was both the system
  message and appended to the user message).
* Synthesis now gets the same retry-with-re-roll loop as positions, and its
  failure string carries the reason. Given how strict the channel is, a
  single-shot synthesis failed routinely rather than rarely.

### 9.7. Failure diagnostics preserved, and the smaller items (W8, W10)

`note=last_error if "last_error" in dir() else ""` — `last_error` was only
assigned in the JSON-parse `except`, never in the `CitationDetected` branch, so a
position that failed three times on citations returned `note=""`,
indistinguishable from a parse failure. (`dir()` also defeats mypy's
possibly-unbound analysis.) Now `last_error` is initialised before the loop, the
`CitationDetected` branch records `exc.hits` and the attempt index, and
`DialecticTurn.regenerated` — which previously existed and was never assigned
anywhere — carries the real retry count across both positions and synthesis.

**Conflict between W8 and acceptance criterion 4.** W8 asks for `note` and the
attempt count to be surfaced "in the API response", but `DialecticResponse` and
the endpoint that builds it live in `src/legal_research/api/`, which criterion 4
forbids modifying. Resolved without touching `src/`: the diagnostics are surfaced
through the `copy_exchange` / `copy_crux_table` strings, which are *already*
fields of `DialecticResponse`, so the information does reach the API payload.
`copy_exchange` now always reports the verification calls spent, adds
"Regeneration attempts spent: N" when any were, and renders each slot's `note`
inside its `[UNSUPPORTED: ...]` marker. The same constraint applies to the new
`Crux.outcome_bearing` / `Crux.nli_source` and `DialecticTurn.crux_note` fields:
they are on the domain models and in the copy output, but adding them to the
typed `DialecticCrux` / `DialecticResponse` schemas needs a `src/` change that
this pass is not permitted to make. **Follow-up:** whoever lifts that restriction
should add `regenerated`, `crux_note`, `outcome_bearing` and `nli_source` to
`src/legal_research/api/schemas.py`, since the copy strings are a workaround, not
a typed contract.

Smaller items:

* `verification.py` logs the block sent and the keys returned, and a slot that
  falls to `NOT_FOUND` because no result matched its key now says so explicitly
  (with the counts of keys returned vs cites sent) instead of
  "CourtListener status: no_match".
* `RateBudget._prune` uses a `collections.deque` and `popleft()` instead of
  `list.pop(0)`.
* `ContentCache.__contains__` takes the lock — it was the only unsynchronised
  read of `_data`.
* `copy._slot_marker` makes the partially-populated verified slot explicit: a
  `verified` slot with an empty `normalized_cite` now renders a marker rather
  than nothing at all. It remains unreachable in normal operation.
* `channel._REPORTERS` had `N.E.2d` and `N.E.3d` twice each; deduplicated, with
  a regression test. **Dead end:** the work order suggests generating the list
  from a table in `src/legal_research/citations/bluebook.py`. There is no such
  table — that module formats citations from a record's `reporter` field and
  holds no reporter list. The list stays hand-maintained; this is recorded in a
  comment at the definition.

**Decision — the daily rate budget does not survive restarts.** `RateBudget`
uses `time.monotonic()`, whose epoch is process start, so the 24-hour window and
the 125/day cap are enforced *per process*, not per calendar day. This is now a
documented choice rather than an accident: persisting it needs a durable store
keyed on wall time, the module is deployed as a single long-lived backend
process, and CourtListener enforces its own server-side limit as the real
backstop. If the daily cap ever has to hold across restarts, `_history` must
move to a persistent store — noted at the class and in OBSERVABLES D5.

### 9.8. Correlation guard re-run (W9)

See §5 for what the original numbers actually measured. The harness was rebuilt
before re-running:

* `_SameFamilyClient` is deleted. `correlation_guard_report` now takes an
  explicit `distinct_pair` and `same_family_pair`, so the same-family arm cannot
  be a relabelled copy of the other client.
* Both arms run the **production path** (`chat()`: retrieval, verification, crux
  extraction, synthesis) via `_arm`, which suppresses only the family guard —
  deliberately, since the same-family arm exists to run the configuration the
  guard forbids. The old `_run_with_clients` bypassed the guard *and*
  verification.
* It returns a `CorrelationGuardReport` carrying per-question counts and both
  model pairs, not a bare `(float, float)`. `separates` is a derived property,
  so the raw data is inspectable rather than collapsed to a verdict.
* `test_correlation_guard_shows_crux_collapse_with_same_family` asserted the
  empirical claim from canned identical strings. It is replaced by
  `test_correlation_guard_runs_both_arms_through_the_production_path`, which
  tests the harness mechanics and **asserts no empirical claim** — a unit test
  with fixed strings cannot answer this question, and the old one only appeared
  to because it was rigged.

`scripts/correlation_guard_experiment.py` runs the real experiment against live
local models.

#### The live result

Run 2026-08-07 on local Ollama, 8 contested legal questions, both arms through
the production path, 60.0 min wall clock:

```
distinct    hermes3:8b / llama3.1:8b : mean 0.38, total 3  [1,1,0,0,0,1,0,0]
same-family llama3.1:8b / llama3.2:3b: mean 0.25, total 2  [0,1,1,0,0,0,0,0]
questions: 8; arms differ on 3 question(s) — in BOTH directions
```

**The crux counts do not separate.** Three cruxes versus two, over eight
questions. Fisher exact, two-sided, on questions-yielding-a-crux (3/8 vs 2/8):
**p = 1.000**. The two arms disagree on three questions and they disagree *in
both directions* — two favour the distinct-family pairing, one favours the
same-family pairing. That is the signature of noise, not of an effect.

Per the work order this result stands as recorded; the experiment was not
adjusted until it confirmed the design. The reported `separates: True` is a bare
direction comparison of 0.38 against 0.25 and is **not** evidence. To stop that
flag being quoted as confirmation the way §5's numbers were, the
`CorrelationGuardReport.separates` docstring now says so explicitly, `summary()`
prints "direction only, NOT a significance test", and the report exposes
`disagreements` so a both-directions split is visible at a glance.

**What this does and does not license.**

* It does **not** support the claim that same-family instances have correlated
  errors. That claim is now unsupported by any evidence in this repository: the
  original number was an artifact (§5) and the corrected re-run is null.
* It does **not** refute the claim either. The experiment is badly
  underpowered — 8 questions, and a base rate so low that the distinct-family
  arm produced a crux on only 3 of them. A null at this n is uninformative
  about a real effect of plausible size.
* The **family guard stays enabled**, on the narrower and honest justification
  that three distinct families is a cheap, low-risk default for an adversarial
  design — not on the strength of a measured collapse. OBSERVABLES §8 records
  that the guard is enforced while its empirical justification is not
  established, which is the accurate state.

**The more interesting finding is the base rate.** Both arms produce a crux on
well under half the questions (3/8 and 2/8) with these 8B-and-below local
models. Whether the module's central output appears at all is dominated by model
capability, not by family pairing. Before the family question is worth
re-running, the yield itself needs attention.

**To make the family question answerable**, a rerun needs: more questions (~50+
rather than 8), several same-family pairs rather than one, repeats per question
to separate decoding variance from family effects, and a crux base rate high
enough to have power — likely larger models. That is a multi-hour job and was
not in scope here. The command is:

```
python scripts/correlation_guard_experiment.py --same-family llama3.1:8b llama3.2:3b
```

**One caveat on the same-family arm.** `llama3.1:8b` and `llama3.2:3b` are
different checkpoints of one family but also different *sizes*, so the arm
confounds family with capability. The cleanest same-family pair would be two
checkpoints of equal size; none was available locally.

## 10. Follow-ups from the §9 pass (2026-08-07)

Two items §9 flagged and could not close: the typed API contract (blocked by
acceptance criterion 4) and the crux base rate (identified but not diagnosed).

### 10.1. The derived fields are now a typed API contract

§9.7 routed `regenerated`, `crux_note`, `outcome_bearing` and `nli_source`
through the pre-rendered `copy_*` strings, because `DialecticResponse` lives in
`src/legal_research/api/` and criterion 4 forbade touching it. With that
restriction lifted, they are proper fields:

* `schemas.py` — `DialecticCrux` gains `outcome_bearing: bool` and
  `nli_source: str`; `DialecticResponse` gains `regenerated: int` and
  `crux_note: str`. All four are documented at the field.
* `app.py` — the endpoint populates them from the turn.
* Regression tests: `test_dialectic_endpoint_exposes_derived_fields` and
  `test_dialectic_endpoint_reports_an_empty_crux_table_reason` in
  `tests/test_api.py`, both driven by a fake engine so they run offline.

The tests also pin that the `Weight`/`SlotStatus` `StrEnum`s serialize as their
values (`"supporting"`, `"pending"`), not as `"Weight.SUPPORTING"` — a real risk
when an enum crosses into a schema field typed `str`.

**A wrinkle worth recording:** `legal_research/api/__init__.py` does
`from .app import app`, so the package attribute `app` shadows the `app`
*submodule*. Both `from legal_research.api import app as m` and
`import legal_research.api.app as m` bind the FastAPI instance, not the module,
and monkeypatching module globals on it fails with a confusing
`'FastAPI' object has no attribute '_dialectic_chat'`. The tests reach the module
through `sys.modules["legal_research.api.app"]`.

### 10.2. The crux base rate: the two sides were arguing the same position

§9.8 reported a crux on only 3 of 8 and 2 of 8 questions and attributed it to
model capability. That was a guess, and it was wrong.

`scripts/crux_yield_diagnostic.py` dumps both sides' propositions and the NLI
label for every cross-product pair. Over 4 questions (baseline, before the fix):

```
proposition counts (thesis, antithesis): [(1,2), (1,2), (1,1), (2,2)]
pairs compared: 9
  contradiction   2  (22%)
  entailment      4  (44%)
  neutral         3  (33%)
cruxes: Q1=1, Q2=1, Q3=0, Q4=0
```

**44% of all pairs were entailment — the two debaters agreeing with each other.**
The clearest case, Q3:

* thesis: "The exclusionary rule should **not** apply when evidence was obtained
  in good-faith reliance on a subsequently invalidated warrant."
* antithesis: "The exclusionary rule should **not** apply when officers
  reasonably relied on a warrant that was later deemed defective."

The same on Q4 (both sides argued a state *may* compel collection). The NLI pass
was right to report entailment; there was no contradiction to find. Q1 and Q2
produced a crux only because the antithesis happened to emit an opposing claim
alongside an agreeing one.

**Root cause.** Thesis and antithesis were generated independently from the same
question, neither saw the other, and nothing in `_POSITION_PROMPT` assigned a
side — it said only "You are the {side} in a legal dialectic". Asked a contested
legal question, two competent models converge on the same answer. A dialectic
whose sides agree yields no cruxes *by construction*, and no amount of NLI
tuning fixes it. This is the same defect family as §9.5: the module's central
output was gated on something nothing actually enforced.

**Fix**, in `engine.py`:

* `_STANCE` assigns each role an explicit side — thesis argues the affirmative,
  antithesis argues the negative and is told not to agree, restate, or hedge.
* `_REBUT_PROMPT` shows the antithesis the thesis's propositions and requires it
  to contradict them *directly*, addressing the same predicate. `chat()`
  generates the thesis first and passes it in.
* `_rebuttal_block` returns empty when the opposing position failed generation,
  so the `(generation failed...)` placeholder is never fed back to another model
  as if it were an argument.
* The prompt now asks for 2-3 propositions stated as complete, directly
  affirmable sentences, and clarifies that `weight` describes the strength of
  the *authority*, not the arguer's stance or confidence — a thesis had been
  labelling its own argument `contra`, which is what `weight` is least meant to
  express.

Regression tests: `test_each_side_is_told_which_way_to_argue`,
`test_antithesis_is_shown_the_thesis_to_contradict`,
`test_failed_thesis_is_not_fed_back_as_an_argument`.

**Consequence for the family guard.** The antithesis now conditions on the
thesis, so the two debaters are no longer independent draws. §9.8's experiment
compared independent draws; any rerun measures a different quantity and its
result is not comparable to the numbers recorded there. This strengthens rather
than weakens the case for reading §5/§9.8 as settled-null-and-superseded.

**Channel safety.** The propositions quoted into the antithesis prompt have
already passed the citation channel, so echoing them cannot introduce a citation
the antithesis did not author. The antithesis's own output is scanned as before.

#### Measured effect

Same 4 questions, same models, before and after:

| | before | after |
|---|---|---|
| proposition counts (T, A) | (1,2) (1,2) (1,1) (2,2) | (3,3) ×4 |
| pairs compared | 9 | 36 |
| **entailment** (the sides agreeing) | **4 (44%)** | **1 (3%)** |
| neutral | 3 (33%) | 8 (22%) |
| contradiction | 2 (22%) | 27 (75%) |
| cruxes per question | 1, 1, 0, 0 | 4, 9, 6, 8 |
| questions yielding ≥1 crux | 2 / 4 | 4 / 4 |

**The diagnosed defect is fixed.** Entailment — the direct measure of "the two
sides are arguing the same position" — collapsed from 44% to 3%, and every
question now yields at least one crux instead of half of them yielding none.

**The raw crux count overstates the gain, and should not be quoted as 2 → 27.**
Two artifacts inflate it:

* *Combinatorial multiplication.* Both sides now emit 3 propositions instead of
  1-2, so the cross-product grew from 9 pairs to 36. More pairs alone produces
  more cruxes.
* *Within-side redundancy.* Each side's propositions are often near-paraphrases
  of each other. Q4's thesis says "physical presence is not always required",
  "remote sellers can have nexus through economic activity alone", and "the
  physical presence rule has been abrogated" — three phrasings of one claim.
  Crossed against a similarly redundant antithesis, that single disagreement
  yields 8 cruxes. The honest count for Q4 is roughly *one* disagreement
  reported eight times.

So the module now reliably finds the disagreement it previously missed, but the
crux table contains near-duplicate rows and the count is not a measure of how
many distinct issues are in dispute.

**Open follow-up: deduplicate the crux table.** Cruxes whose thesis and
antithesis propositions are mutual near-paraphrases of an already-reported pair
should be collapsed, or the table should group by disputed predicate and report
a count. This is deliberately *not* done here: it changes what a crux means in
the output and what the UI ranks, which is a design decision rather than a
correctness fix. `Crux.outcome_bearing` already gives the UI a ranking signal in
the meantime. Until it is done, read the crux count as "how many contradicting
pairs", not "how many issues are in dispute" — and note that
`test_empty_crux_table_states_the_reason` covers the empty case but nothing
covers the over-full one.

**A second-order effect worth watching.** The antithesis frequently negates the
thesis mechanically rather than mounting an independent counter-argument — "The
Supreme Court has recognized ..." becomes "The Supreme Court has not recognized
...". That is a genuine contradiction and the NLI pass is right to flag it, but
it is a weaker adversarial signal than a distinct opposing theory, and it is a
direct consequence of `_REBUT_PROMPT` demanding the same predicate be addressed.
Trading "the sides agree" for "the antithesis mirrors the thesis" is an
improvement, not a solution.

### 10.3. The antithesis now builds an independent counter-theory

§10.2 fixed "the two sides argue the same position" and recorded the side effect
it introduced: the antithesis contradicted the thesis by inserting "not" into the
thesis's own sentence.

```
thesis:     "The Supreme Court has recognized that modern cell phones ..."
antithesis: "The Supreme Court has not recognized that modern cell phones ..."
```

That is a real contradiction and the NLI pass is right to flag it, but it is a
degenerate one. A mirror concedes the thesis's framing, its predicate, and its
choice of authority, and disputes only the sign. Nothing is learned from it that
the thesis did not already assert, and it is precisely the *absence* of an
adversarial second opinion that the module exists to supply.

The cause was the prompt: `_REBUT_PROMPT` literally asked for "the negation of
one of them, addressing the same predicate". It got what it asked for.

**Prompt.** `_REBUT_PROMPT` now asks the antithesis to build its own theory of
the case, forbids polarity-flip restatement explicitly, and requires each
proposition to be grounded in a *different* doctrine, standard, test, or line of
authority than the thesis relies on — attacking the framing (choice of rule, the
analogy, the scope of an exception, the standard of review, a skipped threshold
question) rather than the conclusion.

**Guard.** Prompting was not enough on its own the last time — the same mistake
that produced §9.5 and §10.2 — so independence is *enforced*, in the spirit of
`channel.py`. New `modules/dialectic/independence.py`:

* `IndependenceGuard.measure` reports a `Mirror` when two propositions differ in
  polarity *and* carry essentially the same vocabulary. Both conditions are
  required: shared vocabulary alone is expected (the sides are arguing about the
  same thing) and a polarity difference alone is what real disagreement looks
  like. It is the combination that identifies a restatement.
* Words are stemmed before comparison so "applies"/"apply" do not read as
  different vocabulary, and negation markers are dropped from the content set so
  the inserted "not" does not itself lower the overlap.
* Propositions under 5 content words are exempt: a terse claim ("No warrant is
  required") legitimately shares nearly all its vocabulary with its opposite.

**Thresholds are calibrated against observed data, not guessed.** Scoring the
mirrors and the counter-theories from the live run:

| | jaccard | containment |
|---|---|---|
| real mirrors (n=4) | 0.55 – 1.00 | 0.81 – 1.00 |
| independent counter-theories (n=4) | 0.00 – 0.12 | 0.00 – 0.25 |

The two populations separate with a wide gap, so `MIN_JACCARD = 0.45` and
`MIN_CONTAINMENT = 0.65` sit in the middle of it rather than at the edge of
either. Both populations are fixtures in `tests/test_dialectic.py`
(`_REAL_MIRRORS`, `_INDEPENDENT_THEORIES`) so a threshold change that collapses
the separation fails the suite.

**Retry carries feedback.** A rejected draft is re-rolled with
`_MIRROR_FEEDBACK`, which names the offending propositions. Re-rolling on
temperature alone re-runs the same mistake — the lesson of §8c applied to a new
failure mode.

**It degrades visibly rather than failing closed.** Unlike the citation channel,
independence is a quality property, not a safety one: a mirrored antithesis is
worth more than no antithesis. When every attempt mirrors, the least-bad draft
is kept and each mirroring proposition is flagged in its `note`
("independence guard: this merely negates the opposing proposition ..."), which
flows into the copy payloads and the API. The reader is told the antithesis
conceded the framing rather than being shown a contradiction that is not one.

The thesis is not subject to the guard — it is generated first and has nothing
to mirror.

Tests: `test_independence_guard_catches_polarity_flip_mirrors`,
`test_independence_guard_allows_a_real_counter_theory`,
`test_independence_guard_needs_both_shared_words_and_flipped_polarity`,
`test_independence_guard_ignores_propositions_too_short_to_judge`,
`test_independence_guard_scan_raises_and_names_every_mirror`,
`test_independence_guard_is_inert_without_an_opposing_side`,
`test_antithesis_that_mirrors_is_regenerated_with_feedback`,
`test_persistent_mirroring_degrades_visibly_rather_than_failing_closed`,
`test_thesis_is_not_subject_to_the_independence_guard`.

**Refactor.** Stopwords, negation markers, tokenization and overlap metrics moved
to `modules/dialectic/textnorm.py`, shared by the NLI pass and the guard. They
must agree on what "same words" means: the guard exists to catch exactly the
pattern the NLI pass reports as a contradiction, so two definitions could
contradict each other. `NLIEvaluator` keeps its `_normalize` / `_has_negation` /
`_content_words` methods as thin delegates, since `probe.py` calls them.

#### Measured effect of the independence guard

Same 4 questions and models across all three configurations:

| | baseline (no stance) | stance only (§10.2) | stance + independence guard |
|---|---|---|---|
| **residual mirrors** | — | high (qualitative) | **0 / 12 (0%)** |
| entailment (sides agreeing) | 44% | 3% | 6% |
| contradiction | 22% | 75% | 31% |
| neutral | 33% | 22% | 64% |
| cruxes per question | 1, 1, 0, 0 | 4, 9, 6, 8 | 4, 3, 1, 3 |
| questions yielding ≥1 crux | 2 / 4 | 4 / 4 | 4 / 4 |
| regenerations spent | 0 | 0 | 0, 1, 2, 0 |

**The guard works, and the model can satisfy it.** Zero of twelve antithesis
propositions still mirror. The guard fired on two of four questions and
`llama3.1:8b` produced an acceptable independent draft on the retry both times —
it did not simply burn the budget, which was the live risk.

**The propositions are genuine counter-theories, not metric evasion.** A guard
that scores vocabulary overlap could in principle be satisfied by rewording a
mirror. It was not. Q1's antithesis argues three distinct doctrinal grounds:

1. *textualist* — "The Fourth Amendment's 'effects' clause does not encompass
   digital information ... it was intended to protect tangible property only";
2. *arguendo, different doctrine* — "**Even if** the 'effects' clause were
   interpreted broadly, the warrantless search ... would be justified as an
   exception to the warrant requirement under the 'necessity' doctrine";
3. *institutional competence* — "any such protection is a matter of legislative
   intent rather than constitutional law".

The second concedes the thesis's premise and wins on other ground, which is real
appellate argument and the structural opposite of a polarity flip. Q3 is
similar: the antithesis reframes the exclusionary rule as deterrence rather than
remedy, attacks probable cause as the threshold question, and invokes judicial
integrity — none of which appears in the thesis.

**The honest trade-off: contradiction fell from 75% to 31%, and neutral rose
from 22% to 64%.** This is not a regression. The 75% was inflated by mirrors,
which are contradictions by construction — flipping the sign of a sentence
guarantees the NLI reports a contradiction. Genuine independent theories often
address a *different predicate* than the thesis, and the NLI correctly rates
those `neutral` rather than `contradiction`. So the crux count fell from 27 to
11 while the number of *distinct* disagreements went up.

Entailment stayed low (3% → 6%), so this did not reintroduce the §10.2 defect of
the two sides agreeing.

**A tension worth naming.** Independence and direct contradiction pull against
each other. The more genuinely independent the antithesis's theory, the less
likely it maps onto the same predicate as any thesis proposition, and the fewer
pairs the NLI can call contradictions — Q3 dropped to a single crux for exactly
this reason. The module currently resolves that tension in favour of argument
quality over crux count, which is the right default for a tool whose output a
lawyer reads. It does mean the crux count is not a quality metric and should not
be optimised: a configuration that maximises cruxes is one that maximises
mirrors.

This also partly addresses the §10.2 near-duplicate problem without the
deduplication step: eliminating mirrors removed most of the redundancy that made
one disagreement report as eight cruxes. The follow-up remains open, since
within-side redundancy in the *thesis* is untouched.

**Cost.** 64.9 min for 4 questions versus 46.1 min without the guard, the
difference being the retries on two questions.

## 11. Building the eval harness (2026-08-08 / 09)

A 40-question First Amendment eval set was supplied for the module. Standing it
up surfaced ten further defects. They are recorded together because the
pattern matters more than any one of them: **five would have shipped to users
regardless of any eval, and four were in the eval machinery itself — where a
bug does not break anything visibly, it just makes the numbers wrong.**

A second pattern runs through the last three: each was a step assumed rather
than tested. The judge was assumed to discriminate until it was calibrated; the
cache was assumed to prevent calls until a mixed block was actually run through
a client that refuses to make them. Both assumptions were cheap to check and
expensive to leave unchecked.

Question A1 failed five times in a row, for five different genuine reasons. At
the third failure the intended report was "an 8B synthesis model will not act on
a status flag" — a negative result about model capability. That would have been
wrong, and it would have been an artifact of the harness, which is the same
error class §5 of this file already records. It was avoided only by testing the
two candidate explanations separately rather than iterating on the fix.

### 11.1. The eval's own premises did not fit the module (blocker, resolved)

The set's first hard gate requires that "every case, statute, regulation and
Federal Register document cited must exist in the retrieval log". The module is
*forbidden* from emitting citations: a proposition naming a case is discarded
and regenerated. Tested against the live channel, **12 of 14 sampled must-engage
anchors** — Bernstein, Junger, Sorrell, Reed, TikTok v. Garland, Lamont, HLP,
McCullen, NAACP, Near, West Virginia v. EPA, Defense Distributed — would cause
the turn to be thrown away if a debater wrote them.

Resolved by measuring the gate against the *resolved slots and a retrieval log*
rather than the prose. That is what the module actually claims to do: the
proposition is citation-free, the authority lives in the slot, and
`[UNSUPPORTED: ...]` markers become a first-class eval signal instead of noise.

The corpus was also unusable: 6 records, all securities fraud, containing none
of the ~100 authorities the set names. 34 case authorities were resolved against
CourtListener and 7 non-case records hand-authored and maintainer-verified.

### 11.2. Verified-but-rescinded rendered as clean authority (user-facing)

`copy._slot_marker` returned a bare `[90 Fed. Reg. 4544]` for any slot in
`VERIFIED` state. But **verified means the citation resolves, not that the
authority is still good law.** A rescinded rule that resolved cleanly rendered
to the reader as pasteable authority with nothing indicating it had been
repealed.

This is the same defect class the module already guards against everywhere else
— the `[UNSUPPORTED]` discipline — in the one state that had been assumed safe.
It reached the copy payloads and the UI and was entirely independent of the
eval. The marker now carries the status, and the marker string is a shared
constant so the producer and renderer cannot drift apart and silently stop
matching.

### 11.3. Five Supreme Court authorities were permanently unverifiable (user-facing)

`verify_position` accepted only HTTP `status == 200`. CourtListener answers
`300` when it holds *duplicate records of one opinion*, which it does for
Sorrell, Rice, West Virginia v. EPA, AOSI II and Humanitarian Law Project. Those
five could never verify, and the citation-integrity gate reported them
identically to a fabricated cite.

The ingest script had handled 300 correctly when *building* the corpus; the
runtime path had not. So the corpus contained authorities the module could never
confirm — a mismatch between how the ground truth was built and how the system
consumes it.

Fixed in two steps, and the first was wrong. Requiring every cluster to carry an
identical case name verified Sorrell but still failed AOSI II, which is stored
as both "Agency for Int'l Dev. v. Alliance for Open Soc'y Int'l, Inc." and
"Agency for Int'l Development v. Alliance for Open Society" — one opinion in two
abbreviation styles, read as two different cases. Sameness is now decided on
**filing date** first and name second. A 300 whose clusters carry different
dates stays `NOT_FOUND`, because guessing which case was meant is exactly the
error the gate exists to catch.

### 11.4. Non-case authority could never be proposed (user-facing)

`_CorpusCiteRetriever._format` emitted a citation only for records with a
complete volume/reporter/page triple, so every statute, regulation and Federal
Register document was unretrievable. The temporal-validity gate is *entirely*
about that material, so it could never have fired.

Those now render as `"<code> <section>"`. That raised a verification question:
CourtListener cannot adjudicate `15 C.F.R. 734.7` no matter how correct the
section is, so requiring a cluster would fail every response that correctly
relies on the EAR published exclusion. Such authorities are verified by the
human-checked corpus instead — but must still appear in the retrieval log, so a
fabricated C.F.R. section still fails.

### 11.5. The temporal gate nearly read its own machinery's output

Nothing carried the corpus's status into the slot, so no role learned a rule was
repealed and the module could not have passed. The obvious fix — propagate the
status into `slot.note` — would have **broken the gate**, because the gate read
`slot.note`. Every question would have passed because our own code wrote the
string the gate was looking for: a gate testing itself, reporting green forever.

Two-sided fix. The status is propagated for *rendering* (real correctness,
independent of the eval) and the gate now reads **model-authored prose only** —
propositions and synthesis, never the note. In practice that tests the
synthesis, the only role that runs after retrieval; that is the honest scope and
is documented in the gate's docstring.

Annotation also had to move *after* verification, since `verify_position`
rewrites `note` on all four of its branches and was silently destroying it.

### 11.6. A log wrapper silently disabled the annotator

The temporal gate failed three consecutive live runs while the module was
correct. `LoggingRetriever` wraps the retriever to capture the retrieval log but
forwarded only `propose()`. The engine probes for the optional annotator with
`getattr(retriever, "annotate", None)`, so wrapping turned annotation off
entirely.

A decorator that quietly drops an optional capability is worse than one that
never had it: the probe is designed to degrade gracefully and cannot distinguish
"not supported" from "supported but hidden behind a wrapper".

Diagnosed by testing the halves separately — the annotation plumbing works when
exercised directly, and both `gemma3:4b` and `hermes3:8b` acknowledge a
rescission when the note is genuinely in their prompt. Both healthy in
isolation, so the fault was between them.

**An offline test passed while production failed, because the test used the
adapter directly and production goes through the wrapper.** The test exercised a
path that does not exist in the real system.

### 11.7. The judge could not rank anything

`saul:7b-instruct-v1` scored a response consisting entirely of
`(generation failed or contained a citation string)` placeholders **8.00/10**,
including `crux_identification: 10` for a turn with no crux table, and spread
only **1.00** across deliberately graded fixtures, in the wrong order.

That voided the 8.500 reported from an earlier smoke run. The hard gates are
deterministic code and stood; the 1–10 scores were noise. Left unchecked, the
full run would have produced a plausible ~8.4 and the plateau rule — mean change
< 0.2 across three rounds — would have declared convergence on round one,
because the judge's output barely moves regardless of input.

Of the local models only `hermes3:8b` discriminates: 1.2 / 1.8 / 1.8 / 8.4,
correctly ordered, spread 7.20. `qwen2.5-coder` floors everything at 1.8;
`llama3.1` orders correctly but separates by only 2.80.

`hermes3` was the thesis model and a judge may not share a family with a
debater, so the roles swapped: `saul` argues (it is the legal-domain model,
better suited to argument than to grading rubrics) and `hermes3` judges. It was
checked as a debater first, the same test `gpt-oss` failed in §8a. **Production
defaults moved with the eval defaults deliberately** — an eval that scores a
lineup nobody ships measures the wrong system.

`evals/calibrate_judge.py` is committed so the next judge change is checked
rather than assumed.

### 11.8. A broken judge was scored as zero

D1's judge echoed the crux table back instead of scoring. The harness recorded
that unparseable output as `0.0`, indistinguishable from a failed hard gate or a
worthless response.

The first full run therefore reported **6.519 overall and cluster D at 5.84, the
worst of six**. Excluding that one question: **6.729 overall and cluster D at
7.30, the best**. A single conflation inverted the ranking the eval exists to
produce.

A judge failure is *missing data*. `score` is now `None` when the gates passed
but the judge produced nothing usable, `0.0` only for a real gate failure, and
means are taken over scored results only.

### 11.9. One timeout discarded an entire run

Two repeat runs launched for a variance estimate both died on
`httpx.ReadTimeout` and wrote nothing. Run 2 lost **4.4 hours** of completed
questions to a single slow call. Over a three-hour run that is not an edge case,
it is the expected outcome.

Three causes. The per-call timeout was 300s while Ollama swaps models between
the five roles and a cold load alone costs ~40s — the server log shows a `500`
after `5m09s`; it is now 900s. A failure anywhere killed the process; each
question is now isolated and a dead one is recorded as unmeasured. Nothing was
written until the end; results are now checkpointed after every question.

Fixing this exposed a latent bug: `gates_passed` used `all(self.gates)`, and
`all([])` is `True`, so a crashed question with no gates would have reported as
**passing both of them**.

### 11.9a. Non-case authority was sent to a case-law API (user-facing)

CourtListener indexes case law. A C.F.R. section, a U.S.C. section or a Federal
Register page can never resolve there, and §11.4's corpus-verified path exists
precisely because of that — but nothing stopped those citations being *sent*.

Each wasted a request and returned NOT_FOUND on a perfectly correct citation.
Worse, it poisoned the cache: `PersistentCiteCache` is all-or-nothing per block,
deliberately, so a single unresolvable statute forced a live call for every case
citation beside it. A cache holding all 34 case authorities still missed on any
position that also cited a regulation — which is most of them.

`is_case_citation` matches on the code rather than the shape, because
"90 Fed. Reg. 4544" is volume-reporter-page shaped and would pass any structural
test while being unresolvable. Non-case slots are now left exactly as retrieval
left them rather than marked NOT_FOUND for lacking a record that never existed.

**This cost three attempts at a variance estimate**, each exhausting the 125/day
quota. The pattern in all three was the same and is the lesson worth keeping:
the cache was confirmed complete at 34/34 and that was treated as equivalent to
"the runs will make no calls". It was not, and the step between was never
tested. The check that would have caught it — running a realistic mixed block
through an HTTP client that raises on any call — takes seconds and is now a
test.

Also corrected here: CourtListener's 125/day is a **rolling window, not a daily
reset**. Waiting for the `retry-after` header returns the handful of slots that
aged out at that moment, not a refill, so "wait for the reset then run" is
structurally unsound.

### 11.10. Quota: one full run per day

The two repeat runs were also doomed for a second reason. CourtListener allows
**125 lookups/day** and a full run costs ~64, so the quota had been exhausted by
the first successful run plus the smoke runs. Every citation came back
`not_found` — including ones verified an hour earlier — and the runs were
stopped rather than left to spend nine hours producing sixty-four zeros.

The module's `RateBudget` could not have caught this: it is process-local by
design (§9.7, OBSERVABLES D5) and starts every run believing the full quota is
available. That decision was recorded as low-risk; this is the cost of it
showing up.

`PersistentCiteCache` now stores results on disk, keyed on the **individual
citation** rather than the request's text block, so a later run whose
propositions produce the same authorities in a different order — or a subset —
still hits. Repeat runs cost no quota. Only successful lookups are stored:
caching a 429 would turn a transient outage into a permanent "this citation does
not exist", indistinguishable from a fabricated cite.

### 11.11. Results, the noise floor, and why the threshold moved

Two sets of three consecutive runs, each set on identical code, verification
served entirely from cache so the API is not a variable.

```
set A (11.9 + 11.9a fixes)   means 6.925  6.838  6.831   worst delta 0.087
set B (+ the 11.11a fix)     means 6.769  6.550  6.700   worst delta 0.219
```

Both sets are legitimate measurements of the same unchanged-code question, and
they disagree by a factor of 2.5. **The difference is entirely whether a
question's citation gate happened to flip during the set.** Set A had none; set
B had one, D2 scoring 7.6, 0.0, 7.6. A flip moves a 32-question mean by about
0.24 on its own, which is more than the originally specified 0.2 threshold.

So the honest floor is not a single number. It is roughly 0.09 of ordinary
generation drift, plus about 0.24 for each gate that flips, and whether one does
is a coin toss on any given set.

**The threshold was therefore raised from the specified 0.2 to 0.5** — in
`PLATEAU_DELTA`, in the variance report's default, and in the eval set's own
`harness_rules`, each recording the reason so it is not later "corrected" back.
At 0.2 the rule fires on a single flip; at 0.5 it tolerates one and still
catches two. This supersedes both the earlier 0.193 figure and the claim in the
previous version of this section that 0.2 was safe with a 2.3x margin — that was
true of set A and false of set B, and reporting it as settled was premature.

**Current best estimate: 6.67 ± 0.11, 32 of 32 scored.**

| cluster | mean |
|---|---|
| C — receipt rights, Lamont | 7.33 |
| F — tailoring, less-restrictive alternatives | 7.00 |
| D — deemed exports, academic freedom | 6.76 |
| B — prior restraint, EAR exclusion | 6.43 |
| E — national-security deference | 6.40 |
| A — weights as speech | 6.33 |

#### 11.11a. Non-operative authority disclosed only sometimes

Question D6 acknowledged a rescinded rule on one run and stayed silent on the
next two, failing its temporal gate while the module was otherwise correct. The
synthesis is the only role that sees an authority's status, so it is the only
one that can state it — and §11.5 asked in the prompt without enforcing, which
is the same shape as the mirror problem in §10.3.

It now follows the pattern the rest of the module uses: ask, check, retry with
feedback naming the citation, degrade visibly if it still will not comply.
Verified live — the retry fires once and the model complies in its own words —
and stable at 7.6, 7.6, 7.6 across set B.

**The first version of this fix reintroduced §11.5's self-satisfying gate.** The
fallback warning read "...is no longer operative law...", it was appended to
`turn.synthesis`, and the temporal gate reads `turn.synthesis` — so our own text
would have satisfied the check on every question regardless of what the model
said. It was caught by asking why the gate passed rather than accepting that it
did. The warning now lives on `DialecticTurn.synthesis_note`, rendered for the
reader and invisible to any check of model-authored prose, with a regression
test asserting it cannot pass the gate.

That is the third time in this work that a fix came within one step of making a
check validate itself: the correlation guard measuring identical text (§5), the
temporal gate reading a note the retrieval stage wrote (§11.5), and this. It is
the standing hazard whenever the same code both produces evidence and grades it,
and it always looks like success.

#### 11.11b. Gate flips are a class, not a list

B6 and B7 were fixed by §11.9's retry. D6 was fixed by §11.11a. D2 then flipped
in the next set. Four questions have now flipped across four sets, and each fix
addressed a real cause without exhausting the category.

Chasing them individually looks like it converges and does not. The more durable
answer is to average several runs per optimization round rather than treat one
run as a measurement — the threshold at 0.5 already assumes a flip can land, and
averaging would remove the assumption instead of tolerating it. That is a design
decision about how the eval is used and is left open deliberately.

#### 11.11c. What these numbers do not show

23 to 26 of 32 questions were identical across runs within a set; the rest is
real generation drift. That drift is the floor for this configuration only:
attempt 0 runs at temperature 0 with a fixed seed for both debaters and the
synthesis, and the judge is also temperature 0. Raising any of those raises the
floor.

The judge remains **calibrated, not validated** (§11.7). It separates weak work
from strong; it has not been shown to agree with a lawyer's judgment. For a claim
in a paper, grade a sample by hand and check the correlation.

The variance report itself was wrong until this pass: it compared every result
file ever written, mixing code versions and the quota-poisoned runs of §11.9a
into one figure, and reported a floor of 3.269 where the comparable runs give
0.219. It now defaults to the three most recent runs.

Holdouts A4, A8, B4, C3, D3, E3, F3 and F6 remain unrun.

### 11.12. Environment constraints worth recording

**16 GB RAM cannot hold the five roles.** Loaded footprints run ~1.8× disk size
(`nemotron-3-nano:4b` is 2.8 GB on disk, 5.1 GB resident); the five total ~35 GB
loaded, against capacity for about two. Ollama therefore swaps continuously and
a full run takes ~4.5h. **Raising `OLLAMA_KEEP_ALIVE` does not help** — every
question touches all five roles in sequence, so eviction happens regardless.

The change that would help is batching by role rather than by question: all
theses, then all antitheses, and so on, cutting ~160 model loads to ~5. It is
not done, because the runner would then orchestrate the stages itself rather
than calling `DialecticChat.chat()`, and the eval would exercise a
reimplementation of the production path instead of the path itself. Given four
of the eight bugs above appeared only in the real path, that trade was not
taken.

---

## 12. Defects found by assembling the loop

The maieutic components were each built and tested in isolation and each was
green. Wiring them into `modules/maieutic/loop.py` and driving them through the
CLI surfaced one defect that no unit test could have found, because it was a
property of the composition rather than of any component.

### 12.1. The banality gate was unreachable in the running system

**Symptom.** A CLI session accepted `"It may perhaps arguably possibly seem so."`
as an answer and merged it. The banality gate exists precisely to refuse that.

**Cause.** `BanalityGate.assess` judged only `THESIS` and `ORIGINAL` nodes —
the ones "claiming to be the contribution" — which was written and tested before
the adapter existed. The adapter never produces either type: an author's answer
becomes a `REPLY`, an `OBJECTION` or a `PREMISE` depending on which gap it
answers. So in the assembled loop the gate looked at every node the adapter
produced, found none of them in scope, and passed everything.

Every banality test passed throughout, because they all constructed
`Node.from_human(NodeType.ORIGINAL, ...)` directly — the one shape the loop
never produces. The tests agreed with the code and both were disconnected from
the system.

**Fix.** `_is_a_contribution` now qualifies a node two ways: `Provenance.HUMAN`
(what the author wrote, whatever structural role it plays) or a THESIS/ORIGINAL
type. A vacuous reply is as empty as a vacuous thesis. Machine-proposed premises,
authorities and objections stay exempt, because background and pressure are both
supposed to be unoriginal.

**What it cost, and the general lesson.** Nothing, because the loop had not been
run in anger. The lesson is the one this repository keeps relearning in different
clothes: a component's tests establish that it does what its author thought, not
that it is reachable. §11.6 was a log wrapper that dropped a method; §11.11c was
a CLI no test invoked. This is the same shape — an interface agreed with itself
and with no one else. The first end-to-end run is what found it, and it found it
in the first three commands.
