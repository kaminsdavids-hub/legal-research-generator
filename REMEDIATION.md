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

### 12.2. The live engine ran behind gates that could not confirm anything

**Symptom.** The first `maieutic answer --live` run called four models for 2m51s,
produced four verified authorities, and refused all of them:

```
refused — grounding: unresolved_citation — no verifier configured; an authority
cannot merge on trust. Failing closed here is deliberate: this is the
fabrication wall.
```

**Cause.** `--live` swapped in the real dialectic engine but left `Gates.offline()`
in place. The offline grounding gate has no verifier by design, so every
AUTHORITY node the live exchange produced was refused — for want of a verifier,
not for want of an authority. The two halves of the system were configured
independently and disagreed.

**Fix.** `Gates.live(settings)` builds the grounding gate with the corpus
verifier, and hands the banality gate the same corpus as its view of the
literature. The CLI selects gates from the same `--live` flag that selects the
engine, so they cannot diverge again.

### 12.3. All-or-nothing grounding made every live exchange fail

**Symptom.** With the gates fixed, citations resolved but the merge still failed:

```
refused — grounding: unsupported_by_source — 381 U.S. 301 resolves but does not
support this claim (score 0.18 < 0.34)
```

and the author's own answer was refused along with them. Repeating with the
topic-specific corpus rather than the sample corpus reduced the count from four
bad authorities to two but did not change the outcome, which is what ruled out a
misconfiguration.

**Cause.** Two things, one working and one not.

The refusal itself was **correct**: the models retrieved real cases for claims
the source text does not support, and that is precisely the subtler fabrication
the grounding gate exists to catch (OBSERVABLES M6). It had never fired on real
model output before.

What was wrong was the blast radius. Grounding is all-or-nothing across a patch,
a rule written to stop a fabricated claim entering the manuscript beside verified
material. In the assembled loop that rule had a consequence nobody chose: **the
party that cites badly is the machine, and the party that loses their work is the
author.** Their paragraph was discarded because a model retrieved two citations
that did not hold up. Every live exchange failed this way, so `--live` had never
merged once.

**Fix (a deliberate invariant change).** What actually matters is that nothing
ungrounded reaches the manuscript, and dropping the offending nodes secures that
just as well as refusing the patch. `_without_ungrounded_machine_nodes` drops
machine-proposed nodes that fail grounding, names them in the result, and
re-runs every gate on what is left. Atomicity gives way, and only across parties:
if a node the **author** wrote fails grounding, the patch still fails whole. Only
grounding is resolved this way — a coherence failure is structural and a banality
failure is the author's own, and neither is repaired by deleting someone else's
node. OBSERVABLES M5 is amended rather than quietly reinterpreted, and M41–M43
record the new behaviour.

**A bug inside the fix.** The first version reduced the patch for the gate check
and then merged `adaptation.patch` — the original. The gates judged one patch and
the graph received another, so the ungrounded authority merged anyway and
rendered as `(445 U.S. 222 — UNVERIFIED)`. `test_nothing_ungrounded_reaches_the_manuscript`
caught it, and it is the reason that test asserts on the rendered manuscript
rather than on the gate report: a report is what the system believed, and the
manuscript is what it did.

**Result.** The first live exchange to merge: four nodes from one answer — the
author's paragraph, one machine premise and two machine objections — with two
ungrounded authorities dropped and named, in 2m46s across four local models.

**The pattern across §12.** Three defects, all invisible to the component tests
and all found within minutes of assembling the pieces. §12.1 was a gate nothing
routed to; §12.2 was two halves configured apart; §12.3 was an invariant that was
right in isolation and wrong in composition. None of them were reachable by
testing a component against its own idea of its interface.

---

## 13. First live measurement of the loop

`python evals/run_loop_eval.py --live`, two fixture sessions, four local models
(saul:7b-instruct-v1 / llama3.1:8b / gemma3:4b / nemotron-3-nano:4b), corpus
`data/corpus/openweights.jsonl`. Answers are stand-ins, not a real author's
(OBSERVABLES D22).

| | S2-expressive-conduct | S1-export-control |
|---|---|---|
| exchanges | 4 | 5 |
| ran dry | no | no |
| answer survival | 100% | 100% |
| machine objections | 5 | 13 |
| objections per exchange | 1.25 | 2.60 |
| authorities grounded | 1 / 14 | 0 / 5 |
| nodes in manuscript | 13 | 30 |
| open problems | 3 | 10 |
| wall time | 944s | 1469s |

**What went right.** Answer survival is 100%, where before §12.3 every live
exchange scored 0. The citation channel recorded **zero** boundary refusals
across nine live exchanges — the dialectic module's central invariant held
without exception on real model output. Novelty delta stayed high (0.67–1.00),
so the machine's contributions were genuinely new to the manuscript rather than
restatement.

**What the numbers expose.** Authority survival of 1/19 (D25) and open problems
accumulating faster than they are closed (D26) are both first-class findings, and
neither was visible from any component test. The advisory limbs reported nothing
at all, because `Gates.live` never wires an entailment critic (D27).

**Cost.** Roughly four to five minutes per answer on this hardware — 236s and
294s per exchange respectively. A ten-answer session is an hour. That is a
constraint on how the loop can be used, not an implementation detail.

---

## 14. D7: the support scorer is not why authorities fail

**Question.** §13 measured 1-in-19 authority survival and left three explanations
unseparated: the models cite badly, the corpus is small, or `LexicalSupportScorer`
is too crude a proxy for support. This isolates the third.

**Method.** One live session (S2, 4 exchanges, 14 proposed authorities, 18
minutes of model time) with the harness recording each authority's claim and
citation. The same 14 pairs were then re-scored under all three scorers — no
models re-run, so the scorer is the only thing that varies.

**The trap, avoided deliberately.** `build_support_scorer` silently returns
`LexicalSupportScorer` when the gpu extra is missing, and `sentence_transformers`
was in fact absent. Setting `LRG_SUPPORT_SCORER=embedding` and reporting "no
difference" would have measured lexical three times and produced a confident
wrong answer — the §11.6 and §11.11c shape again. The comparison therefore checks
each scorer's concrete class against what was requested and raises rather than
falling back. `sentence-transformers 5.7.0` was installed from the declared extra
first.

**Result.**

| scorer | class | threshold | supported |
|---|---|---|---|
| lexical | LexicalSupportScorer | 0.34 | 1 / 14 |
| embedding | EmbeddingSupportScorer | 0.55 | 1 / 14 |
| nli | NliSupportScorer | 0.55 | 1 / 14 |

Scores differed on **all 14 rows** (0 identical), so these are three real
scorers. But the three passes are three *different* claims: pairwise agreement on
the positive class is **zero**. The stable aggregate is a coincidence of where
each threshold happens to fall, not agreement about what support is.

**The finding that was not on the list.** Inspecting the three survivors, all
look like false positives. Lexical and embedding both matched a *model weights*
claim to Bernstein's passage about *encryption source code* — the passage
supports the source-code proposition, and extending it to weights is exactly what
the paper is arguing. NLI scored 0.969 on a pair that is simply unrelated.

That points at a structural cause: **an AUTHORITY node attached to the paper's
novel claim cannot be grounded, by construction.** If the corpus supported the
claim it would be COMMONPLACE under the banality gate's own definition, and the
paper would have nothing to say. The dialectic engine is attaching citations to
the extensions rather than to the established propositions the extensions rest
on. That is a question of what the engine is asked to cite for, not a threshold
to tune — and it is why raising or lowering a cut would not have helped.

**What this does and does not establish.** It closes D7: the scorer is not the
bottleneck, and nobody should spend more time tuning it. It does not establish
the fourth explanation, which rests on one session, 14 claims, a 40-record
corpus, and a legal reading of Bernstein that is defensible but mine. The next
measurement worth taking is whether the engine cites differently when asked for
the *premises* of a claim rather than for the claim itself.

---

## 15. Citing for premises: a measured negative result

**Hypothesis (from §14).** The engine was attaching authority to the paper's
novel claims, which by construction no source supports. Asking it to cite for the
*established propositions* the claim rests on should raise authority survival.

**Method.** Same session, same answers, same corpus, same four models; only
`debate_prompt` differs. The claim arm has a replication — two independent live
runs both produced exactly 14 authorities with 1 grounded — which is what makes a
single premise-arm run readable at all.

| | claim | premise |
|---|---|---|
| authorities grounded | 1 / 14 | 4 / 15 |
| machine nodes merged | 8 | 9 |
| **objections** | **5** | **1** |
| open problems raised | 3 | 0 |
| wall time | 1098s | 758s |

**The headline is not real.** 1/14 against 4/15 is Fisher two-sided **p = 0.33**.
At n≈14 per arm this is indistinguishable from noise, and 7% → 27% should not be
quoted as an effect.

**The survivors are worse, not better.** All four premise-arm passes reduce to
two distinct claims, each appearing twice:

* *"The First Amendment protects freedom of speech and expression"* matched to
  Keyishian's *"Academic freedom as a special concern of the First Amendment."*
  True, and so general it would match almost any First Amendment passage. This is
  what "cite for established propositions" degenerates into.
* *"Expression must be intended for public dissemination to qualify as protected
  speech"* matched to Bernstein's *"Encryption source code as expression
  protected by the First Amendment."* The passage does not say this — the claim
  adds a requirement the source never states. A false positive, and the same
  Bernstein passage that produced the claim arm's false positive.

**And it cost the thing the loop is for.** Objections fell from 5 to 1 and open
problems from 3 to 0. Telling the models not to argue the claim removed most of
the pressure, which is the loop's entire product. Even had the grounding gain
been real, this trade is a bad one.

**The tension the experiment actually exposed.** Groundability and
informativeness pull against each other. A proposition general enough to be
supported by an existing source is close to a truism; a proposition specific
enough to advance the argument is an extension no source states. Prompting cannot
resolve that, because it is a property of the corpus and the task, not of the
wording.

**The likelier bottleneck, and the next thing to fix.** `openweights.jsonl`
stores **headnote-style topic labels, not quotable text**: 68 passages, median 12
words, 52 of them under 20. "Encryption source code as expression protected by
the First Amendment" is a subject heading. A 12-word label cannot support a
specific proposition — it can only topic-match, which is exactly the behaviour
observed: everything sharing First Amendment vocabulary scores moderately,
nothing scores as entailment, and whichever claim shares the most words wins. It
also explains why all three scorers behaved alike in §14 and why NLI returned
near-zero for almost every pair.

Before any further prompt or threshold work, the corpus needs real passage text —
quotable paragraphs from the opinions rather than summaries of them. Until then
no support check over this corpus can measure more than topic overlap, and
OBSERVABLES D25's numbers should be read as a property of the corpus first.

**Disposition.** `cite_for` stays in the code with `claim` as the default. The
premise arm is retained as a measured negative result, not adopted.

---

## 16. The corpus was not the bottleneck either

§15 predicted that headnote-style passages were why authorities fail to ground,
and that real opinion text would fix it. All 34 case records were rebuilt from
CourtListener: 326 passages at a median of 141 words, against 68 at a median of
12. The prediction was wrong.

**Every cell measured.** "Supported" is out of the authorities that arm proposed.

| pairs (retrieval ran over) | scored against | lexical | embedding | nli |
|---|---|---|---|---|
| claim arm (headnotes) | headnotes | 1/14 | 1/14 | 1/14 |
| claim arm (headnotes) | full text | 3/14 | 1/14 | 2/14 |
| premise arm (headnotes) | headnotes | 4/15 | 5/15 | 1/15 |
| premise arm (headnotes) | full text | 7/15 | 2/15 | 3/15 |
| **claim arm (full text)** | **full text** | **1/10** | **1/10** | **0/10** |

The last row is the one that answers the question: retrieval *and* scoring both
over real opinion text, end to end, 4 live exchanges. **1 of 10 grounded** — the
same rate as the headnote corpus produced. Fisher against the 1/14 baseline is
not close to significant.

**The apparent lexical gains are an artifact, and this is the important part.**
`lexical_support` is *recall of the claim's tokens present in the passage*, with
no penalty for what else the passage says, so it is monotonically non-decreasing
in passage length. Replacing 12-word labels with 141-word passages raises every
lexical score mechanically, and taking the max over 12 passages instead of 2
raises it again. Demonstrated: a passage about statutory severability, padded
with common legal vocabulary and saying nothing about the claim, scores **0.429
against a 0.34 threshold**. Two tests now pin this.

**The consequence is operational, not academic.** `support_scorer` defaults to
`lexical`. Switching the default corpus to `openweights_fulltext.jsonl` while
leaving that default in place would make the fabrication wall *weaker* — more
citations would pass on vocabulary overlap alone. That is why
`openweights.jsonl` remains the default here and the rebuilt file ships as an
experimental arm. **Do not switch the corpus without switching the scorer.**

**What is left, having eliminated three explanations.** Not the scorer (§14), not
the prompt framing (§15), not the corpus (§16). What remains is that the
retrieval-plus-model pipeline proposes citations that genuinely do not support
the claims attached to them: Rice v. Paladin — a murder-manual aiding-and-abetting
case — offered for a claim about the legibility of encrypted material; Brown v.
EMA offered for a claim about commercial value against a passage on educating
future generations. These are real cases cited for propositions they do not
contain, which is precisely what OBSERVABLES M6 exists to catch.

**So the gate is working and the generator is not.** A 5–10% authority survival
rate is not a defect in the wall; it is a measurement of how often a 7–8B local
model, plus vector retrieval over 40 records, produces a citation that survives
contact with its own source. Read that way, every experiment in §14–§16 is
consistent, and the number to try to move is the generator's, not the gate's.

**Caveats.** One session per arm, 10–15 authorities each, one corpus, one
hardware configuration. The false-positive readings are mine and not a lawyer's.
None of this has been checked against a frontier model, which is the obvious
control and the one I cannot run locally.

---

## 17. The frontier control: not run, and what was in the way

§16 concluded that a 5–10% authority survival rate is a property of the
generator, not of the gate, and named the obvious control: the same loop against
a frontier model. It has **not been run**. No frontier credential is configured
here — `LRG_LLM_API_KEY` is the 16-character Ollama placeholder and every base
URL points at `127.0.0.1:11434` — and a key must not be pasted into a chat
transcript, for the reason §9 already records about the CourtListener token.

Attempting the setup surfaced two defects that would have blocked the run
regardless of credentials. Both are fixed.

### 17.1. No frontier model could satisfy the family guard

`detect_family` recognised saul, nemotron, hermes, apertus, gemma, llama and gpt,
and returned `"unknown"` for everything else. Two unknowns collide by design — a
deliberate conservatism, since guessing that two unfamiliar names are different
families would let correlated models debate each other. The consequence was that
**every** hosted model mapped to `unknown`, so any lineup with two of them was
rejected as same-family, and Claude versus Gemini was indistinguishable from
Claude versus Claude.

Added cases for claude, gemini, mistral, qwen, deepseek, grok, cohere and phi.
The unknown-collides-with-unknown rule is unchanged and now documented as
deliberate at the point where a future reader would be tempted to loosen it.

### 17.2. One base URL for four roles

Every role shared `dialectic_base_url`, so a role could not be pointed at a
hosted API while the others stayed on Ollama — and with only one frontier
provider available, a mixed lineup is the *only* way to satisfy distinctness.
The comparison could not be configured at all.

Per-role `dialectic_<role>_base_url` and `dialectic_<role>_api_key` now exist,
empty meaning "use the shared pair", so existing single-endpoint configurations
are untouched. A test asserts `build_dialectic_chat` actually reads them: the
setting would be useless if the builder ignored it, which is the §11.6 shape.

### 17.3. Preflight

`evals/preflight_models.py` (`make preflight`) validates a lineup before it costs
anything: distinct families, resolved endpoints, and one minimal completion per
role. A collision fails before any model is called — a test asserts no probe runs
once the lineup is invalid — and `--no-probe` checks configuration with no calls
at all. It prints which *key source* each role resolved to and never a key.

Verified against the current local lineup: four distinct families, four
endpoints reachable, "lineup is ready".

### To run the control

Configure a key outside this transcript, then:

```
export LRG_DIALECTIC_THESIS_BASE_URL=https://api.anthropic.com/v1
export LRG_DIALECTIC_THESIS_MODEL=claude-opus-4
export LRG_DIALECTIC_THESIS_API_KEY=...      # in a shell, not in chat
make preflight
python evals/run_loop_eval.py --live --only S2-expressive-conduct --out frontier.json
python evals/compare_support_scorers.py frontier.json
```

Thesis is the role worth upgrading first: it proposes the propositions that
retrieval then attaches citations to, so it is the arm of §16's conclusion under
test. Antithesis, synthesis and NLI stay local, which also keeps the comparison
honest — only one variable moves. Expect roughly 16 paid calls for four answers,
before retries.

The prediction §16 implies, recorded now so it cannot be adjusted afterwards: if
the generator is the bottleneck, thesis-side authority survival should rise
materially above 1-in-10. If it does not, the remaining suspect is the retrieval
stage, which chooses *which* corpus record to attach and which no experiment in
§14–§17 has varied.

---

## 18. Varying retrieval: there is headroom, and the obvious fix is Goodhart

§17 left retrieval as the last unvaried suspect. Rather than compare retrieval
modes — a narrower question, and one live run per mode — this measures the
**ceiling**: for every authority claim, score it against *every* record in the
corpus and ask whether any record at all would clear the threshold. That is what
a perfect retriever could achieve. No models are called.

Measured over the 10 authorities from the end-to-end full-text run:

| scorer | retrieval's pick | ceiling | recoverable | chose the best record |
|---|---|---|---|---|
| lexical | 1/10 | 8/10 | 7 | 2/10 |
| embedding | 1/10 | 4/10 | 3 | 1/10 |

**A methodological bug, found and fixed before reading anything into this.** The
first version compared the *live run's* grounding verdict — taken with the
lexical scorer — against a ceiling computed with embedding, so a scorer
disagreement would have been reported as a retrieval failure. `actual` now scores
the record retrieval chose with the same scorer as the ceiling. The corrected
numbers are unchanged, which is worth knowing but was not knowable in advance.

**The finding.** Retrieval selects the best-available record 1–2 times in 10, and
under embedding leaves roughly 3 of 10 groundable claims on the table. §16
concluded the generator was the bottleneck; that is now too strong. Retrieval is
a real and separate contributor.

**Two limits on how far to read it.** The lexical ceiling of 8/10 is inflated by
the length confound of §16 — scoring one claim against 40 records of long
passages will find *something* over 0.34 nearly always — so the embedding ceiling
of 4/10 is the trustworthy figure. And the ceiling is an upper bound on
*threshold-passing*, not on genuine support: spot-checking it turns up
`15 C.F.R. 734.7` offered for "no requirement that protected information be
readable", which is the same false-positive pattern as §14.

**Why the obvious fix is wrong.** The natural response is a support-aware
retriever that ranks records by the scorer the grounding gate uses. That would
reach the ceiling by construction — and it would make the gate decorative.
Retrieval would then propose only what the gate is guaranteed to accept, so the
gate would report a pass forever while checking nothing. This is exactly the
self-satisfying arrangement that cost this repository three gates (§5, §11.5,
§11.11a), arrived at from a new direction, and it is the more interesting result
than the headroom itself.

**What legitimately closes the gap.** Retrieval must improve on a signal
*independent* of the gate's: a better index, better query construction, or
reranking on features the support check does not use. The gate then remains an
independent check on retrieval's output rather than a mirror of its objective.
Any change here should be measured against the ceiling *and* audited for whether
it has quietly aligned the two — the ceiling tooling makes the first easy and the
second is a judgement no metric will make.

---

## 19. Reranking on an independent signal: it does not help

§18 required that retrieval improve on a signal the grounding gate does not read,
or the gate stops being a check. The gate reads exactly one thing:
`record.passages`, scored against the claim. So the reranker reads everything
else — the curated `headnotes`, `title`, `court`, `type`, `code`/`section` — and
never passages. Disjointness is enforced, not intended: for the seven non-case
records that have no headnotes, `passages` *is* the gate's input, so those score
on title and code alone.

**A silent data loss found first.** The first run printed "40 records, 0 with
headnotes". `CorpusRecord` is a pydantic model with no `headnotes` field, and
pydantic drops unknown keys, so the rebuilt corpus's headnotes vanished on load
and the reranker had been scoring metadata only. The field is now declared.

This matters beyond the bug: **the broken version scored better.** Claim-arm
embedding showed 5/14 reranked without headnotes and 1/14 with them. Reporting
the first would have been reporting a spurious success, and the only reason it
was caught is that the run prints how many records carry the signal it depends
on. A measurement that cannot say whether its input arrived is not a measurement.

**Result, over three sets of authorities.**

| authorities | scorer | retrieval | reranked | ceiling |
|---|---|---|---|---|
| end-to-end (10) | lexical | 1 | 1 | 8 |
| end-to-end (10) | embedding | 1 | 1 | 4 |
| claim arm (14) | lexical | 3 | 3 | 9 |
| claim arm (14) | embedding | 1 | 1 | 8 |
| premise arm (15) | lexical | 7 | 8 | 13 |
| premise arm (15) | embedding | 2 | 2 | 12 |

Five of six cells are flat; one improves by one. **Headnote and metadata
reranking does not close the gap**, and nothing here justifies shipping it.

**Why it fails, and it is the same wall as before.** A headnote says what a case
is *about*. Support requires that a specific passage entail a specific claim.
Topical aboutness does not predict passage-level support, especially when the
claim is an extension of what the source says — which is D28's tension arriving
for the third time from a third direction.

**The methodology held even though the result did not.** Agreement between the
reranker's pick and the gate's favourite ran at 0–1 of n throughout. That is the
number to watch: it says the signal really was independent. Had reranking hit the
ceiling *and* agreed with the gate's favourite, the right reading would have been
collapse, not success.

**Disposition.** `evals/rerank_experiment.py` ships as a measurement tool with
the independence check built in. No production retriever changes. The headroom
found in §18 is real and remains open; three candidate closures — the gate's own
scorer (§18), prompt framing (§15), and now independent-signal reranking — have
each been tried and rejected on evidence.

---

## 20. `learn/`: the journal was in the wrong place

The learned question policy was built with the journal on `Session`, alongside
the graph and the asked-log. It passed its tests. Driving it through the CLI
showed the policy reporting "not enough evidence yet" for every gap kind after a
full session, and it would have done so forever.

**Cause.** `MIN_EVIDENCE` is 3 episodes per gap kind, and D23 already measured
that an offline manuscript runs dry after two or three questions *in total*. A
per-manuscript journal therefore cannot reach the floor for any kind. The policy
would have shipped, been wired in, passed every test, and never once adjusted
anything.

This is §12.1's shape — a component unreachable in the running system — caught
this time before commit rather than after, and only because the demonstration was
run rather than assumed.

**Fix.** The journal persists to its own file, shared across manuscripts, because
what is being learned is a property of the author and not of one paper. That was
a modelling error as much as a reachability bug: the journal was never manuscript
state. `Session.to_dict` no longer carries it. A test drives three manuscripts
through the CLI and asserts engagement becomes measurable, so the reachability is
pinned rather than argued.

**A docstring corrected in the same pass.** It claimed a structural defect could
not be demoted below a stylistic gap. The bound makes that true of *demotion*,
but a neighbour being *promoted* can still overtake — and the first live
demonstration showed exactly that, `unanswered_attack` rising to −0.50 past
`self_grounding` at 0. The real property is that nothing travels further than
`MAX_SHIFT`, so the declared order dominates at any distance greater than the
bound while adjacent kinds may swap. The prose now says that, and two tests pin
both halves.

---

## 21. The pipeline verified 0 of 61 citations, for two reasons and neither was the gate

Running the full pipeline end to end against the live backend produced a
9,653-word draft, 61 proposed citations, **0 verified, 61 removed**, an empty
table of authorities — and `shippable=True`.

**Where the removals came from.** All 61 failed the same rule, and it was not the
interesting one. Rule 1 (authority must come from retrieval) passed: no
hallucinated cites. Rule 2 (must resolve in the corpus) passed. Rule 3 (quotes
exact) passed. Every removal was Rule 4, support scoring, with the reported
threshold **0.34**.

**0.34 is `LEXICAL_THRESHOLD`, and the configuration said `nli`.** The running
API had `LRG_SUPPORT_SCORER=nli` in its environment and could import
`sentence_transformers`; built by hand on the same box, `NliSupportScorer`
constructs fine and scores 0.994 on a supporting pair. The process had simply
been started at a moment when the cross-encoder could not be built,
`build_support_scorer` caught the exception, returned lexical, said nothing, and
the pipeline held that scorer for the process's entire lifetime. Every
verification since had been token recall wearing the name of entailment.

This is the third time silent degradation has cost this repository an
investigation: §11.6 (a log wrapper that dropped a method), §14 (a scorer
fallback that would have made the D7 experiment measure lexical three times).
The fallback is correct — a missing extra should degrade rather than crash — so
what changed is the silence. `build_support_scorer` now logs at WARNING with the
mode requested, the exception, the substituted threshold, and the fact that
scores are no longer comparable to a semantic run. Four tests cover it, including
that a *chosen* lexical scorer logs nothing.

**The scorer was not the whole story.** Re-scoring those same 61 citations with a
working NLI model gave the same answer: **0/61 under lexical, 0/61 under NLI**.
Consistent with §14, and it pointed at the second cause.

**`LRG_CORPUS_PATH=data/corpus/sample_corpus.jsonl`** — 3.4 KB, unrelated to
export control. The writer produced propositions about deemed exports and the
published-information exclusion; retrieval attached the closest records a sample
corpus could offer; the support check correctly refused all of them. The gate was
working. It was being asked to confirm claims against a corpus that could not
support them. Repointed at `openweights.jsonl` (40 citable records) and the
service restarted, which also rebuilt the verifier with NLI.

**What was fixed here, and what was not.** Fixed: the silent fallback, and the
corpus configuration. Also fixed in the same pass (§20 above): `is_shippable`
returning True when every citation had been removed. Not fixed: the underlying
citation-survival rate, which §14–§19 traced to retrieval and the generator
rather than to any gate, and which no experiment has yet moved.

---

## 22. The propositions were not propositions

§21 fixed the scorer and the corpus. A clean re-run — NLI confirmed active at
threshold 0.55, `openweights.jsonl` loaded — still verified **0 of 122**, every
one refused for "source does not support the proposition", with scores at median
0.01 and maximum 0.02. Not near-misses. Near-zero.

Sampling what was actually being checked settled it:

```
PROP : Analyzing The Impact of Open Model Weights on Regulatory Compliance
       Under the EAR: A Case Study On Deep Learning
PROP : Investigating Potential Security Threats Associated With Sharing AI
       Models Across International Borders: Implications For Open Model Weight...
```

Those are research topics. The Ideator emits titles; `_default_props` passed idea
text straight through; and the same string became both the retrieval query *and*
the `Authority.proposition` the Verifier later asks a source to entail. Asking an
NLI model whether a passage entails a title returns ~0 by construction. **The
gate was working perfectly on an impossible question.**

This is the fourth distinct cause behind a citation-survival number in this
repository, and the first one that is squarely a bug rather than a limit:

| § | cause | verdict |
|---|---|---|
| 14 | the support scorer | not it — three scorers agree |
| 15 | the prompt framing | not it — p = 0.33, and it cost the objections |
| 16 | the corpus text | not it end-to-end |
| 21 | scorer silently degraded to lexical | **real, fixed** |
| 21 | corpus pointed at a 3.4 KB sample | **real, fixed** |
| 22 | proposition was a topic title | **real, fixed** |

**The fix.** The query and the proposition are now separate concerns. A title
still retrieves — it is a serviceable search string — but when the query is not
assertable, the claim the authority is recorded as supporting is the paper's
thesis, which is what it is actually being cited for. `is_assertable` is
conservative and rejects only the shapes the Ideator produces: gerund openers and
case-study suffixes. A false negative costs a citation the thesis as its
proposition, still true of the paper; a false positive puts a title back in front
of the entailment check.

**What this does not fix.** Attaching the thesis is honest but coarse: a source
cited in section four is being checked against the paper's overall claim rather
than against the sentence it actually supports. The architecturally right target
is the drafted sentence carrying the citation, which the Writer knows and does
not record. That is the next thing worth building, and it is a larger change than
this one.

---

## 23. The Writer already knew which claim each paragraph argued, and threw it away

§22 stopped topic titles reaching the Verifier. This fixes the layer beneath it.

`_expand_section` drafts each paragraph from a specific point:

```python
focus = propositions[len(paragraphs) % len(propositions)]
```

That association was discarded, and the citation loop then re-derived one with a
*second, independent* round-robin:

```python
proposition = propositions[i % len(propositions)]
```

The two indices are computed over different sequences and need not agree, so a
paragraph could be — and routinely was — cited for a proposition it had never
been written about. The Verifier then asked a source to support that
proposition. Both round-robins were doing arithmetic where the answer was
already in scope.

The fix is to stop throwing it away: `_expand_section` returns
`(paragraph, point)` pairs and each paragraph is cited for the point it was
actually drafted from. No text analysis, no inference.

**Two heuristics were built and discarded before this.** The first extracted the
paragraph's last sentence, on the theory that the marker sits at the end — it
picks the transition ("The remainder of this Part defends the claim in detail").
The second ranked sentences by content-word count — it picks connective prose,
which is long and content-rich while committing to nothing. A third, scoring
sentences for named authority, then failed on a signpost. At that point the
approach was wrong, not the tuning: guessing which sentence a citation supports
is a real NLP problem, and the answer was available structurally the whole time.
The discarded work is recorded here because "I tried three heuristics and each
failed on plausible input" is the evidence for taking the structural route.

**Measured.** Against the sample corpus, before: 0 of 61 verified. After: **14 of
105 verified**, with propositions like "Private plaintiffs may not maintain
aiding-and-abetting suits under Section 10(b)" — a claim a source can actually
support. `tests/test_pipeline.py` asserts at least one verified citation and now
passes for the right reason rather than because a topic matched itself.

**A bug I introduced and caught.** Renaming the drafting result to `drafted`
shadowed the `drafted = 0` section counter, so every section after the first
silently produced a single paragraph and the manuscript fell from 7,500+ words to
2,142. The word-range test caught it. Renamed to `drafted_paragraphs`.

---

## 24. The nine verifications were one title

The §23 run reported **9 of 39 verified** — the first non-zero result after
0/61 and 0/122. Checking what had verified before reporting it:

```
1 distinct verified proposition:
  Analyze how AI-generated model weights and training data could be
  treated under existing law
```

All nine were the same proposition, and it is a topic title. §22's
`is_assertable` should have caught it; the opener list held the gerund
`"analyzing"` and not the imperative `"analyze"`, so the one form the Ideator
had actually produced was the one form not covered. A title then matched a
passage on shared vocabulary at 0.74 and passed.

Had the headline been reported unexamined, "0 → 9 verified" would have read as
the fix working, when what it measured was a classifier gap.

**Fixed twice, because the first fix was also wrong.** Generating openers from
stems — `stem + ("e", "es", "ing")` — produced `reviewe`, `reviewes`,
`reviewing` and never `review`, so the base form of several verbs stayed
uncovered. That is clever and wrong in the same way the original list was; the
openers are now written out explicitly, with the reason recorded at the
definition so nobody regenerates them. Interrogative openers (`whether`, `how`,
`why`, `what`) are covered too, since the Ideator produces those as well.

**Where this leaves the number.** Offline against the sample corpus the pipeline
verifies 14 of 105, all one proposition — the raw idea, which *is* a claim. That
is a real verification rather than a title matching itself, but "14" overstates
it: it is one proposition cited fourteen times. The honest summary is that the
pipeline can now verify the claims it is given, and is given very few distinct
ones, because the Ideator emits topics and only the thesis survives
`is_assertable`.

**The remaining work is upstream of everything in §21–§24.** The Ideator should
emit claims, not titles. Every fix in this sequence has been damage control on a
generator that produces research topics where the pipeline needs assertions, and
no amount of classification downstream turns a topic into a claim.

---

## 25. The Ideator now emits claims, which is where §21–§24 should have started

Every fix in §21–§24 was downstream damage control. The scorer was degraded, the
corpus was wrong, the proposition was a round-robin topic, the classifier had
gaps — but underneath all of it the generator was producing *research topics*
where the pipeline needed *assertions*, and no amount of classification
downstream turns a topic into a claim.

**The change.** An `Idea` now carries a `claim` alongside its `text`: one
declarative sentence the paper asserts and a source could support or contradict.
The Ideator asks for `TOPIC || CLAIM` per bullet and validates what comes back
with the same `is_assertable` the researcher uses — an unassertable "claim" is
rejected to empty rather than passed on, because an empty claim falls back to the
thesis (which is a claim) while a bad one travels to the Verifier and gets asked
of a source. That is exactly how a title came to account for all nine
"verifications" in §24.

A bullet without the separator still yields a topic, so a generator that ignores
the format degrades to the old behaviour rather than losing the idea.

The researcher and the writer both prefer `idea.claim or idea.text`, so the
proposition retrieved on, drafted from, and verified against is the same
assertion.

**Measured, offline against the sample corpus.**

| | before §25 | after |
|---|---|---|
| distinct propositions put to sources | 4 | 4 |
| of those, assertable claims | 1 | **4** |
| of those, topic titles | 3 | **0** |

Verified citations stay at 14 of 109, all of the thesis. That is the correct
outcome and worth being plain about: the mock's other three claims are invented
("priority of claims explains why...") and the corpus does not support them, so
they fail verification as they should. The fix does not make more citations
verify; it makes the question asked of every source a fair one. Whether a claim
then verifies depends on whether the corpus actually supports it, which is what
the gate is for.

**What is still open.** The gate now receives well-formed questions, and a real
corpus with real claims is what would show whether the answers are any good. The
citation-survival rate on live models remains the open measurement, and §14–§19
should be re-run against this pipeline before anything further is concluded from
those numbers — they were all taken while topics were being verified.

---

## 26. §14–§19 did not need re-running, and the reason is worth recording

§25 closed by saying §14–§19's numbers "were taken while topics were being
verified and should be re-measured". That was wrong, and checking it before
spending the model time is the only reason it did not become a forty-minute
confirmation of a false premise.

**Three checks, all negative.**

*Were the maieutic claims ever topics?* No — **39 of 39** across all three
collected arms are assertable by the same `is_assertable` that rejects the
pipeline's titles. They read "The First Amendment protects expressive material,
including model weights", not "Analyze how weights are treated". The propositions
came from the dialectic engine, which emits assertions; the Ideator, which emits
topics, is not in that path.

*Could the scorer have silently degraded?* No. `evals/compare_support_scorers.py`
was built in §14 precisely to make that impossible: it checks each scorer's
concrete class against what was requested and raises `FellBackToLexical` rather
than proceeding. The guard that §21 later added to production had already been
applied to the experiment.

*Do the §21–§25 fixes touch the loop at all?* No. `modules/maieutic/` imports
nothing from `agents/ideator.py`, `agents/legal_researcher.py` or
`agents/writer.py`.

**So §14–§19 stand as measured.** The scorer is not the bottleneck, the prompt
framing is not, the corpus is not, retrieval leaves real headroom, and reranking
on an independent signal does not close it.

**What was genuinely unmeasured.** Those end-to-end runs used
`support_scorer: lexical` as the live gate — §14 compared scorers by *re-scoring
fixed pairs*, which holds retrieval constant, and no run has ever had NLI both
gating and shaping what survives end to end. That is the one re-run with new
information in it, and it is what was actually run.

**The general point.** "Re-measure everything downstream of a fix" sounds
conscientious and is often waste. The cheap check is which code paths the fix
actually reaches; here that was three greps and a classifier pass over stored
data, against forty minutes of GPU time and a result that would have been
identical.
