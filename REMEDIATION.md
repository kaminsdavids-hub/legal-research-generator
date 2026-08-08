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
