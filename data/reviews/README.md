# Coding reviews

One file per coded case. Each was checked axis by axis against the full majority
opinion, fetched this session from the Caselaw Access Project bulk data at
`static.case.law`. Nothing in them is recalled; every load-bearing axis carries
the passage it rests on.

**None of this is verification.** `Coding.MODEL_VERIFIED` asserts that *a human*
confirmed the coding axis by axis, and reading an opinion with a model is not
that. What these files do is make the human pass short: the quotations are
already pulled, the corrections are already argued, and the open questions are
marked `CHECK`. Confirm a case and set `coding: model_verified` and
`coded_by: <your name>` on its record in `data/cases.yaml`.

The two regulatory instruments — `ear-diffusion-2025` and `deemed-export-rule` —
have no review here. CAP holds cases, not the Federal Register or the CFR, so
their citations were never machine-confirmed either. They need a different kind
of check.

## Where to spend the first hour

Ranked by how much a wrong answer would cost the project, not by how uncertain
the coding is:

1. **`corley-2d-2001`** — `restriction_type: incidental`. The Second Circuit
   treated the DMCA as content-neutral because it targets code's *functional*
   aspect. That functional/expressive split is the thing this project argues
   about; the case is the strongest authority against the paper's position, and
   this one value decides whether the corpus records it accurately.
2. **`sorrell-2011`** — `distribution_modality: weights`. A dataset coded as the
   nearest analogue to model weights. Sorrell is in the corpus *because* the
   project wants a data-as-speech authority, which makes this the coding most at
   risk of having been picked for the conclusion it produces.
3. **`hlp-2010`** — `actor_type: foreign_state_entity` codes the *recipient*,
   where every other record codes the distributor, and
   `distribution_modality: training_code` describes training and expert advice.
   Two loose analogies in the record that most directly supports restriction.
4. **`bartnicki-2001`** — the Court called the statute content-neutral and then
   applied a demanding standard anyway. `incidental` records the first and loses
   the second.

## Findings that are about the schema, not the cases

Three defects surfaced in every record at once, and no per-case fix will help:

**`actor_type` has no value for an individual professional.** Karn (an
engineer), Marchetti and Snepp (former intelligence officers) and Stevens (a
seller) are all coded `hobbyist`, which is false about all four. The record is
perfectly clear in each; the axis simply lacks the value. That is why none of
them is coded `inapplicable` — the question arises and has an answer the schema
cannot express.

**`distribution_modality` has four values for fifteen artifacts.** A radio
broadcast, videotapes, a magazine article, a murder manual, retail video games,
expert training and a prescriber database are being sorted into `paper`,
`training_code`, `weights` and `api`. Several of those are translations rather
than observations, and each is flagged in its own file.

**`outcome` cannot say "decided coverage, not validity".** Junger holds that
source code is protected and remands everything else; Bernstein decided one
procedural question; the Pentagon Papers per curiam allocated a burden. This is
the shape of most appellate decisions in the area, and the single `outcome`
string flattens them into a holding they did not make. Junger is coded with an
empty outcome as the least-wrong option available.

## What changed during the review

Four codings stated the opposite of what their opinion held:

| case | axis | was | is | the sentence that decided it |
| --- | --- | --- | --- | --- |
| `junger-6th-2000` | `outcome` | `permitted` | `""` | reversed and remanded; nothing was permitted |
| `junger-6th-2000` | `restriction_type` | `content_based` | `incidental` | the panel framed the analysis under O'Brien |
| `corley-2d-2001` | `harm_proximity` | `foreseeable` | `imminent` | "very substantial risk of imminent harm" |
| `rice-4th-1997` | `restriction_type` | `content_based` | `incidental` | "regulated incidentally to ... generally applicable statutes" |
| `bartnicki-2001` | `restriction_type` | `content_based` | `incidental` | "is in fact a content-neutral law of general applicability" |

And every judicial record lost its `capability_tier`. No opinion grades an
artifact on a five-point capability scale, because the scale postdates them all.
Seven are `undetermined` — the artifact is a means of doing something and the
court did not quantify it — and eight are `inapplicable`, because a policy
history, a scandal sheet, two memoirs, a phone call, a prescriber database, a
video game and a depiction of conduct have no capability to grade.
