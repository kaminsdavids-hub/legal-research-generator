# Coding review: United States v. Marchetti, 466 F.2d 1309 (4th Cir. 1972)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> We affirm the substance of the decision below, limiting the order, however to
> the language of the secrecy agreement Marchetti signed when he joined the
> Agency.

## `restriction_type: inapplicable`

Not a silence. The court analysed a **secrecy agreement and a fiduciary
relation**, not a speech regulation, so neither `content_based` nor `incidental`
describes what was upheld:

> the Government's need for secrecy in this area lends justification to a system
> of prior restraint against disclosure by employees and former employees of
> classified information obtained during the course of employment.

No further reading fills this in — the category does not apply to enforcing a
contract. This is why the sentinel exists.

## `harm_proximity: foreseeable`

> the risk of harm from disclosure is so great

## `capability_tier: inapplicable`

An account of Agency operations conveys information, not a means.

## The axis that does not fit

**`actor_type: hobbyist`** — Marchetti was a former CIA officer. As with Karn,
the record is perfectly clear and the axis has no value for it, so this is a
schema gap rather than a blank. CHECK.

## Axes the opinion supports

`distribution_modality: paper`, `forum: journal`, `jurisdiction: us_federal`,
`post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
