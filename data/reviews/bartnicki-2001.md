# Coding review: Bartnicki v. Vopper, 532 U.S. 514 (2001)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: permitted`

> The judgment is affirmed.

Publication of the illegally intercepted call was protected.

## `restriction_type: incidental` — CORRECTED, was `content_based`

> We agree with petitioners that § 2511(1)(c), as well as its Pennsylvania
> analog, is in fact a content-neutral law of general applicability

This is the Court's own holding on the point, stated in terms. Note the
complication a reviewer should weigh: having called it content-neutral, the
Court then observed that

> the naked prohibition against disclosures is fairly characterized as a
> regulation of pure speech

and applied a demanding standard anyway. `incidental` records the
characterisation; it does not capture the scrutiny actually applied. CHECK.

## `harm_proximity: remote`

> The justification for any such novel burden on expression must be "far stronger
> than mere speculation about serious harms."

## `capability_tier: inapplicable`

The artifact is a recording of a conversation. There is no capability, which is
a different fact from the Court having failed to mention one.

## Axes the opinion supports

`distribution_modality: paper` — CHECK, it was a radio broadcast;
`actor_type: firm` (a radio commentator and stations), `forum: journal`,
`jurisdiction: us_federal`, `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
