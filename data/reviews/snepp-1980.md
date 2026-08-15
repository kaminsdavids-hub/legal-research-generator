# Coding review: Snepp v. United States, 444 U.S. 507 (1980)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> We therefore reverse the judgment of the Court of Appeals insofar as it refused
> to impose a constructive trust on Snepp's profits, and we remand ... for
> reinstatement of the full judgment of the District Court.

## `harm_proximity: foreseeable`

The Court affirmed the finding that Snepp's failure to submit his manuscript

> inflicted "irreparable harm" on intelligence activities vital to our national
> security.

Note what that finding did **not** require: the district court reached it
without any showing that the book contained classified material.

## `restriction_type: inapplicable` and `capability_tier: inapplicable`

Same reasoning as Marchetti. A constructive trust for breach of a
prepublication-review agreement is not a speech regulation with a content
character, and a memoir has no capability tier. Both are permanent, not gaps in
the record.

## The axis that does not fit

**`actor_type: hobbyist`** — Snepp was a former CIA agent. CHECK, as with
Marchetti and Karn: three of the fifteen records are misfiled on this axis for
the same missing value.

## Axes the opinion supports

`distribution_modality: paper`, `forum: journal`, `jurisdiction: us_federal`,
`post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
