# Coding review: United States v. Progressive, Inc., 467 F. Supp. 990 (W.D. Wis. 1979)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> Plaintiff has proven all necessary prerequisites for issuance of a preliminary
> injunction restraining defendants from publishing or disclosing any Restricted
> Data contained in the Morland article until a final determination in this
> action has been made by the Court.

The court was conscious of what it was doing:

> If a preliminary injunction is issued, it will constitute the first instance of
> prior restraint against a publication in this fashion in the history of this
> country, to this Court's knowledge.

The case became moot before appellate review, so this is a district court order
that was never tested. Weight it accordingly.

## `harm_proximity: imminent`

> if when drawn together, synthesized and collated, such information acquires the
> character of presenting immediate, direct and irreparable harm to the interests
> of the United States.

## `capability_tier: undetermined` — and this is the sharpest case for the blank

This is the nearest analogue in the corpus to a capability-transfer claim, and
the reason the axis exists at all. It is also the clearest demonstration that a
judicial record cannot supply it: the court found grave harm and described the
article as synthesised technical data, and still nothing in the opinion grades
the artifact on any scale. If *Progressive* cannot answer this axis, no case can.

## Axes the opinion supports

`distribution_modality: paper` (a magazine article), `actor_type: firm` (The
Progressive, Inc.), `forum: journal`, `jurisdiction: us_federal`,
`restriction_type: content_based` (the Atomic Energy Act restricts by subject
matter — Restricted Data), `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
