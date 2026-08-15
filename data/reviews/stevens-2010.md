# Coding review: United States v. Stevens, 559 U.S. 460 (2010)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: permitted`

> We hold only that § 48 is not so limited but is instead substantially
> overbroad, and therefore invalid under the First Amendment.

Note "We hold only" — the Court was marking the narrowness itself.

## `harm_proximity: undetermined`

The statute fell as substantially overbroad, which is a holding about the
statute's *reach*, not about what depictions of animal cruelty cause. The Court
reached no such question.

## `restriction_type: content_based`

The Court of Appeals had

> held that § 48 could not survive strict scrutiny as a content-based regulation
> of protected speech

and the Supreme Court affirmed, though on overbreadth. CHECK: the coding takes
the characterisation from the court below.

## `capability_tier: inapplicable`

A depiction of conduct conveys what happened, not a means of doing it.

## Why this record matters to the project

Stevens refused to create a new category of unprotected speech by balancing
costs against benefits. That is directly relevant to any argument that weights
are categorically outside the First Amendment, and it is the reason to keep this
record even though several of its axes fit badly.

## The axis that does not fit

**`actor_type: hobbyist`** — Stevens compiled and sold dogfighting videos. CHECK.

## Axes the opinion supports

`distribution_modality: paper` — CHECK, videotapes; `forum: public_repository` —
CHECK, mail-order sale; `jurisdiction: us_federal`,
`post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
