# Coding review: Universal City Studios, Inc. v. Corley, 273 F.3d 429 (2d Cir. 2001)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> We affirm.

The injunction against posting DeCSS and against linking to it stood.

## `harm_proximity: imminent` — CORRECTED, was `foreseeable`

> Here, dissemination itself carries very substantial risk of imminent harm
> because the mechanism is so unusual by which dissemination of means of
> circumventing access controls to copyrighted works threatens to produce
> virtually unstoppable infringement.

The opinion uses the word. The drafted `foreseeable` understated its own source.

## `restriction_type: incidental`

> because the DMCA is targeting the "functional" aspect of that speech, it is
> "content neutral," and the intermediate scrutiny of United States v. O'Brien
> applies

**This is the single most load-bearing coding in the corpus.** The
functional/expressive split is the thing the project is arguing about, and this
case is the strongest authority for treating the functional component as
separable. If a human overturns one coding in this file, it should be this one
they think hardest about.

## `capability_tier: undetermined`

DeCSS is a means of doing something, so the axis applies — the Second Circuit
simply never graded it. Contrast the information cases, where it is
`inapplicable`.

## Axes the opinion supports

`distribution_modality: training_code` (source and object code for DeCSS),
`actor_type: firm` (2600 Enterprises, publisher of *2600: The Hacker
Quarterly*), `forum: public_repository` (posted on a website),
`jurisdiction: us_federal`, `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
