# Coding review: Rice v. Paladin Enterprises, Inc., 128 F.3d 233 (4th Cir. 1997)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> we hold that the First Amendment does not pose a bar to the plaintiffs' civil
> aiding and abetting cause of action against Paladin Press.

## `restriction_type: incidental` — CORRECTED, was `content_based`

> [speech] that is tantamount to legitimately proscribable nonexpressive conduct
> may itself be legitimately proscribed, punished, or regulated incidentally to
> the constitutional enforcement of generally applicable statutes.

Liability rests on a generally applicable tort rule, not a rule aimed at the
message.

## `harm_proximity: undetermined` — and pointedly so

The court held that the imminence question is not the test here:

> the question of whether criminal conduct is "imminent" is relevant for
> constitutional purposes only where, as in Brandenburg itself, the government
> attempts to restrict advocacy, as such.

So this is a silence with a holding behind it. The drafted `imminent` supplied
an answer the court declined to reach — and reached *around*.

## What no axis captures

Paladin **stipulated** that it intended the book to be used by criminals. That
stipulation is doing nearly all the work in the opinion, and the schema has no
axis for the publisher's intent. CHECK: this may mean the case is a poor fit for
the corpus rather than a poor coding.

## Axes the opinion supports

`distribution_modality: paper` (the *Hit Man* manual), `actor_type: firm`
(Paladin Press), `forum: journal` — CHECK, a book publisher is not a journal and
the axis has no better value — `jurisdiction: us_federal`,
`capability_tier: undetermined` (instructional means, ungraded),
`post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
