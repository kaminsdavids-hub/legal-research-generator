# Coding review: Brown v. Entertainment Merchants Ass'n, 564 U.S. 786 (2011)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: permitted`

> We affirm the judgment below.

California's violent-video-game statute fell.

## `harm_proximity: remote`

> At the outset, it acknowledges that it cannot show a direct causal link between
> violent video games and harm to minors.

and on the studies:

> They do not prove that violent video games cause minors to act aggressively
> (which would at least be a beginning).

## `restriction_type: content_based`

> Because the Act imposes a restriction on the content of protected speech, it is
> invalid unless California can demonstrate that it passes strict scrutiny

## `capability_tier: inapplicable`

An entertainment work is not a means of doing anything this axis measures.

## Axes the opinion supports

`distribution_modality: training_code` — CHECK, video games are software sold to
a general audience; `training_code` is the closest of four values and a poor
description. `actor_type: firm` (an industry association),
`forum: public_repository` — CHECK, retail sale is not a repository —
`jurisdiction: us_state`, `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
