# Coding review: Sorrell v. IMS Health Inc., 564 U.S. 552 (2011)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: permitted`

> The judgment of the Court of Appeals is affirmed.

Vermont's restriction on prescriber-identifying data fell.

## `restriction_type: content_based`

> Any doubt that § 4631(d) imposes an aimed, content-based burden on detailers is
> dispelled by the record and by formal legislative findings.

## `harm_proximity: undetermined`

The Court struck the law and made no finding about what disclosure of prescriber
data would cause.

## `capability_tier: inapplicable`

Prescriber-identifying data reports what physicians did. The axis grades what an
artifact *can do*.

## The coding a human most needs to rule on

**`distribution_modality: weights`** — this is an argument, not an observation.
It rests on the theory that a dataset is the nearest analogue to model weights
available in this schema. Sorrell is in the corpus precisely because the project
wants to say something about data-as-speech, which makes this the coding most at
risk of being chosen for the conclusion it produces. CHECK, hard.

## Axes the opinion supports

`actor_type: firm` (IMS Health and the detailers), `forum: private_transfer`
(sale of the data to pharmaceutical manufacturers), `jurisdiction: us_state`
(a Vermont statute), `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
