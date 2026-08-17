# Coding review: New York Times Co. v. United States, 403 U.S. 713 (1971)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: permitted`

> The judgment of the Court of Appeals for the District of Columbia Circuit is
> therefore affirmed. The order of the Court of Appeals for the Second Circuit is
> reversed and the case is remanded ... The stays entered June 25, 1971, by the
> Court are vacated.

## Three blanks, and why this record is below the floor

The per curiam is three paragraphs. It says one thing:

> "Any system of prior restraints of expression comes to this Court bearing a
> heavy presumption against its constitutional validity." ... The Government
> "thus carries a heavy burden of showing justification for the imposition of
> such a restraint." [The courts below] held that the Government had not met that
> burden. We agree.

That allocates a burden and decides nothing else. `harm_proximity` and
`restriction_type` are `undetermined` accordingly — and the drafted
`foreseeable` on the first inverted the result, since the holding is that the
Government failed to show the harm.

**The six concurrences are not available to a coder.** They discuss harm,
content and national security at length and command no majority. A coding that
drew on them would be laundering a concurrence into a holding, which is exactly
the error this corpus exists to avoid making mechanically.

`capability_tier: inapplicable` — the study was a history of decision-making. A
narrative of what officials did has no capability to grade.

## Axes the opinion supports

`distribution_modality: paper`, `actor_type: firm` (the newspapers),
`forum: journal`, `jurisdiction: us_federal`, `post_release_modification: none`.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
