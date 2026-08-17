# Coding review: Bernstein v. United States Department of Justice, 176 F.3d 1132 (9th Cir. 1999)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Read this first

**This opinion was withdrawn** on rehearing en banc, 192 F.3d 1308. It is coded
for exploration only. No finding may ever cite it, and promoting this record
without resolving that would put an un-issued opinion in a manuscript.

## Disposition → `outcome: permitted`

> we hold that it constitutes an impermissible prior restraint on speech.

Bernstein won. Subject to the withdrawal above.

## Three axes the panel did not reach

The panel was explicit about how narrow it was going:

> The parties and amici urge a number of theories on us. We limit our attention
> here, for the most part, to only one: whether the EAR [is] a prior restraint.

- **`restriction_type: undetermined`** — it never decided content-based versus
  incidental, because it did not need to.
- **`harm_proximity: undetermined`** — it recited the standard, that the
  government must show publication would "surely result in direct, immediate,
  and irreparable damage to our Nation or its people", without applying it.
- **`capability_tier: undetermined`** — no capability fact appears.

Three blanks puts this below the determinacy floor, so it cannot anchor a
finding even once verified. That is the right result for an opinion that decided
one procedural question.

## Axes the opinion supports

| axis | value | basis |
| --- | --- | --- |
| `distribution_modality` | `training_code` | encryption source code (Snuffle) |
| `actor_type` | `academic` | Bernstein was a graduate student, later a professor |
| `jurisdiction` | `us_federal` | 9th Cir.; EAR |
| `forum` | `journal` | CHECK: he sought to publish and to post; `journal` may understate the internet posting |
| `post_release_modification` | `none` | not in issue |

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
