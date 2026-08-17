# Coding review: Karn v. U.S. Department of State, 925 F. Supp. 1 (D.D.C. 1996)

Reviewed against the full majority opinion from
`static.case.law`. The CAP text carries no star pagination, so quotations are
verbatim but unpinned; a reviewer confirming this coding should re-cite from the
reporter.

**Status: still `model_draft`.** A model read the opinion and checked each axis
against it. That is not what `model_verified` asserts, which is that a human did.

## Disposition → `outcome: restricted`

> the defendants are entitled to summary judgment as a matter of law.

Karn lost; the export control on the diskette stood.

## `restriction_type: incidental`

The court's own section heading:

> The Court Shall Grant The Defendants' Motion For Summary Judgment On The
> Plaintiff's First Amendment Claim Because The Regulation Is Content-Neutral
> And Meets The O'Brien Test.

## Two blanks

- **`capability_tier: undetermined`** — the opinion states no algorithm or key
  length. The diskette carried the source code from a book that was itself
  freely exportable, which is the case's central irony and tells you nothing
  about strength.
- **`harm_proximity: undetermined`** — zero hits for imminence, immediacy or
  conjecture. The court found the national-security interest substantial under
  O'Brien without ever locating the harm in time.

## The axis that does not fit

**`actor_type: hobbyist`** — Karn was an individual engineer. The axis has
`academic`, `hobbyist`, `firm`, `foreign_state_entity` and none of them is
"individual professional". This is a gap in the schema, not a silence in the
record, so it is *not* coded `inapplicable`: the record says exactly who he was.
CHECK — this is the weakest coding in the record and the fix is probably an axis
value, not a different pick among these four.

## What is left for a human

Read the opinion, confirm the axes above — in particular any marked CHECK — and
then set `coding: model_verified` and `coded_by: <your name>` in
`data/cases.yaml`. Both gates read those fields.
