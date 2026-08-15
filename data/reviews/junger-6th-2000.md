# Coding review: Junger v. Daley, 209 F.3d 481 (6th Cir. 2000)

Reviewed against the full majority opinion (Martin, C.J.), fetched from
`static.case.law/f3d/209/cases/0481-01.json`. Sole opinion; no dissent or
concurrence. The CAP text carries no star pagination, so quotations below are
verbatim but unpinned — a reviewer confirming this coding should re-cite from
the reporter.

**Status after this review: still `model_draft`.** A model read the opinion and
checked each axis against it; that is not what `model_verified` asserts. See
"What is left for a human" at the bottom.

## Corrections made

### `outcome`: `permitted` → `""` (the court decided coverage, not validity)

The draft recorded Junger as a case in which the distribution was permitted. It
was not. The Sixth Circuit reversed a summary judgment and remanded:

> Having concluded that the First Amendment protects computer source code, we
> reverse the district court and remand this case for further consideration of
> Junger's constitutional claims in light of the amended regulations.

It expressly declined to decide whether the export restriction survives:

> We recognize that national security interests can outweigh the interests of
> protected speech and require the regulation of speech. In the present case,
> the record does not resolve whether the exercise of presidential power in
> furtherance of national security interests should overrule the interests in
> allowing the free exchange of encryption source code.

Junger never applied for a license after the classification determination, and
the panel directed the district court to decide on remand whether he could even
bring a facial challenge to the amended regulations. Nothing was permitted.

`outcome` is left empty rather than set to `uncertain`. Empty is the encoding
`match_precedents` already understands — `if case.outcome and ...` — and it
means "this case anchors a real point in the fact space but holds nothing the
readings can be wrong about." `uncertain` would have been worse in two ways: it
would mark every reading that reaches a confident outcome as *contradicted* by
Junger, when the truth is that those readings are unconfirmed rather than
refuted; and `graph_adapter._objection_text` would render the sentence "Junger
v. Daley (209 F.3d 481) was decided uncertain", which is not a thing a case is.

### `restriction_type`: `content_based` → `incidental`

The draft note asked for this to be checked. The opinion answers it. The
district court held the Regulations were content-neutral, and the Sixth Circuit
did not disturb that; its own framing of the analysis to come is O'Brien
intermediate scrutiny:

> Under intermediate scrutiny, the regulation of speech is valid, in part, if
> "it furthers an important or substantial governmental interest."
> O'Brien, 391 U.S. at 377.

The EAR are general-application export controls keyed to an Export Control
Classification Number, not a restriction aimed at the message. `incidental` is
the supported value.

## Axes the opinion supports

| axis | value | evidence |
| --- | --- | --- |
| `actor_type` | `academic` | "Peter Junger is a professor at the Case Western University School of Law." |
| `jurisdiction` | `us_federal` | Sixth Circuit; Export Administration Regulations, 15 C.F.R. Parts 730–74. |
| `post_release_modification` | `none` | Nothing downstream of publication was in issue; the challenge is to posting simpliciter. |

## Axes the opinion does not support

Both are now coded `undetermined`, a value added to the schema in response to
this review. It was previously impossible to say "the record does not answer
this", so a coding had to invent something.

- **`capability_tier: undetermined`** — the opinion states no key length,
  algorithm, or strength for any of Junger's five programs. The only capability
  signal in it cuts both ways: four programs were classified under ECCN 5D002
  "for national security reasons", while the printed chapter of his textbook
  *Computers and the Law* was an allowable unlicensed export. A reviewer wanting
  this axis answered will have to go outside the opinion.
- **`harm_proximity: undetermined`** — this is the question the court refused to
  answer. It quoted Turner for the proposition that the government "must
  demonstrate that the recited harms are real, not merely conjectural, and that
  the regulation will in fact alleviate these harms in a direct and material
  way", and then held the record did not resolve it. The drafted `remote`
  supplied the answer the panel withheld.

Junger answers 6 of 8 axes, a determinacy of 0.75, which is exactly the floor
`match_precedents` requires to let a case anchor a finding. One more silence and
it would be listed as a lead rather than a precedent.

## Axes where the value is the nearest available, not the right one

- **`distribution_modality: training_code`** — it was encryption source code,
  not ML training code. `training_code` is the closest of the four values and
  the one that keeps the case in the "code, not weights" half of the space,
  which is the distinction the project cares about.
- **`forum: public_repository`** — his own web site, not a repository.

## Two schema gaps this case exposed

The first has been fixed; see `UNDETERMINED` in `modules/casework/schema.py`.
The second has not.

**The operative distinction in Junger is one the schema cannot represent.**
Printed encryption software is exempt from the Regulations, 15 C.F.R.
§ 734.3(b)(2); the same code in electronic form requires a license, and for
encryption software "export" includes publication on the Internet. The
Export Administration told Junger the book chapter could go and the electronic
text could not. `distribution_modality` is about *what artifact*; the case turns
on *what medium*, and there is no axis for it.

**`outcome` cannot express "decided coverage, not validity".** Junger holds
source code is within the First Amendment and remands everything else. That is
the shape of most appellate encryption-code decisions — the Bernstein panel
opinion was withdrawn before the en banc court reached the merits, and Karn was
remanded twice. A corpus of this area will be mostly threshold holdings, and the
single `outcome` string flattens them into either a false holding or silence.

## What this coding does to the readings

As coded, only one of the four frozen readings will answer:

| reading | verdict | why |
| --- | --- | --- |
| `deemed_export` | uncertain | reads `capability_tier` |
| `functional_artifact` | uncertain | reads `capability_tier` |
| `expressive_code` | uncertain (intermediate) | applies: incidental burden, O'Brien |
| `published_information` | permitted | applies: published, so outside the EAR |

That is the corrected coding earning its keep. Three of four readings turn on a
fact the Sixth Circuit did not have, which is a more useful thing to know about
this case than any verdict they would have produced from an invented tier. Under
the drafted coding all four answered confidently — `permitted / permitted /
uncertain / permitted` — and three of those answers rested on a `capability_tier`
that appears nowhere in the opinion.

Since `outcome` is empty, none of this is reported as a contradiction. The case
locates a region of the space and says nothing about who is right there.

## What is left for a human

1. Read the opinion (it is five pages) and confirm the two corrections.
2. Decide `capability_tier` and `harm_proximity` on some basis other than this
   opinion, or leave them `undetermined` — which is now a coding the schema can
   express, and which three of the four readings will respect by declining to
   answer.
3. Then set `coding: model_verified` and `coded_by: <your name>` in
   `data/cases.yaml`. Both gates read those fields: `match_precedents` will
   admit the case as a precedent, and `graph_adapter.emit_precedents` will
   write objections built on it into the manuscript.
