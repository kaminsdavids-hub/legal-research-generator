"""Reading 2: code and weights are expression; restrictions face scrutiny."""

from modules.casework.schema import Outcome, Scrutiny, Verdict

NAME = "expressive_code"
READING = (
    "Source code and model weights are expressive for First Amendment purposes "
    "(Junger; the withdrawn Bernstein panel), so a content-based restriction on "
    "their publication faces strict scrutiny."
)


def verdict(facts):
    expressive = facts["distribution_modality"] in ("weights", "training_code", "paper")
    if not expressive:
        return Verdict(Outcome.RESTRICTED, Scrutiny.RATIONAL_BASIS, "service, not expression")
    if facts["restriction_type"] == "content_based":
        return Verdict(Outcome.PERMITTED, Scrutiny.STRICT, "content-based burden on expression")
    return Verdict(Outcome.UNCERTAIN, Scrutiny.INTERMEDIATE, "incidental burden: O'Brien")
