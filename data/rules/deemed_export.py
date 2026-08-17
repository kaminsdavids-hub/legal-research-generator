"""Reading 4: the deemed-export rule turns on the recipient, not the item."""

from modules.casework.schema import Outcome, Scrutiny, Verdict

NAME = "deemed_export"
READING = (
    "A release of controlled technology to a foreign person is an export "
    "wherever it happens (15 C.F.R. 734.13), so the recipient decides the "
    "question and publication to the world includes foreign persons."
)


def verdict(facts):
    if facts["actor_type"] == "foreign_state_entity":
        return Verdict(Outcome.RESTRICTED, Scrutiny.NONE, "release to a foreign person")
    if facts["forum"] == "public_repository" and int(facts["capability_tier"]) >= 3:
        return Verdict(Outcome.RESTRICTED, Scrutiny.RATIONAL_BASIS, "world includes foreign persons")
    return Verdict(Outcome.PERMITTED, Scrutiny.NONE, "no controlled release")
