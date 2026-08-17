"""Reading 1: the EAR's published-information exclusion controls."""

from modules.casework.schema import Outcome, Scrutiny, Verdict

NAME = "published_information"
READING = (
    "Publication without restriction on further dissemination removes the item "
    "from the EAR entirely, whatever the item is (15 C.F.R. 734.7)."
)


def verdict(facts):
    public = facts["forum"] in ("public_repository", "journal", "conference")
    if not public:
        return Verdict(Outcome.RESTRICTED, Scrutiny.NONE, "not published: private transfer")
    if facts["distribution_modality"] == "api":
        return Verdict(Outcome.RESTRICTED, Scrutiny.NONE, "API access is not publication")
    return Verdict(Outcome.PERMITTED, Scrutiny.NONE, "published, so outside the EAR")
