"""Reading 3: weights are functional, and function is regulable."""

from modules.casework.schema import Outcome, Scrutiny, Verdict

NAME = "functional_artifact"
READING = (
    "Weights are a functional artifact rather than communication: what is "
    "transferred is a capability, so export control reaches it and the First "
    "Amendment does not."
)


def verdict(facts):
    tier = int(facts["capability_tier"])
    if facts["distribution_modality"] == "paper":
        return Verdict(Outcome.PERMITTED, Scrutiny.STRICT, "a paper communicates")
    if tier >= 4 or facts["post_release_modification"] == "alignment_removal":
        return Verdict(Outcome.RESTRICTED, Scrutiny.RATIONAL_BASIS, "capability transfer")
    if facts["harm_proximity"] == "imminent":
        return Verdict(Outcome.RESTRICTED, Scrutiny.NONE, "imminent harm")
    return Verdict(Outcome.UNCERTAIN, Scrutiny.INTERMEDIATE, "mixed function and expression")
