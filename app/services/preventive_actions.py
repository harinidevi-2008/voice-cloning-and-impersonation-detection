"""Single source of truth for actions attached to a completed analysis."""

from typing import Optional


_ACTIONS_BY_RISK = {
    "LOW": [
        "Proceed with the transaction.",
        "Log the interaction.",
        "No further verification required.",
    ],
    "MEDIUM": [
        "Verify one personal identity detail.",
        "Confirm payment using the registered number.",
        "Do not share OTP or PIN.",
        "Record the interaction.",
    ],
    "HIGH": [
        "Pause the transaction immediately.",
        "Verify identity via another channel.",
        "Contact the organization directly.",
        "Escalate to fraud monitoring.",
    ],
    "CRITICAL": [
        "Block the transaction.",
        "Disconnect the call.",
        "Freeze payment authorization.",
        "Preserve recording as evidence.",
        "Notify cybersecurity response.",
    ],
}


def generate_preventive_actions(
    verdict: str,
    spoof_score: float,
    amount: Optional[float],
    urgency: Optional[str],
) -> list[str]:
    """Generate the frozen dashboard action wording once, on the backend."""
    actions = list(_ACTIONS_BY_RISK.get((verdict or "").upper(), _ACTIONS_BY_RISK["MEDIUM"]))
    additions = []
    if (amount or 0) > 100_000:
        additions.append("Require dual approval before payment.")
    if (urgency or "").upper() == "HIGH":
        additions.append("Ignore pressure tactics requesting immediate action.")
    if (spoof_score or 0) > 0.70:
        additions.append("Treat the voice as potentially AI-generated.")
    for action in additions:
        if action not in actions and len(actions) >= 5:
            actions.pop()
        if action not in actions:
            actions.append(action)
    return actions[:5]
