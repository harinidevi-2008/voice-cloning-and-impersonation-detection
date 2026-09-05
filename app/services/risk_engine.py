"""Centralized, explainable risk calibration for call analysis."""

from typing import Optional

from app.config import (
    HIGH_VALUE_THRESHOLD,
    MED_VALUE_THRESHOLD,
    RISK_WEIGHTS,
    SPEAKER_LIKELY,
    SPEAKER_REVIEW,
    SPEAKER_VERIFIED,
    URGENCY_RISK_MAP,
    VERDICT_LABELS,
    VERDICT_THRESHOLDS,
)
from app.services.ai_models.aasist_scoring import spoof_label_from_score


def clamp_score(value: Optional[float]) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def classify_speaker_similarity(similarity: Optional[float]) -> str:
    """Map ECAPA evidence into an actionable, configurable status."""
    if similarity is None:
        return "Needs Verification"
    if similarity >= SPEAKER_VERIFIED:
        return "Verified Identity"
    if similarity >= SPEAKER_LIKELY:
        return "Likely Match"
    if similarity >= SPEAKER_REVIEW:
        return "Needs Verification"
    return "No Match"


def classify_spoof_score(spoof_score: float) -> str:
    """Calibrate the AASIST score into a risk category, never certainty."""
    return spoof_label_from_score(clamp_score(spoof_score))


def compute_identity_mismatch_risk(speaker_similarity: Optional[float]) -> float:
    # No claimed identity means there is no match evidence, not a mismatch.
    # Keep this signal neutral so unknown callers are not automatically treated
    # as spoofed or fraudulent.
    return 0.50 if speaker_similarity is None else round(1 - clamp_score(speaker_similarity), 4)


def compute_amount_risk(amount: Optional[float]) -> float:
    value = float(amount or 0.0)
    if value >= HIGH_VALUE_THRESHOLD:
        return 1.0
    if value >= MED_VALUE_THRESHOLD:
        return 0.6
    return 0.15 if value > 0 else 0.0


def compute_weighted_risk(
    spoof_score: float,
    speaker_similarity: Optional[float],
    urgency: str,
    transaction_amount: Optional[float],
) -> float:
    """The sole final-risk calculation used by the analysis pipeline."""
    risk = (
        RISK_WEIGHTS["spoof"] * clamp_score(spoof_score)
        + RISK_WEIGHTS["identity"] * compute_identity_mismatch_risk(speaker_similarity)
        + RISK_WEIGHTS["urgency"] * URGENCY_RISK_MAP.get((urgency or "low").lower(), 0.1)
        + RISK_WEIGHTS["amount"] * compute_amount_risk(transaction_amount)
    )
    return round(clamp_score(risk), 4)


def get_verdict(impersonation_risk: float) -> str:
    score = clamp_score(impersonation_risk)
    if score >= VERDICT_THRESHOLDS["critical"]:
        return VERDICT_LABELS["critical"]
    if score >= VERDICT_THRESHOLDS["high"]:
        return VERDICT_LABELS["high"]
    if score >= VERDICT_THRESHOLDS["medium"]:
        return VERDICT_LABELS["medium"]
    return VERDICT_LABELS["low"]


# Compatibility for third-party callers of the former helper.  The route uses
# compute_weighted_risk, keeping the production decision in one place.
def compute_impersonation_risk(
    spoof_score: float, identity_mismatch_risk: float, context_risk: float
) -> float:
    return round(clamp_score(
        RISK_WEIGHTS["spoof"] * clamp_score(spoof_score)
        + RISK_WEIGHTS["identity"] * clamp_score(identity_mismatch_risk)
        + RISK_WEIGHTS["urgency"] * clamp_score(context_risk)
    ), 4)
