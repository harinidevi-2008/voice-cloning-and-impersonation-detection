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
    urgency: Optional[str] = None,
    transaction_amount: Optional[float] = None,
    *,
    context_risk: Optional[float] = None,
    prosody_risk: Optional[float] = None,
    prosody_confidence: float = 0.0,
) -> float:
    """Fuse model evidence and explainable context into final risk.

    Unavailable or low-confidence prosody is excluded and the remaining
    weights are renormalized, rather than treating missing audio evidence as
    suspicious.
    """
    if context_risk is None:
        # Backward-compatible public helper path. Production supplies its
        # already-computed context score directly from context_engine.
        from app.services.context_engine import compute_context_risk
        context_risk = compute_context_risk(
            caller_known=bool(speaker_similarity is not None and speaker_similarity >= SPEAKER_VERIFIED),
            transaction_value=transaction_amount,
            urgency=urgency or "low",
        )["context_risk"]
    signals = {
        "spoof": clamp_score(spoof_score),
        "identity": compute_identity_mismatch_risk(speaker_similarity),
        "context": clamp_score(context_risk),
    }
    weights = {name: RISK_WEIGHTS[name] for name in signals}
    confidence = clamp_score(prosody_confidence)
    if prosody_risk is not None and confidence > 0:
        signals["prosody"] = clamp_score(prosody_risk)
        weights["prosody"] = RISK_WEIGHTS["prosody"] * confidence
    total_weight = sum(weights.values())
    risk = sum(weights[name] * signals[name] for name in signals) / total_weight
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


def get_recommended_action(impersonation_risk: float) -> str:
    """Return the deterministic, backend-owned action for a final score.

    This deliberately returns one concise instruction.  The frozen dashboard
    may continue to present its existing richer action card without depending
    on this additive API field.
    """
    actions = {
        "LOW": "Continue the transaction and monitor the interaction.",
        "MEDIUM": "Use secondary verification before proceeding.",
        "HIGH": "Pause the transaction and verify through an independent trusted channel.",
        "CRITICAL": "Hold or block the transaction and escalate for out-of-band security review.",
    }
    return actions[get_verdict(impersonation_risk)]


# Compatibility for third-party callers of the former helper.  The route uses
# compute_weighted_risk, keeping the production decision in one place.
def compute_impersonation_risk(
    spoof_score: float, identity_mismatch_risk: float, context_risk: float
) -> float:
    # This legacy helper receives an already-derived mismatch score rather
    # than a similarity and has no prosody argument.  Excluding prosody must
    # not lower every result merely because the signal is unavailable, so use
    # the same documented renormalization as compute_weighted_risk().
    weights = {
        "spoof": RISK_WEIGHTS["spoof"],
        "identity": RISK_WEIGHTS["identity"],
        "context": RISK_WEIGHTS["context"],
    }
    total = sum(weights.values())
    risk = (
        weights["spoof"] * clamp_score(spoof_score)
        + weights["identity"] * clamp_score(identity_mismatch_risk)
        + weights["context"] * clamp_score(context_risk)
    ) / total
    return round(clamp_score(risk), 4)
