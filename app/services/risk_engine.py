"""
risk_engine.py
==============
Pure functions implementing the agreed formula:

    IdentityMismatchRisk = 1 - speaker_similarity
    ImpersonationRisk    = 0.5*SpoofRisk + 0.3*IdentityMismatchRisk + 0.2*ContextRisk

Kept independent of FastAPI so it can be unit-tested with plain numbers and
so the "why did this get flagged" logic is never buried inside a route.
"""

from typing import Optional
from app.config import RISK_WEIGHTS, VERDICT_THRESHOLDS, VERDICT_LABELS, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, CRITICAL_RISK_THRESHOLD


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_identity_mismatch_risk(speaker_similarity: Optional[float]) -> float:
    """
    IdentityMismatchRisk = 1 - speaker_similarity

    DESIGN DECISION (reviewed during hardening, kept as-is): when
    speaker_similarity is None — no claimed_user_id was provided, so
    identity could not be checked at all — identity risk is treated as
    maximal (1.0), not zero and not excluded from the formula.

    This is intentional, not an oversight: this system exists to flag
    *impersonation* risk, and an unverifiable identity is itself a risk
    signal in that context, not neutral information. A caller who won't or
    can't be matched against an enrolled voiceprint should not score better
    than one who was checked and failed to match — "unknown" must never be
    cheaper than "known and risky" for a fraud-detection system, or it
    creates an incentive to simply not claim an identity. Excluding the
    identity term entirely (rather than maxing it) would have the same
    problem: it would let spoof_score and context_risk alone decide the
    verdict, silently dropping identity mismatch as a factor exactly when
    identity is the very thing in question.

    If a future version of this project wants a middle-ground default
    (e.g. a fixed moderate risk for "unknown", rather than maximal), that
    should be a deliberate product decision with its own justification, not
    a silent default. See tests/test_risk_engine.py::
    test_identity_mismatch_risk_none_is_max for the behavior this locks in.
    """
    if speaker_similarity is None:
        return 1.0
    similarity = max(0.0, min(1.0, speaker_similarity))
    return round(1 - similarity, 4)


def compute_prosody_risk(prosody_score: Optional[float]) -> float:
    if prosody_score is None:
        return 0.0
    return round(clamp01(prosody_score), 4)


def compute_impersonation_risk(
    spoof_score: float,
    identity_mismatch_risk: float,
    context_risk: float,
    prosody_risk: float = 0.0,
    prosody_confidence: float = 1.0,
) -> float:
    w = RISK_WEIGHTS
    if clamp01(prosody_risk) == 0.0 or clamp01(prosody_confidence) == 0.0:
        # Preserve the established three-signal formula when prosody is
        # unavailable. This makes absence of optional acoustic evidence
        # neutral rather than silently lowering every legacy result.
        risk = 0.5 * clamp01(spoof_score) + 0.3 * clamp01(identity_mismatch_risk) + 0.2 * clamp01(context_risk)
    else:
        risk = (
            w["spoof"] * clamp01(spoof_score)
            + w["identity"] * clamp01(identity_mismatch_risk)
            + w["context"] * clamp01(context_risk)
            # Prosody is supporting evidence only. A low-confidence estimate is
            # attenuated rather than converted into a suspicious default.
            + w["prosody"] * clamp01(prosody_risk) * clamp01(prosody_confidence)
        )
    return round(clamp01(risk), 4)


def get_verdict(impersonation_risk: float) -> str:
    if impersonation_risk >= VERDICT_THRESHOLDS["high"]:
        return VERDICT_LABELS["high"]
    if impersonation_risk >= VERDICT_THRESHOLDS["medium"]:
        return VERDICT_LABELS["medium"]
    return VERDICT_LABELS["low"]


def get_recommended_action(impersonation_risk: float) -> str:
    if impersonation_risk >= CRITICAL_RISK_THRESHOLD:
        return "Block or hold the transaction and trigger an out-of-band security review."
    if impersonation_risk >= HIGH_RISK_THRESHOLD:
        return "Pause the transaction and verify the caller through an independent trusted channel."
    if impersonation_risk >= MEDIUM_RISK_THRESHOLD:
        return "Proceed with caution and request secondary verification before sensitive actions."
    return "Allow the interaction to continue while monitoring for unusual behavior."


def explain_risk_factors(
    spoof_score: float,
    identity_mismatch_risk: float,
    context_risk: float,
    prosody_risk: Optional[float] = None,
    urgency: Optional[str] = None,
    amount: Optional[float] = None,
) -> list:
    factors = []
    if spoof_score >= 0.6:
        factors.append("Synthetic speech evidence is elevated")
    if identity_mismatch_risk >= 0.5:
        factors.append("Speaker identity does not closely match the enrolled profile")
    if context_risk >= 0.5:
        factors.append("The transaction context is unusually risky")
    if prosody_risk is not None and prosody_risk >= 0.5:
        factors.append("Prosody patterns show unusual vocal behavior")
    if urgency and urgency.lower() in {"high", "medium"}:
        factors.append(f"Urgency signal detected ({urgency})")
    if amount is not None and amount >= 50000:
        factors.append("The requested amount is high-value for a fraud-sensitive interaction")
    return factors or ["No strong risk factors were detected from the available signals"]
