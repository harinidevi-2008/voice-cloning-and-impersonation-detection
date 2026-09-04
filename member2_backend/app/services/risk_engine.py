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
from app.config import RISK_WEIGHTS, VERDICT_THRESHOLDS, VERDICT_LABELS


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


def compute_impersonation_risk(
    spoof_score: float,
    identity_mismatch_risk: float,
    context_risk: float,
) -> float:
    w = RISK_WEIGHTS
    risk = (
        w["spoof"] * spoof_score
        + w["identity"] * identity_mismatch_risk
        + w["context"] * context_risk
    )
    return round(max(0.0, min(1.0, risk)), 4)


def get_verdict(impersonation_risk: float) -> str:
    if impersonation_risk >= VERDICT_THRESHOLDS["high"]:
        return VERDICT_LABELS["high"]
    if impersonation_risk >= VERDICT_THRESHOLDS["medium"]:
        return VERDICT_LABELS["medium"]
    return VERDICT_LABELS["low"]
