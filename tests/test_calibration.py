"""Regression tests for presentation calibration and weighted risk fusion."""

from app.services import risk_engine
from app.services.ai_models.aasist_scoring import spoof_label_from_score


def test_spoof_confidence_bands_are_complete_and_inclusive():
    assert spoof_label_from_score(0.00) == "Genuine (Very High Confidence)"
    assert spoof_label_from_score(0.25) == "Genuine (Very High Confidence)"
    assert spoof_label_from_score(0.26) == "Probably Genuine"
    assert spoof_label_from_score(0.45) == "Probably Genuine"
    assert spoof_label_from_score(0.46) == "Suspicious"
    assert spoof_label_from_score(0.65) == "Suspicious"
    assert spoof_label_from_score(0.66) == "Likely AI Generated"
    assert spoof_label_from_score(0.85) == "Likely AI Generated"
    assert spoof_label_from_score(0.86) == "Highly Likely AI Generated"


def test_weighted_fusion_respects_identity_evidence():
    # Low similarity and strong spoof evidence should receive a HIGH verdict.
    risky = risk_engine.compute_weighted_risk(1.0, 0.10, "high", 500_000)
    assert risk_engine.get_verdict(risky) in {"HIGH", "CRITICAL"}

    # A strong voice match is a meaningful counter-signal when no other
    # contextual evidence is present; spoof evidence is not absolute truth.
    matched = risk_engine.compute_weighted_risk(1.0, 0.90, "low", 0)
    assert risk_engine.get_verdict(matched) != "HIGH"
    assert risk_engine.get_verdict(matched) != "CRITICAL"


def test_unknown_caller_is_neutral_identity_evidence():
    assert risk_engine.compute_identity_mismatch_risk(None) == 0.50

