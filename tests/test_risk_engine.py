from app.services import risk_engine


def test_identity_mismatch_risk_normal():
    assert risk_engine.compute_identity_mismatch_risk(0.9) == 0.1
    assert risk_engine.compute_identity_mismatch_risk(0.2) == 0.8


def test_identity_mismatch_risk_none_is_max():
    assert risk_engine.compute_identity_mismatch_risk(None) == 1.0


def test_impersonation_risk_formula():
    # 0.5*0.8 + 0.3*0.5 + 0.2*0.1 = 0.4 + 0.15 + 0.02 = 0.57
    risk = risk_engine.compute_impersonation_risk(
        spoof_score=0.8, identity_mismatch_risk=0.5, context_risk=0.1
    )
    assert risk == 0.57


def test_verdict_thresholds():
    assert risk_engine.get_verdict(0.9) == "HIGH_RISK_LIKELY_IMPERSONATION"
    assert risk_engine.get_verdict(0.5) == "MEDIUM_RISK_MANUAL_REVIEW"
    assert risk_engine.get_verdict(0.1) == "LOW_RISK_LIKELY_GENUINE"


def test_verdict_thresholds_are_consistent_at_boundaries():
    # Boundaries are inclusive on the lower edge of each band, per config.py
    # VERDICT_THRESHOLDS = {"high": 0.70, "medium": 0.40}
    assert risk_engine.get_verdict(0.70) == "HIGH_RISK_LIKELY_IMPERSONATION"
    assert risk_engine.get_verdict(0.6999) == "MEDIUM_RISK_MANUAL_REVIEW"
    assert risk_engine.get_verdict(0.40) == "MEDIUM_RISK_MANUAL_REVIEW"
    assert risk_engine.get_verdict(0.3999) == "LOW_RISK_LIKELY_GENUINE"
    # No gaps or overlaps across the full [0, 1] range
    import random
    random.seed(42)
    for _ in range(200):
        r = round(random.uniform(0.0, 1.0), 4)
        verdict = risk_engine.get_verdict(r)
        assert verdict in {
            "HIGH_RISK_LIKELY_IMPERSONATION",
            "MEDIUM_RISK_MANUAL_REVIEW",
            "LOW_RISK_LIKELY_GENUINE",
        }


def test_impersonation_risk_bounded_even_with_extreme_inputs():
    # Formula inputs are always in [0,1] by construction, but confirm the
    # clamp holds even if a future model returns something out-of-spec.
    assert risk_engine.compute_impersonation_risk(1.0, 1.0, 1.0) == 1.0
    assert risk_engine.compute_impersonation_risk(0.0, 0.0, 0.0) == 0.0
    assert risk_engine.compute_impersonation_risk(2.0, 2.0, 2.0) == 1.0  # clamped
    assert risk_engine.compute_impersonation_risk(-1.0, -1.0, -1.0) == 0.0  # clamped
