from app.services import risk_engine


def test_identity_mismatch_risk_normal():
    assert risk_engine.compute_identity_mismatch_risk(0.9) == 0.1
    assert risk_engine.compute_identity_mismatch_risk(0.2) == 0.8


def test_identity_mismatch_risk_none_is_neutral():
    # An unknown caller provides no identity-comparison evidence; it is not
    # proof of a mismatched or AI-generated voice.
    assert risk_engine.compute_identity_mismatch_risk(None) == 0.5


def test_impersonation_risk_formula():
    # Legacy helper has no prosody signal, so its configured spoof/identity/
    # context weights are renormalized: (.35*.8 + .35*.5 + .2*.1) / .9.
    risk = risk_engine.compute_impersonation_risk(
        spoof_score=0.8, identity_mismatch_risk=0.5, context_risk=0.1
    )
    assert risk == 0.5278


def test_verdict_thresholds():
    assert risk_engine.get_verdict(0.9) == "CRITICAL"
    assert risk_engine.get_verdict(0.7) == "HIGH"
    assert risk_engine.get_verdict(0.5) == "MEDIUM"
    assert risk_engine.get_verdict(0.1) == "LOW"


def test_verdict_thresholds_are_consistent_at_boundaries():
    # Boundaries are inclusive on the lower edge of each configured band.
    assert risk_engine.get_verdict(0.80) == "CRITICAL"
    assert risk_engine.get_verdict(0.7999) == "HIGH"
    assert risk_engine.get_verdict(0.60) == "HIGH"
    assert risk_engine.get_verdict(0.5999) == "MEDIUM"
    assert risk_engine.get_verdict(0.35) == "MEDIUM"
    assert risk_engine.get_verdict(0.3499) == "LOW"
    # No gaps or overlaps across the full [0, 1] range
    import random
    random.seed(42)
    for _ in range(200):
        r = round(random.uniform(0.0, 1.0), 4)
        verdict = risk_engine.get_verdict(r)
        assert verdict in {
            "CRITICAL", "HIGH", "MEDIUM", "LOW",
        }


def test_impersonation_risk_bounded_even_with_extreme_inputs():
    # Formula inputs are always in [0,1] by construction, but confirm the
    # clamp holds even if a future model returns something out-of-spec.
    assert risk_engine.compute_impersonation_risk(1.0, 1.0, 1.0) == 1.0
    assert risk_engine.compute_impersonation_risk(0.0, 0.0, 0.0) == 0.0
    assert risk_engine.compute_impersonation_risk(2.0, 2.0, 2.0) == 1.0  # clamped
    assert risk_engine.compute_impersonation_risk(-1.0, -1.0, -1.0) == 0.0  # clamped


def test_recommended_actions_are_deterministic_for_each_band():
    assert "monitor" in risk_engine.get_recommended_action(0.1).lower()
    assert "secondary verification" in risk_engine.get_recommended_action(0.4).lower()
    assert "independent trusted channel" in risk_engine.get_recommended_action(0.7).lower()
    assert "out-of-band security review" in risk_engine.get_recommended_action(0.9).lower()
