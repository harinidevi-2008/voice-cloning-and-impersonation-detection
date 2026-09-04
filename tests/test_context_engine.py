from datetime import datetime
from app.services import context_engine


def test_known_caller_lower_risk_than_unknown():
    known = context_engine.compute_context_risk(True, 1000, "low")
    unknown = context_engine.compute_context_risk(False, 1000, "low")
    assert unknown["context_risk"] > known["context_risk"]


def test_high_value_increases_risk():
    low_value = context_engine.compute_context_risk(True, 100, "low")
    high_value = context_engine.compute_context_risk(True, 1_000_000, "low")
    assert high_value["context_risk"] > low_value["context_risk"]


def test_urgency_increases_risk():
    low_urgency = context_engine.compute_context_risk(True, 1000, "low")
    high_urgency = context_engine.compute_context_risk(True, 1000, "high")
    assert high_urgency["context_risk"] > low_urgency["context_risk"]


def test_unusual_time_window_wraps_midnight():
    late_night = datetime(2024, 1, 1, 2, 0)   # 2 AM -> unusual
    midday = datetime(2024, 1, 1, 14, 0)      # 2 PM -> normal
    r1 = context_engine.compute_context_risk(True, 1000, "low", check_time=late_night)
    r2 = context_engine.compute_context_risk(True, 1000, "low", check_time=midday)
    assert r1["breakdown"]["time_risk"] == 1.0
    assert r2["breakdown"]["time_risk"] == 0.0
