from app.db import analysis_db
from app.services.preventive_actions import generate_preventive_actions


def test_base_actions_cover_all_risk_tiers():
    assert generate_preventive_actions("LOW", 0.1, 0, "low") == [
        "Proceed with the transaction.", "Log the interaction.", "No further verification required.",
    ]
    assert generate_preventive_actions("MEDIUM", 0.1, 0, "low") == [
        "Verify one personal identity detail.", "Confirm payment using the registered number.",
        "Do not share OTP or PIN.", "Record the interaction.",
    ]
    assert generate_preventive_actions("HIGH", 0.1, 0, "low") == [
        "Pause the transaction immediately.", "Verify identity via another channel.",
        "Contact the organization directly.", "Escalate to fraud monitoring.",
    ]
    assert generate_preventive_actions("CRITICAL", 0.1, 0, "low") == [
        "Block the transaction.", "Disconnect the call.", "Freeze payment authorization.",
        "Preserve recording as evidence.", "Notify cybersecurity response.",
    ]


def test_evidence_additions_are_preserved_and_capped():
    assert "Require dual approval before payment." in generate_preventive_actions("LOW", 0.1, 100_001, "low")
    assert "Ignore pressure tactics requesting immediate action." in generate_preventive_actions("LOW", 0.1, 0, "high")
    assert "Treat the voice as potentially AI-generated." in generate_preventive_actions("LOW", 0.71, 0, "low")
    assert len(generate_preventive_actions("HIGH", 0.9, 500_000, "high")) == 5


def test_actions_round_trip_through_analysis_database():
    analysis_db.init_db()
    actions = generate_preventive_actions("HIGH", 0.8, 200_000, "high")
    call_id = analysis_db.save_analysis(
        transcript="test", spoof_score=0.8, similarity=None, amount=200_000,
        urgency="high", risk="HIGH", preventive_actions=actions,
    )
    record = next(item for item in analysis_db.list_recent_analyses(limit=100) if item["call_id"] == call_id)
    assert record["preventive_actions"] == actions
