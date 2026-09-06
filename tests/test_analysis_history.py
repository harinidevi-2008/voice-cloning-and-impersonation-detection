from app.db import analysis_db


def test_analysis_history_persists_the_actual_claimed_user_id(monkeypatch, tmp_path):
    database_path = tmp_path / "analysis.db"
    monkeypatch.setattr(analysis_db, "ANALYSIS_DB_PATH", str(database_path))
    analysis_db.init_db()

    analysis_db.save_analysis(
        transcript="Please transfer money",
        spoof_score=0.25,
        similarity=0.91,
        amount=500.0,
        urgency="medium",
        risk="LOW_RISK_LIKELY_GENUINE",
        speaker_name="Harini",
        speaker_user_id=1,
    )

    records = analysis_db.list_recent_analyses()
    assert len(records) == 1
    assert records[0]["speaker_name"] == "Harini"
    assert records[0]["speaker_user_id"] == 1
