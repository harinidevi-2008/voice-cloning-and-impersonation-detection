import math

from app.services.entity_extraction import extract_amount
from app.services.urgency_detector import detect_urgency_detailed
from app.services import risk_engine
from app.services.prosody_analyzer import analyze_prosody
from app.services.transcription_service import transcribe_detailed


def test_extract_amount_supports_indian_variants():
    samples = [
        "₹50,000",
        "Rs 50000",
        "INR 50000",
        "50 thousand",
        "5 lakh",
        "2 lakh",
        "1 crore",
        "1.5 crore",
    ]
    for sample in samples:
        amount = extract_amount(sample)
        assert amount is not None, sample
        assert amount > 0


def test_detect_urgency_handles_multilingual_phrases():
    english = detect_urgency_detailed("Please transfer the money immediately")
    tamil = detect_urgency_detailed("இப்போதே பணம் மாற்ற urgently")
    hindi = detect_urgency_detailed("तुरंत पैसे ट्रांसफर करें")

    assert english["urgency"] == "high"
    assert tamil["urgency"] == "high"
    assert hindi["urgency"] == "high"
    assert tamil["matched_keywords"]
    assert hindi["matched_keywords"]


def test_risk_engine_includes_prosody_signal():
    total = risk_engine.compute_impersonation_risk(
        spoof_score=0.6,
        identity_mismatch_risk=0.4,
        context_risk=0.3,
        prosody_risk=0.8,
    )
    assert total > 0.5
    assert total <= 1.0


def test_prosody_analysis_handles_silence_without_crashing():
    features = analyze_prosody("tests/data/silence.wav")
    assert features["status"] in {"silence", "short", "invalid", "ok"}
    assert features["prosody_score"] in (None, 0.0)


def test_transcription_detailed_response_is_structured_in_mock_mode():
    result = transcribe_detailed("unused.wav", filename_hint="urgent_50000.wav")
    assert set(result) >= {"text", "language", "language_probability"}
    assert result["language"] == "en"


def test_low_confidence_prosody_does_not_change_legacy_fusion():
    baseline = risk_engine.compute_impersonation_risk(0.8, 0.5, 0.1)
    weak = risk_engine.compute_impersonation_risk(0.8, 0.5, 0.1, prosody_risk=1.0, prosody_confidence=0.0)
    assert weak == baseline


def test_risk_actions_escalate_by_score():
    assert "monitoring" in risk_engine.get_recommended_action(0.1).lower()
    assert "secondary verification" in risk_engine.get_recommended_action(0.5).lower()
    assert "block or hold" in risk_engine.get_recommended_action(0.9).lower()
