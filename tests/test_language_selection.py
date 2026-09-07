from app.services.transcription_service import (
    _candidate_score,
    _requires_candidate_verification,
    _select_candidate,
)


def _segments(logprob, no_speech=0.05, text="text"):
    return [{
        "start": 0.0, "end": 2.0, "text": text,
        "avg_logprob": logprob, "no_speech_prob": no_speech,
        "compression_ratio": 1.1,
    }]


def test_high_confidence_supported_languages_keep_automatic_selection():
    for language in ("ta", "hi", "en"):
        assert not _requires_candidate_verification(language, 0.91, 2.0)


def test_low_confidence_and_unsupported_language_trigger_verification():
    assert _requires_candidate_verification("ta", 0.40, 2.0)
    assert _requires_candidate_verification("hi", 0.40, 2.0)
    assert _requires_candidate_verification("ml", 0.90, 2.0)
    assert _requires_candidate_verification("en", 0.95, 0.5)


def test_candidate_selection_uses_quality_evidence_not_language_pair_rules():
    candidates = {
        "ta": ("என் கணக்குக்கு பணம் அனுப்புங்கள்", _segments(-0.12, text="என் கணக்குக்கு பணம் அனுப்புங்கள்")),
        "hi": ("गलत प्रतिलेख", _segments(-1.20, text="गलत प्रतिलेख")),
        "en": ("unrelated transcription", _segments(-1.10, text="unrelated transcription")),
    }
    assert _select_candidate(candidates) == "ta"
    assert _candidate_score("ta", *candidates["ta"]) > _candidate_score("hi", *candidates["hi"])


def test_code_switched_transcripts_remain_usable_for_supported_candidates():
    tamil_english = "என் accountக்கு 20 லட்சம் transfer பண்ணுங்க"
    hindi_english = "मेरे account में 20 लाख transfer करो"
    assert _candidate_score("ta", tamil_english, _segments(-0.2, text=tamil_english)) > float("-inf")
    assert _candidate_score("hi", hindi_english, _segments(-0.2, text=hindi_english)) > float("-inf")
