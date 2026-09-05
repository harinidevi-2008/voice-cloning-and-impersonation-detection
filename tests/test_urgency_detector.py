from app.services.urgency_detector import detect_urgency, matched_urgency_keywords, detect_urgency_detailed


def test_high_urgency_keywords():
    assert detect_urgency("Please transfer this immediately") == "high"
    assert detect_urgency("This is urgent, send now") == "high"
    assert detect_urgency("Do it right now") == "high"
    assert detect_urgency("Don't tell anyone about this transfer") == "high"
    assert detect_urgency("This is an emergency situation") == "high"
    assert detect_urgency("Please act quickly") == "high"


def test_medium_urgency_keywords():
    assert detect_urgency("Please send this soon") == "medium"


def test_low_urgency_default():
    assert detect_urgency("Just checking in, how are you") == "low"
    assert detect_urgency("Hello, this is your bank calling") == "low"
    assert detect_urgency("") == "low"
    assert detect_urgency(None) == "low"


def test_high_takes_priority_over_medium():
    # Contains both a "medium" word (soon) and a "high" word (urgent)
    assert detect_urgency("This is urgent, please send soon") == "high"


def test_case_insensitive():
    assert detect_urgency("IMMEDIATELY send the money") == "high"


def test_matched_keywords_returned_for_explainability():
    matches = matched_urgency_keywords("This is urgent, please hurry")
    assert "urgent" in matches
    assert "hurry" in matches


def test_detect_urgency_detailed_matches_spec_example():
    result = detect_urgency_detailed(
        "Hello Harini, please transfer fifty thousand immediately, don't tell anyone"
    )
    assert result["urgency"] == "high"
    assert "immediately" in result["matched_keywords"]
    assert "don't tell anyone" in result["matched_keywords"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_detect_urgency_detailed_more_keywords_higher_confidence():
    one_keyword = detect_urgency_detailed("This is urgent")
    two_keywords = detect_urgency_detailed("This is urgent, act now")
    assert two_keywords["confidence"] >= one_keyword["confidence"]


def test_detect_urgency_detailed_low_has_no_keywords():
    result = detect_urgency_detailed("Just checking in, how are you")
    assert result["urgency"] == "low"
    assert result["matched_keywords"] == []