from app.services.entity_extraction import extract_amount, format_inr


def test_spelled_out_thousand():
    assert extract_amount("Transfer fifty thousand rupees immediately") == 50000


def test_lakhs():
    assert extract_amount("Send 2 lakhs to this account") == 200000


def test_crore_decimal():
    assert extract_amount("Send 1.5 crore to complete the deal") == 15_000_000


def test_digit_amount_with_currency_marker():
    assert extract_amount("Please send rs 50000 now") == 50000
    assert extract_amount("Transfer \u20b950,000 to account") == 50000


def test_bare_large_number_without_marker():
    assert extract_amount("Wire 300000 immediately") == 300000


def test_one_lakh_spelled_out():
    assert extract_amount("Please deposit one lakh rupees") == 100000


def test_no_amount_present_returns_none():
    assert extract_amount("Call me at 2 pm tomorrow") is None
    assert extract_amount("Hello how are you") is None
    assert extract_amount("") is None
    assert extract_amount(None) is None


def test_format_inr():
    assert format_inr(50000) == "\u20b950,000"
    assert format_inr(None) == "Not detected"
    assert format_inr(200000) == "\u20b9200,000"


def test_extract_amount_details_matches_spec_shape():
    from app.services.entity_extraction import extract_amount_details

    assert extract_amount_details("transfer fifty thousand") == {
        "amount": 50000.0, "amount_text": "\u20b950,000",
    }
    assert extract_amount_details("send 2 lakhs") == {
        "amount": 200000.0, "amount_text": "\u20b9200,000",
    }
    assert extract_amount_details("pay 999 rupees") == {
        "amount": 999.0, "amount_text": "\u20b9999",
    }
    assert extract_amount_details("hello there") == {
        "amount": None, "amount_text": "Not detected",
    }
