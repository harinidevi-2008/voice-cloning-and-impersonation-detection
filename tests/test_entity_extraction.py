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
    assert extract_amount("Transfer INR 50000 now") == 50000
    assert extract_amount("Transfer 50,000 rupees now") == 50000


def test_requested_indian_and_english_scales():
    assert extract_amount("Send 2 lakh now") == 200000
    assert extract_amount("Send 2 lakhs now") == 200000
    assert extract_amount("Send 1 crore now") == 10_000_000
    assert extract_amount("Send 50 thousand now") == 50000


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

    unknown = extract_amount_details("transfer fifty thousand")
    assert unknown["amount"] == 50000.0
    assert unknown["currency"] == "UNKNOWN"
    assert unknown["amount_text"] == "50,000"
    indian = extract_amount_details("send 2 lakhs")
    assert indian["amount"] == 200000.0
    assert indian["currency"] == "INR"
    assert indian["amount_text"] == "\u20b92,00,000"
    rupees = extract_amount_details("pay 999 rupees")
    assert rupees["amount"] == 999.0
    assert rupees["currency"] == "INR"
    assert rupees["amount_text"] == "\u20b9999"
    assert extract_amount_details("hello there") == {
        "amount": None, "amount_text": "Not detected",
    }
