from app.services.financial_entity_extractor import amount_to_inr, extract_financial_entities


def primary(text):
    return extract_financial_entities(text)["primary"]


def test_indian_amount_and_duration_are_distinct():
    result = extract_financial_entities("transfer 20 lakhs to my account within 1 month")
    assert result["primary"]["amount"] == 2_000_000
    assert result["primary"]["currency"] == "INR"
    assert result["primary"]["display_amount"] == "₹20,00,000"
    assert result["durations"] == ["1 month"]


def test_supported_currencies_are_not_mislabeled_as_inr():
    assert (primary("transfer 5000 rupees")["amount"], primary("transfer 5000 rupees")["currency"]) == (5000, "INR")
    assert (primary("transfer 20 dollars")["amount"], primary("transfer 20 dollars")["currency"]) == (20, "USD")
    assert primary("transfer $500")["display_amount"] == "$500"
    assert primary("transfer 200 euros")["currency"] == "EUR"
    assert primary("transfer 100 pounds")["currency"] == "GBP"


def test_context_words_and_spoken_numbers_select_money():
    assert primary("transfer an amount of 20 lakhs")["amount"] == 2_000_000
    assert primary("transfer twenty lakhs")["amount"] == 2_000_000
    assert primary("transfer 2 crore")["amount"] == 20_000_000
    dollar = primary("transfer 2 million dollars")
    assert (dollar["amount"], dollar["currency"]) == (2_000_000, "USD")
    assert primary("credit an amount of 10,000")["currency"] == "UNKNOWN"


def test_duration_never_becomes_a_transaction_amount():
    assert primary("complete this within 1 month") is None
    mixed = extract_financial_entities("transfer 20 lakhs within 1 month")
    assert mixed["primary"]["amount"] == 2_000_000
    assert "1 month" in mixed["durations"]


def test_code_switched_financial_context_uses_same_parser():
    tamil = primary("என் accountக்கு 20 lakhs transfer பண்ணுங்க")
    hindi = primary("20 lakh rupees transfer kar do")
    assert (tamil["amount"], tamil["currency"]) == (2_000_000, "INR")
    assert (hindi["amount"], hindi["currency"]) == (2_000_000, "INR")


def test_duration_clock_time_and_currency_candidates_are_scored_semantically():
    assert (primary("Send 5 crore within 2 months")["amount"], primary("Send 5 crore within 2 months")["currency"]) == (50_000_000, "INR")
    assert (primary("Transfer 500 dollars within 3 days")["amount"], primary("Transfer 500 dollars within 3 days")["currency"]) == (500, "USD")
    clock = primary("Call me at 2 pm and transfer 500 dollars")
    assert (clock["amount"], clock["currency"]) == (500, "USD")
    assert primary("Meet me in 2 hours") is None
    assert primary("There are 3 people on the call") is None


def test_clock_times_and_all_duration_units_are_never_money():
    detailed_time = extract_financial_entities("Transfer after 5:30 pm")
    assert detailed_time["primary"] is None
    assert detailed_time["times"] == ["5:30 pm"]
    assert primary("Call me at noon") is None
    assert primary("Wait 30 seconds before calling") is None
    assert primary("Send it after 1 year") is None


def test_extended_currencies_and_spoken_decimals_are_supported():
    assert primary("Transfer ₹20,00,000 within 2 months")["amount"] == 2_000_000
    assert primary("Send fifty thousand rupees")["amount"] == 50_000
    assert primary("Please transfer bees lakh to my account")["amount"] == 2_000_000
    assert primary("one point five crore")["amount"] == 15_000_000
    assert primary("Send 100 dirhams")["currency"] == "AED"
    assert primary("Send 100 Saudi riyal")["currency"] == "SAR"
    assert primary("Send 100 taka")["currency"] == "BDT"
    assert primary("Send 100 ringgit")["currency"] == "MYR"


def test_tamil_money_currency_context_and_indian_scales():
    assert (primary("எனக்கு 50000 ரூபாய் வேண்டும்")["amount"], primary("எனக்கு 50000 ரூபாய் வேண்டும்")["currency"]) == (50_000, "INR")
    assert (primary("என் accountக்கு 50000 transfer பண்ணுங்க")["amount"], primary("என் accountக்கு 50000 transfer பண்ணுங்க")["currency"]) == (50_000, "INR")
    assert (primary("20 லட்சம் ரூபாய் transfer பண்ணுங்க")["amount"], primary("20 லட்சம் ரூபாய் transfer பண்ணுங்க")["currency"]) == (2_000_000, "INR")
    assert primary("2 கோடி ரூபாய் transfer பண்ணுங்க")["amount"] == 20_000_000
    assert primary("ஐம்பது ஆயிரம் ரூபாய் அனுப்புங்கள்")["amount"] == 50_000
    assert primary("ஐம்பதாயிரம் ரூபாய் தேவை")["amount"] == 50_000


def test_hindi_money_currency_context_and_indian_scales():
    assert (primary("मुझे 50000 रुपये चाहिए")["amount"], primary("मुझे 50000 रुपये चाहिए")["currency"]) == (50_000, "INR")
    assert (primary("मेरे account में 50000 transfer करो")["amount"], primary("मेरे account में 50000 transfer करो")["currency"]) == (50_000, "INR")
    assert (primary("20 लाख रुपये transfer करो")["amount"], primary("20 लाख रुपये transfer करो")["currency"]) == (2_000_000, "INR")
    assert primary("2 करोड़ रुपये transfer करो")["amount"] == 20_000_000
    assert primary("पचास हजार रुपये भेजो")["amount"] == 50_000


def test_native_duration_and_time_are_not_money():
    tamil_duration = extract_financial_entities("2 மணி நேரம் காத்திருங்கள்")
    hindi_duration = extract_financial_entities("2 घंटे बाद कॉल करो")
    tamil_time = extract_financial_entities("5 மணி வருங்கள்")
    hindi_time = extract_financial_entities("5 बजे आना")
    assert tamil_duration["primary"] is None and tamil_duration["durations"] == ["2 மணி நேரம்"]
    assert hindi_duration["primary"] is None and hindi_duration["durations"] == ["2 घंटे"]
    assert tamil_time["primary"] is None and tamil_time["times"] == ["5 மணி"]
    assert hindi_time["primary"] is None and hindi_time["times"] == ["5 बजे"]
    assert primary("எனக்கு 5 வேண்டும்") is None
    assert primary("मुझे 5 चाहिए") is None


def test_fx_normalization_is_explicit_and_never_assumes_inr(monkeypatch):
    assert amount_to_inr(20, "USD") is None
    monkeypatch.setenv("VISL_FX_TO_INR_JSON", '{"USD": 83.0}')
    assert amount_to_inr(20, "USD") == 1660.0
    assert amount_to_inr(20, "INR") == 20.0
