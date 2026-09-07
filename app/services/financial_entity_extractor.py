"""Deterministic, context-aware monetary entity extraction.

This module intentionally operates on ASR text after transcription.  It does
not rewrite or translate a multilingual transcript in order to find money.
"""

import json
import os
import re
import unicodedata
from typing import Optional

from app.config import FX_TO_INR_JSON

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Common Romanized Hindi wording observed in code-switched ASR output.
    "bees": 20,
    # Tamil cardinal numbers commonly preserved by Whisper. Compound forms
    # such as ஐம்பதாயிரம் are values in their own right; normal spaced
    # forms (ஐம்பது ஆயிரம்) are handled by the scale parser below.
    "பூஜ்யம்": 0, "ஒரு": 1, "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3,
    "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8,
    "ஒன்பது": 9, "பத்து": 10, "பதினைந்து": 15, "இருபது": 20,
    "முப்பது": 30, "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60,
    "எழுபது": 70, "எண்பது": 80, "தொண்ணூறு": 90,
    "பத்தாயிரம்": 10_000, "இருபதாயிரம்": 20_000,
    "ஐம்பதாயிரம்": 50_000,
    # Hindi cardinal numbers and common orthographic variants. The same
    # compositional parser handles e.g. बीस लाख and पचास हजार.
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5,
    "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60,
    "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}
_SCALES = {
    "hundred": 100, "thousand": 1_000, "k": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
    # Tamil magnitude forms, including practical inflections emitted by ASR.
    "ஆயிரம்": 1_000, "ஆயிரங்கள்": 1_000,
    "லட்சம்": 100_000, "லட்ச": 100_000, "லட்சங்கள்": 100_000,
    "கோடி": 10_000_000, "கோடிகள்": 10_000_000,
    # Hindi magnitude forms and common diacritic/inflection variants.
    "हजार": 1_000, "हज़ार": 1_000, "हजारों": 1_000, "हज़ारों": 1_000,
    "लाख": 100_000, "लाखों": 100_000,
    "करोड़": 10_000_000, "करोड़": 10_000_000, "करोड़ों": 10_000_000,
}
_INDIAN_SCALES = {
    "lakh", "lakhs", "lac", "lacs", "crore", "crores",
    "லட்சம்", "லட்ச", "லட்சங்கள்", "கோடி", "கோடிகள்",
    "लाख", "लाखों", "करोड़", "करोड़", "करोड़ों",
}
_NUMBER_TOKEN_PATTERN = "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))
_DURATION_RE = re.compile(
    rf"\b(?:\d+(?:\.\d+)?|{_NUMBER_TOKEN_PATTERN})\s+"
    r"(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years|"
    r"வினாடி(?:கள்)?|நிமிடம்(?:கள்)?|மணி(?:\s*நேரம்)?|நாள்(?:கள்)?|வாரம்(?:கள்)?|மாதம்(?:கள்)?|வருடம்(?:கள்)?|"
    r"सेकंड|मिनट(?:ों)?|घंटा|घंटे|दिन|हफ्ता|हफ्ते|महीना|महीने|साल)(?=\s|$|[.,!?])",
    re.I,
)
_TIME_RE = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
    r"(?:a\.?(?:m\.?)?|p\.?(?:m\.?)?)\b|"
    r"\b\d{1,2}(?::\d{2})?\s*(?:மணி|बजे)(?=\s|$|[.,!?])|"
    r"(?:\b(?:noon|midnight)\b|நண்பகல்|மதியம்|நள்ளிரவு|दोपहर|आधी\s*रात)(?=\s|$|[.,!?])",
    re.I,
)
_TRANSACTION_RE = re.compile(
    r"\b(?:transfer|send|pay|payment|amount|credit|debit|deposit|withdraw|remit|wire|withdrawal|"
    r"remittance|paying|money|transaction|funds?|balance|account|bank)\b|"
    r"(?:டிரான்ஸ்ஃபர்|அனுப்பு(?:ங்கள்)?|செலுத்து(?:ங்கள்)?|பணம்|தொகை|கணக்கு|வங்கி|ரூபா(?:ய்|ய்கள்|ய்க்கு|யை|யில்)?|தேவை|வேண்டும்)|"
    r"(?:भेजो|भेजना|भेजिए|भुगतान|पैसा|पैसे|रकम|राशि|धन|खाता|बैंक|रुपया|रुपये|रुपए|रूपये|जमा|निकालना|चाहिए)",
    re.I,
)
# Request words such as வேண்டும் / चाहिए can describe an ordinary count.
# Bare native-script numbers therefore need an actual payment, transfer,
# account, or bank signal before becoming INR.
_STRONG_TRANSACTION_RE = re.compile(
    r"\b(?:transfer|send|pay|payment|credit|debit|deposit|withdraw|remit|wire|withdrawal|remittance|"
    r"transaction|funds?|balance|account|bank)\b|"
    r"(?:டிரான்ஸ்ஃபர்|அனுப்பு(?:ங்கள்)?|செலுத்து(?:ங்கள்)?|கணக்கு|வங்கி)|"
    r"(?:भेजो|भेजना|भेजिए|भुगतान|खाता|बैंक|जमा|निकालना)",
    re.I,
)
_CURRENCIES = {
    "INR": ("₹", ("inr", "rs", "rupee", "rupees", "indian rupee", "indian rupees", "₹",
                  "ரூபாய்", "ரூபாய்கள்", "ரூபா", "ரூபாய்க்கு", "ரூபாயை", "ரூபாயில்",
                  "रुपया", "रुपये", "रुपए", "रूपये")),
    "USD": ("$", ("usd", "dollar", "dollars", "us dollar", "us dollars", "bucks", "$")),
    "EUR": ("€", ("eur", "euro", "euros", "€")),
    "GBP": ("£", ("gbp", "pound", "pounds", "british pound", "£")),
    "JPY": ("¥", ("jpy", "yen", "¥")),
    "AED": ("AED", ("aed", "dirham", "dirhams")),
    "SAR": ("SAR", ("sar", "riyal", "riyals", "saudi riyal", "saudi riyals")),
    "BDT": ("BDT", ("bdt", "taka", "bangladeshi taka")),
    "MYR": ("MYR", ("myr", "ringgit", "malaysian ringgit")),
    "SGD": ("SGD", ("sgd", "singapore dollar", "singapore dollars")),
    "AUD": ("A$", ("aud", "australian dollar", "australian dollars")),
    "CAD": ("C$", ("cad", "canadian dollar", "canadian dollars")),
    "CHF": ("CHF", ("chf", "swiss franc", "swiss francs")),
}

_CURRENCY_NAMES = {
    "INR": "Indian Rupee", "USD": "US Dollar", "EUR": "Euro",
    "GBP": "British Pound", "JPY": "Japanese Yen", "AED": "UAE Dirham",
    "SAR": "Saudi Riyal", "BDT": "Bangladeshi Taka", "MYR": "Malaysian Ringgit",
}
_INDIAN_LANGUAGE_RE = re.compile(r"[\u0900-\u097F\u0B80-\u0BFF]")


def _parse_integer_words(tokens: list[str]) -> Optional[float]:
    total = current = 0.0
    matched = False
    for token in tokens:
        if token == "and":
            continue
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            matched = True
        elif token in _SCALES:
            scale = _SCALES[token]
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
            matched = True
        else:
            return None
    value = total + current
    return value if matched and value > 0 else None


def _parse_words(words: str) -> Optional[float]:
    tokens = words.lower().split()
    if "point" not in tokens:
        return _parse_integer_words(tokens)
    point_index = tokens.index("point")
    whole = _parse_integer_words(tokens[:point_index])
    fractional_tokens = tokens[point_index + 1:]
    scale = next((token for token in fractional_tokens if token in _SCALES), None)
    digits = [token for token in fractional_tokens if token in _NUMBER_WORDS]
    if whole is None or not digits:
        return None
    fraction = float("0." + "".join(str(_NUMBER_WORDS[token]) for token in digits))
    return (whole + fraction) * _SCALES.get(scale, 1)


def _currency_for_context(text: str, scale: Optional[str]) -> tuple[str, str]:
    lower = text.casefold()
    # Explicit currency wins over the Indian denomination default.
    for code, (symbol, markers) in _CURRENCIES.items():
        if any(
            marker in {"₹", "$", "€", "£", "¥"} and marker in lower
            # Indic scripts frequently end words with combining characters,
            # for which Python's \b is not a dependable token boundary.
            # Native markers are complete lexicon entries and are therefore
            # matched directly; Latin markers retain word boundaries.
            or marker not in {"₹", "$", "€", "£", "¥"}
            and (
                marker in lower if not marker.isascii()
                else bool(re.search(rf"\b{re.escape(marker)}\b", lower))
            )
            for marker in markers
        ):
            return code, symbol
    if scale and scale.casefold() in _INDIAN_SCALES:
        return "INR", "₹"
    return "UNKNOWN", ""


def _format_indian(value: float) -> str:
    whole = str(int(round(value)))
    if len(whole) <= 3:
        return whole
    suffix, prefix = whole[-3:], whole[:-3]
    groups = []
    while prefix:
        groups.insert(0, prefix[-2:])
        prefix = prefix[:-2]
    return ",".join(groups + [suffix])


def _display_amount(value: float, currency: str, symbol: str) -> str:
    digits = _format_indian(value) if currency == "INR" else f"{int(round(value)):,}"
    if currency in {"INR", "USD", "EUR", "GBP", "JPY"}:
        return f"{symbol}{digits}"
    return f"{symbol} {digits}".strip() if currency != "UNKNOWN" else digits


def _duration_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _DURATION_RE.finditer(text)] + [match.span() for match in _TIME_RE.finditer(text)]


def _is_duration(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start >= left and end <= right for left, right in spans)


def _financial_context(text: str, start: int, end: int) -> tuple[bool, int, bool]:
    window_start, window_end = max(0, start - 55), min(len(text), end + 35)
    window = text[window_start:window_end]
    verbs = list(_TRANSACTION_RE.finditer(window))
    if not verbs:
        return False, 0, False
    distance = min(abs((window_start + verb.start()) - start) for verb in verbs)
    strong_context = bool(_STRONG_TRANSACTION_RE.search(window))
    return True, max(0, 100 - distance) + (20 if strong_context else 0), strong_context


def _candidate(text: str, start: int, end: int, value: float, scale: Optional[str], source: str,
               duration_spans: list[tuple[int, int]]) -> Optional[dict]:
    if _is_duration(start, end, duration_spans):
        return None
    local = text[max(0, start - 20):min(len(text), end + 25)]
    currency, symbol = _currency_for_context(local, scale)
    has_context, context_rank, strong_context = _financial_context(text, start, end)
    # A bare digit is not implicitly INR. However, a Tamil/Hindi transcript
    # with nearby transaction evidence is an Indian financial expression in
    # this product's supported-language scope, so it can safely take the
    # INR risk path while still preserving explicit foreign currencies.
    if currency == "UNKNOWN" and strong_context and _INDIAN_LANGUAGE_RE.search(text):
        currency, symbol = "INR", "₹"
    # Never turn a bare incidental number into money. A currency, denomination,
    # or nearby transaction verb is required.
    if currency == "UNKNOWN" and not scale and (
        not has_context or (_INDIAN_LANGUAGE_RE.search(text) and not strong_context)
    ):
        return None
    confidence = 0.95 if currency != "UNKNOWN" or scale else 0.72
    if has_context:
        confidence = min(0.98, confidence + 0.03)
    return {
        "amount": int(round(value)), "currency": currency, "currency_symbol": symbol,
        "currency_name": _CURRENCY_NAMES.get(currency, "Unknown currency"),
        "display_amount": _display_amount(value, currency, symbol), "spoken_amount": source,
        "source_text": source, "confidence": round(confidence, 2), "entity_type": "transaction_amount",
        # Currency/Indian-denomination evidence outranks a nearby bare number
        # such as a clock time, without selecting based on numerical size.
        "_rank": context_rank + (40 if currency != "UNKNOWN" else 0) + (30 if scale else 0),
        "_position": start,
    }


def extract_financial_entities(transcript: Optional[str]) -> dict:
    """Extract money candidates, their currency, and distinct duration phrases."""
    # Internal Unicode normalization makes canonically equivalent Tamil/Hindi
    # input match consistently. It never alters the transcript returned by
    # Whisper to the user or translates it to English.
    text = unicodedata.normalize("NFC", (transcript or "").strip())
    durations = [match.group(0) for match in _DURATION_RE.finditer(text)]
    times = [match.group(0) for match in _TIME_RE.finditer(text)]
    if not text:
        return {"primary": None, "amounts": [], "durations": durations, "times": times}
    spans = _duration_spans(text)
    candidates = []
    scale_pattern = "|".join(sorted(map(re.escape, _SCALES), key=len, reverse=True))
    digit_re = re.compile(r"(?<![\w.])(?:₹|\$|€|£|¥|INR\b|USD\b|EUR\b|GBP\b|AED\b|SGD\b|AUD\b|CAD\b|CHF\b|Rs\.?\s*)?\s*"
                          rf"(\d[\d,]*(?:\.\d+)?)\s*({scale_pattern})?", re.I)
    for match in digit_re.finditer(text):
        raw_number, scale = match.group(1), match.group(2)
        try:
            value = float(raw_number.replace(",", "")) * _SCALES.get((scale or "").casefold(), 1)
        except ValueError:
            continue
        item = _candidate(text, *match.span(), value, scale, match.group(0).strip(), spans)
        if item:
            candidates.append(item)
    word_tokens = "|".join([*map(re.escape, _NUMBER_WORDS), *map(re.escape, _SCALES), "point", "and"])
    # Do not use \b here: Indic words often end with a combining mark that
    # Python does not treat as a word character. Number words in ASR output
    # are whitespace-delimited, so explicit surrounding separators are both
    # clearer and multilingual-safe.
    word_re = re.compile(
        rf"(?<!\S)((?:(?:{word_tokens})\s+){{0,6}}(?:{word_tokens}))(?=\s|$|[.,!?])",
        re.I,
    )
    for match in word_re.finditer(text):
        phrase = match.group(1)
        if phrase.casefold().split()[0] in _SCALES:
            continue
        value = _parse_words(phrase)
        if value is None:
            continue
        scale = next((token for token in phrase.casefold().split() if token in _SCALES), None)
        item = _candidate(text, *match.span(), value, scale, phrase, spans)
        if item and not any(item["_position"] == old["_position"] for old in candidates):
            candidates.append(item)
    candidates.sort(key=lambda item: (-item["_rank"], item["_position"]))
    for item in candidates:
        item.pop("_rank", None)
        item.pop("_position", None)
    return {
        "primary": candidates[0] if candidates else None,
        "amounts": candidates,
        "durations": durations,
        "times": times,
    }


def amount_to_inr(amount: Optional[float], currency: Optional[str]) -> Optional[float]:
    """Return a configured INR risk basis, never a guessed conversion.

    ``VISL_FX_TO_INR_JSON`` is a deliberate prototype-only configuration
    table. Production should inject rates from an approved institutional FX
    service. Missing, malformed, or non-positive rates make foreign monetary
    risk unknown rather than treating the value as INR.
    """
    if amount is None or not currency:
        return None
    if currency == "INR":
        return float(amount)
    try:
        configured = json.loads(os.environ.get("VISL_FX_TO_INR_JSON", FX_TO_INR_JSON))
        rate = float(configured.get(currency))
    except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return None
    return float(amount) * rate if rate > 0 else None
