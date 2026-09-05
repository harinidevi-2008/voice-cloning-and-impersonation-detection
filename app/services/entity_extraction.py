"""
entity_extraction.py
=====================
Extracts a monetary amount (in rupees) from a call transcript, replacing
manual transaction-amount entry (Task 3). Pure Python / regex — no ML
model needed for this part, so it's testable and demoable without any
heavy dependency, and runs instantly regardless of AI_BACKEND.

Supports:
  - Plain digits with optional currency words/symbols: "rs 50000", "₹50,000"
  - Indian numbering scale: "2 lakhs" -> 200000, "1.5 crore" -> 15000000
  - Spelled-out English numbers up to crores: "fifty thousand" -> 50000

Deliberately conservative: returns None (not 0) when no amount is found,
so callers can distinguish "no amount mentioned" from "amount is zero" —
see app/routers/analyze.py for how the None case is handled.
"""

import re
from typing import Optional

_WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000,
}

# Matches digit-based amounts with optional currency markers and Indian/
# English scale words directly after the number, e.g. "50000", "2.5 lakh",
# "rs. 1,20,000", "₹300000".
_DIGIT_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s*(lakhs?|lacs?|crores?|thousand|million)?",
    re.IGNORECASE,
)


def _parse_number_word_sequence(words: list) -> Optional[int]:
    """
    Parses a short sequence of number/scale words (e.g. ["fifty", "thousand"]
    or ["two", "lakh"]) into an integer. Returns None if the sequence
    doesn't resolve to a recognizable number.
    """
    total = 0
    current = 0
    matched_any = False

    for word in words:
        w = word.lower().strip(",.")
        if w in _WORD_TO_NUMBER:
            current += _WORD_TO_NUMBER[w]
            matched_any = True
        elif w in _SCALE_WORDS:
            scale = _SCALE_WORDS[w]
            if scale >= 1000:
                # "fifty thousand" -> (50) * 1000 added to running total
                current = current or 1
                total += current * scale
                current = 0
            else:  # "hundred"
                current = (current or 1) * scale
            matched_any = True
        else:
            # Stop at the first word that isn't part of the number phrase.
            break

    total += current
    return total if matched_any and total > 0 else None


def extract_amount(transcript: str) -> Optional[float]:
    """
    Returns the detected transaction amount as a float (in rupees), or
    None if no amount could be confidently extracted from the transcript.
    """
    if not transcript or not transcript.strip():
        return None

    text = transcript.strip()

    # --- Try digit-based amounts first (most common in real transcripts) ---
    for match in _DIGIT_AMOUNT_RE.finditer(text):
        digits_str, scale_word = match.groups()
        if not digits_str:
            continue
        has_currency_marker = bool(
            re.match(r"^\s*(rs\.?|inr|₹)", text[max(0, match.start() - 5):match.start() + 3], re.IGNORECASE)
        )
        cleaned_digits = digits_str.replace(",", "")
        try:
            value = float(cleaned_digits)
        except ValueError:
            continue

        if scale_word:
            scale_key = scale_word.lower()
            multiplier = _SCALE_WORDS.get(scale_key) or _SCALE_WORDS.get(scale_key.rstrip("s"))
            if multiplier:
                value *= multiplier
            return value
        if has_currency_marker or len(cleaned_digits.split(".")[0]) >= 3:
            return value

    # --- Fall back to spelled-out numbers: "fifty thousand", "two lakh" ---
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    number_words = set(_WORD_TO_NUMBER) | set(_SCALE_WORDS)
    i = 0
    while i < len(tokens):
        if tokens[i] in number_words:
            j = i
            while j < len(tokens) and tokens[j] in number_words:
                j += 1
            result = _parse_number_word_sequence(tokens[i:j])
            if result is not None:
                return float(result)
            i = j
        else:
            i += 1

    return None


def format_inr(amount: Optional[float]) -> str:
    """Formats an amount as an Indian-style rupee string, e.g. '₹50,000'."""
    if amount is None:
        return "Not detected"
    return f"₹{int(round(amount)):,}"


def extract_amount_details(transcript: str) -> dict:
    """
    Returns {"amount": float_or_None, "amount_text": str} — the exact
    shape requested for Task 3's example (e.g.
    {"amount": 50000, "amount_text": "₹50,000"}).

    NOTE on why this isn't the actual AnalyzeResponse field shape: the
    established API contract already has a `detected_amount: float` field
    (app/schemas.py), documented and tested since the previous refactor.
    Changing that field's type to a nested object would be a breaking
    change to an already-shipped additive field, not just an extension —
    so the API keeps `detected_amount` as a bare float, and this function
    exists as a convenience for exactly the requested shape wherever code
    wants it (e.g. it's what the dashboard's metric card actually reads
    from, computing amount_text via format_inr() rather than duplicating
    that logic). Behavior is otherwise identical to extract_amount().
    """
    amount = extract_amount(transcript)
    return {"amount": amount, "amount_text": format_inr(amount)}
