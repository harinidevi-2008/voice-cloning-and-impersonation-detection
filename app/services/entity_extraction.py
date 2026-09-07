"""Backward-compatible wrappers around the financial entity extractor."""

from typing import Optional

from app.services.financial_entity_extractor import extract_financial_entities


def extract_amount(transcript: Optional[str]) -> Optional[int]:
    """Return the primary transaction amount, preserving the historic API."""
    primary = extract_financial_entities(transcript)["primary"]
    return primary["amount"] if primary else None


def format_inr(amount: Optional[float]) -> str:
    """Legacy INR formatter retained for callers that explicitly request it."""
    if amount is None:
        return "Not detected"
    return f"₹{int(round(amount)):,}"


def extract_amount_details(transcript: Optional[str]) -> dict:
    """Return the structured primary financial entity without changing text."""
    primary = extract_financial_entities(transcript)["primary"]
    if not primary:
        return {"amount": None, "amount_text": "Not detected"}
    return {"amount": float(primary["amount"]), "amount_text": primary["display_amount"], **primary}
