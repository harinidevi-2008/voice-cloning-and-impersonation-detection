"""
context_engine.py
==================
Pure, framework-agnostic functions that turn call/transaction metadata into
a context_risk score in [0, 1]. Kept separate from risk_engine.py so each
piece of the formula can be tested and explained independently.

context_risk = 0.40*known_risk + 0.30*value_risk + 0.20*urgency_risk + 0.10*time_risk
(weights configurable in app/config.py)
"""

from datetime import datetime
from typing import Optional

from app.config import (
    CONTEXT_WEIGHTS,
    MED_VALUE_THRESHOLD,
    HIGH_VALUE_THRESHOLD,
    URGENCY_RISK_MAP,
    DEFAULT_URGENCY_RISK,
    UNUSUAL_TIME_START_HOUR,
    UNUSUAL_TIME_END_HOUR,
)


def _known_risk(caller_known: bool) -> float:
    return 0.0 if caller_known else 1.0


def _value_risk(transaction_value: float) -> float:
    """
    Piecewise-linear normalization of transaction value into [0, 1]:
      - 0 .. MED_VALUE_THRESHOLD          -> 0.0 .. 0.5
      - MED_VALUE_THRESHOLD .. HIGH..     -> 0.5 .. 1.0
      - >= HIGH_VALUE_THRESHOLD           -> 1.0
    """
    if transaction_value is None or transaction_value <= 0:
        return 0.0
    if transaction_value >= HIGH_VALUE_THRESHOLD:
        return 1.0
    if transaction_value >= MED_VALUE_THRESHOLD:
        span = HIGH_VALUE_THRESHOLD - MED_VALUE_THRESHOLD
        progress = (transaction_value - MED_VALUE_THRESHOLD) / span
        return 0.5 + 0.5 * progress
    return 0.5 * (transaction_value / MED_VALUE_THRESHOLD)


def _urgency_risk(urgency: str) -> float:
    return URGENCY_RISK_MAP.get((urgency or "").strip().lower(), DEFAULT_URGENCY_RISK)


def _unusual_time_risk(check_time: Optional[datetime] = None) -> float:
    """
    1.0 if the call falls inside the configured "unusual hours" window
    (default 11 PM - 5 AM local server time), else 0.0.
    Wraps around midnight correctly (start hour > end hour).
    """
    t = check_time or datetime.now()
    hour = t.hour

    if UNUSUAL_TIME_START_HOUR > UNUSUAL_TIME_END_HOUR:
        is_unusual = hour >= UNUSUAL_TIME_START_HOUR or hour < UNUSUAL_TIME_END_HOUR
    else:
        is_unusual = UNUSUAL_TIME_START_HOUR <= hour < UNUSUAL_TIME_END_HOUR

    return 1.0 if is_unusual else 0.0


def compute_context_risk(
    caller_known: bool,
    transaction_value: float,
    urgency: str,
    check_time: Optional[datetime] = None,
) -> dict:
    """
    Returns:
        {
            "context_risk": float,       # final weighted score in [0, 1]
            "breakdown": {                # kept for internal transparency/tests/logs
                "known_risk": float,
                "value_risk": float,
                "urgency_risk": float,
                "time_risk": float,
            }
        }
    Only `context_risk` is surfaced in the public /analyze response, per the
    API contract; `breakdown` exists so the score is never a black box
    internally (useful for debugging, tests, and future dashboard drill-down).
    """
    known_risk = _known_risk(caller_known)
    value_risk = _value_risk(transaction_value)
    urgency_risk = _urgency_risk(urgency)
    time_risk = _unusual_time_risk(check_time)

    w = CONTEXT_WEIGHTS
    context_risk = (
        w["known"] * known_risk
        + w["value"] * value_risk
        + w["urgency"] * urgency_risk
        + w["time"] * time_risk
    )

    return {
        "context_risk": round(context_risk, 4),
        "breakdown": {
            "known_risk": round(known_risk, 4),
            "value_risk": round(value_risk, 4),
            "urgency_risk": round(urgency_risk, 4),
            "time_risk": round(time_risk, 4),
        },
    }
