"""
urgency_detector.py
=====================
Classifies call urgency from transcript text, replacing manual urgency
selection (Task 4). Pure keyword matching — deliberately simple and
explainable (consistent with this project's stated preference for
transparent, justifiable logic over black-box behavior), not a trained
classifier. Keyword lists live in app/config.py so they're tunable
alongside every other risk parameter.

Returns one of "low" / "medium" / "high" — the exact same three values
app/services/context_engine.py's _urgency_risk() already expects, so no
changes were needed downstream: this just replaces how the value gets
INTO the pipeline (detected, not typed).
"""

from app.config import HIGH_URGENCY_KEYWORDS, MEDIUM_URGENCY_KEYWORDS


def detect_urgency(transcript: str) -> str:
    """
    Returns "high", "medium", or "low" based on keyword presence in the
    transcript (case-insensitive substring match). "high" takes priority
    over "medium" if both kinds of keywords are present.
    """
    if not transcript:
        return "low"

    text = transcript.lower()

    for keyword in HIGH_URGENCY_KEYWORDS:
        if keyword.lower() in text:
            return "high"

    for keyword in MEDIUM_URGENCY_KEYWORDS:
        if keyword.lower() in text:
            return "medium"

    return "low"


def matched_urgency_keywords(transcript: str) -> list:
    """
    Returns which keywords actually matched — useful for the dashboard's
    live transcript panel to show *why* a given urgency level was assigned
    (explainability, matching this project's design principle throughout).
    """
    if not transcript:
        return []
    text = transcript.lower()
    return [
        kw for kw in (HIGH_URGENCY_KEYWORDS + MEDIUM_URGENCY_KEYWORDS)
        if kw.lower() in text
    ]


def detect_urgency_detailed(transcript: str) -> dict:
    """
    Returns {"urgency": str, "confidence": float, "matched_keywords": [str]}
    — everything the dashboard's urgency badge + explainability panel needs
    in one call (Task 4: "confidence score" and "matched keywords").

    Confidence is a simple, explainable heuristic (not a statistical
    estimate) — consistent with this being a keyword classifier, not a
    trained model: each additional matched keyword at the winning tier adds
    confidence, capped at 1.0. "low" with zero matches gets a lower base
    confidence (0.4) than a keyword-backed classification, since the
    absence of a keyword is weaker evidence than its presence.
    """
    urgency = detect_urgency(transcript)
    matched = matched_urgency_keywords(transcript)

    if urgency == "high":
        tier_matches = [kw for kw in matched if kw in HIGH_URGENCY_KEYWORDS]
        confidence = min(1.0, 0.6 + 0.15 * len(tier_matches))
    elif urgency == "medium":
        tier_matches = [kw for kw in matched if kw in MEDIUM_URGENCY_KEYWORDS]
        confidence = min(1.0, 0.55 + 0.15 * len(tier_matches))
    else:
        confidence = 0.4 if not transcript else 0.6  # no keywords found

    return {
        "urgency": urgency,
        "confidence": round(confidence, 2),
        "matched_keywords": matched,
    }
