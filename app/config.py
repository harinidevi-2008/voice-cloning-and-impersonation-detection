"""
Central configuration for the Voice Integrity Security Layer backend.

Every tunable number in the risk/context model lives here so the formula
stays transparent and easy to justify in a demo ("why did this get flagged?"
-> "because of these weights, in this file").
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # member2_backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_UPLOAD_DIR = os.path.join(DATA_DIR, "audio_uploads")
DB_PATH = os.path.join(DATA_DIR, "voice_integrity.db")

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB, generous for a demo

# ---------------------------------------------------------------------------
# Final impersonation-risk fusion weights
# ImpersonationRisk = 0.5*SpoofRisk + 0.3*IdentityMismatchRisk + 0.2*ContextRisk
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "spoof": 0.5,
    "identity": 0.3,
    "context": 0.2,
}

# ---------------------------------------------------------------------------
# Context sub-weights (must sum to 1.0)
# ---------------------------------------------------------------------------
CONTEXT_WEIGHTS = {
    "known": 0.40,     # caller_known == False -> risk
    "value": 0.30,     # transaction_value magnitude -> risk
    "urgency": 0.20,   # urgency level -> risk
    "time": 0.10,      # unusual call time -> risk
}

# Transaction value bands (in whatever currency unit the demo uses, e.g. INR/USD)
MED_VALUE_THRESHOLD = 50_000
HIGH_VALUE_THRESHOLD = 500_000

# Urgency keyword -> risk contribution
URGENCY_RISK_MAP = {
    "low": 0.1,
    "medium": 0.5,
    "high": 1.0,
}
DEFAULT_URGENCY_RISK = 0.5  # fallback if an unrecognized string is sent

# "Unusual time" window, treated as higher risk (24h clock, local server time)
UNUSUAL_TIME_START_HOUR = 23  # 11 PM
UNUSUAL_TIME_END_HOUR = 5     # 5 AM

# ---------------------------------------------------------------------------
# Verdict thresholds on the final impersonation_risk score [0, 1]
# ---------------------------------------------------------------------------
VERDICT_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
}

VERDICT_LABELS = {
    "high": "HIGH_RISK_LIKELY_IMPERSONATION",
    "medium": "MEDIUM_RISK_MANUAL_REVIEW",
    "low": "LOW_RISK_LIKELY_GENUINE",
}

# ---------------------------------------------------------------------------
# AI backend selection
# ---------------------------------------------------------------------------
# "mock" (default): deterministic hash-based stand-ins, no heavy ML deps.
# "real": Member 1's actual XLS-R+AASIST / ECAPA-TDNN models. Requires
#         requirements-real-ai.txt to be installed. See
#         app/services/ai_service.py and app/services/real_ai_service.py.
AI_BACKEND = os.environ.get("VISL_AI_BACKEND", "mock").strip().lower()

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ALLOW_ORIGINS = ["*"]  # hackathon-friendly; tighten before any real deployment
