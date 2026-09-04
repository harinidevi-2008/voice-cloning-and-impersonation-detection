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

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"}
# .webm added: browsers' native MediaRecorder API (used by the dashboard's
# "Speak Now" mic recording via streamlit-mic-recorder) outputs webm/Opus,
# not wav, directly. This was a real bug found during testing — real mic
# recordings were being rejected with HTTP 400 before ever reaching ffmpeg
# conversion, because .webm wasn't in this set. ffmpeg decodes webm/Opus
# natively (already confirmed present in this project's ffmpeg build).
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB, generous for a demo

# Every uploaded/recorded audio is normalized to this format before ANY
# model (AASIST, ECAPA, transcription) ever sees it — see
# app/services/audio_conversion.py.
CONVERTED_AUDIO_SAMPLE_RATE = 16000
CONVERTED_AUDIO_CHANNELS = 1  # mono

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
# Automatic metadata extraction (replaces manual amount/urgency/known-contact
# entry in the dashboard — see app/services/entity_extraction.py,
# app/services/urgency_detector.py, and app/routers/analyze.py)
# ---------------------------------------------------------------------------
# "real" (default when AI_BACKEND=real): transcribe with faster-whisper.
# "mock": deterministic stand-in transcript (same filename-hint pattern as
#         the rest of the mock stack), so the whole pipeline is testable
#         and demoable without downloading a transcription model.
TRANSCRIPTION_BACKEND = os.environ.get("VISL_TRANSCRIPTION_BACKEND", AI_BACKEND).strip().lower()
WHISPER_MODEL_SIZE = os.environ.get("VISL_WHISPER_MODEL_SIZE", "tiny")

# Speaker similarity at/above this is treated as "recognized speaker" ->
# caller_known=True, when not explicitly provided (Task 5: automatic known-
# contact detection). Deliberately equal to the identity risk formula's own
# implicit midpoint is NOT required — this is a separate, tunable threshold.
KNOWN_CONTACT_SIMILARITY_THRESHOLD = 0.75

# Keyword lists for the urgency NLP detector (app/services/urgency_detector.py).
# Checked case-insensitively as substrings of the transcript.
HIGH_URGENCY_KEYWORDS = [
    "immediately", "urgent", "urgently", "right now", "don't tell anyone",
    "do not tell anyone", "emergency", "quickly", "hurry", "asap",
    "before it's too late", "act now", "final warning",
]
MEDIUM_URGENCY_KEYWORDS = [
    "soon", "today", "as soon as possible", "please hurry", "time sensitive",
    "before end of day", "shortly",
]

# ---------------------------------------------------------------------------
# analysis.db — persistent log of every analyzed call (Task 6)
# ---------------------------------------------------------------------------
ANALYSIS_DB_PATH = os.path.join(DATA_DIR, "analysis.db")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ALLOW_ORIGINS = ["*"]  # hackathon-friendly; tighten before any real deployment
