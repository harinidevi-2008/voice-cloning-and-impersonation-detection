"""
Response schemas. Request bodies for /enroll and /analyze are multipart/form-data
(they carry a file), so they are declared inline in the routers via FastAPI's
Form(...)/File(...) parameters rather than as Pydantic request models.
"""

from typing import Optional
from pydantic import BaseModel


class EnrollResponse(BaseModel):
    user_id: int


class UserOut(BaseModel):
    user_id: int
    name: str
    role: str
    enrolled_at: str


class AnalyzeResponse(BaseModel):
    # --- Original Section 3 interface contract fields (unchanged) ---
    spoof_score: float
    speaker_similarity: Optional[float] = None
    context_risk: float
    impersonation_risk: float
    verdict: str

    # --- Additive fields (dashboard refactor) ---
    # Adding new OPTIONAL fields to a JSON response is backward compatible
    # by REST convention — existing clients that only read the 5 fields
    # above are unaffected. These exist so the dashboard can show what
    # used to be manually typed (transaction amount, urgency, known-
    # contact status) now that it's auto-derived instead — see
    # app/routers/analyze.py for how each is computed, and
    # app/services/entity_extraction.py / urgency_detector.py for the
    # underlying logic.
    transcript: Optional[str] = None
    detected_amount: Optional[float] = None
    detected_urgency: Optional[str] = None
    urgency_confidence: Optional[float] = None
    urgency_keywords: Optional[list] = None
    known_contact: Optional[bool] = None
    call_id: Optional[str] = None

    # Layered-security evidence (all additive to the original response).
    prosody_score: Optional[float] = None
    prosody_confidence: Optional[float] = None
    prosody_features: Optional[dict] = None
    prosody_anomalies: Optional[list[str]] = None
    recommended_action: Optional[str] = None
    risk_factors: Optional[list[str]] = None
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    processing_time_ms: Optional[float] = None
