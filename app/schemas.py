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
    spoof_score: float
    speaker_similarity: Optional[float] = None
    context_risk: float
    impersonation_risk: float
    verdict: str
