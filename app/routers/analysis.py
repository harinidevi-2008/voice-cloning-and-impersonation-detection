from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.db import analysis_db

router = APIRouter()


class CallLogOut(BaseModel):
    call_id: str
    timestamp: str
    speaker_name: Optional[str] = None
    transcript: Optional[str] = None
    amount: Optional[float] = None
    urgency: Optional[str] = None
    spoof_score: Optional[float] = None
    similarity: Optional[float] = None
    risk: Optional[str] = None


@router.get("/analysis/recent", response_model=List[CallLogOut])
async def get_recent_analyses(limit: int = 10):
    """
    Task 6: "Recent Analyses" dashboard section — the latest N analyzed
    calls from call_logs (app/db/analysis_db.py). Not part of the original
    Section 3 interface contract (/enroll, /users, /analyze) — this is a
    new, additive endpoint, added rather than having the dashboard read
    the SQLite file directly, to keep the dashboard a pure HTTP client
    with no direct backend/filesystem access (existing project principle).
    """
    return analysis_db.list_recent_analyses(limit=limit)
