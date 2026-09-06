import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ALLOW_ORIGINS, AI_BACKEND
from app.db.database import init_db
from app.db import analysis_db
from app.routers import enroll, users, analyze, analysis

# Without this, INFO-level logs (e.g. app/routers/analyze.py's per-request
# latency line) are silently dropped — Python's logging module only emits
# WARNING+ by default when nothing has configured a handler. This is what
# makes the "measure end-to-end latency" logging actually show up in the
# console uvicorn is run from.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    analysis_db.init_db()
    yield


app = FastAPI(
    title="Voice Integrity Security Layer — Backend",
    description=(
        "Near-real-time Voice Integrity Security Layer prototype: AASIST, "
        "ECAPA-TDNN, Faster-Whisper, prosody, context, and explainable risk fusion."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "voice-integrity-security-layer-backend",
        "status": "ok",
        "endpoints": ["/enroll", "/users", "/analyze", "/docs"],
        # Not part of the Section 3 interface contract (that's /enroll,
        # /users, /analyze only) — added so the dashboard can show the
        # ACTUAL configured backend instead of a hardcoded guess. See
        # dashboard/streamlit_app.py's check_backend_health().
        "ai_backend": AI_BACKEND,
    }


@app.get("/health")
async def health():
    """Non-sensitive readiness status for demo-day checks.

    This endpoint reports whether each lazy model has been initialized; it
    never triggers a download or model load by itself.
    """
    components = {"aasist": False, "ecapa": False, "whisper": False, "prosody": True}
    if AI_BACKEND == "real":
        try:
            from app.services.ai_models import spoof_detector, speaker_verifier
            from app.services import transcription_service
            components.update({
                "aasist": spoof_detector._detector is not None,
                "ecapa": speaker_verifier._classifier is not None,
                "whisper": transcription_service._whisper_model is not None,
            })
        except Exception:
            components["prosody"] = False
    else:
        components.update({"aasist": None, "ecapa": None, "whisper": None})
    return {"status": "ok", "ai_backend": AI_BACKEND, "components": components}


app.include_router(enroll.router, tags=["enrollment"])
app.include_router(users.router, tags=["users"])
app.include_router(analyze.router, tags=["analysis"])
app.include_router(analysis.router, tags=["analysis-history"])
