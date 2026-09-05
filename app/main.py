import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import AI_BACKEND, CORS_ALLOW_ORIGINS, DEVELOPMENT_RESET_ON_STARTUP
from app.db.database import init_db
from app.db import analysis_db, crud
from app.db.development_reset import reset_development_data
from app.services.ai_models.embedding_store import has_valid_embedding, init_db as init_embedding_db
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
    init_embedding_db()
    analysis_db.init_db()
    if DEVELOPMENT_RESET_ON_STARTUP:
        reset_development_data()
        logging.getLogger("visl.startup").info("Development runtime data reset complete.")
        yield
        return
    # Legacy profiles may predate voiceprint persistence. Keep them in the
    # database for audit/history, but never expose them as verifiable.
    for user in crud.list_users():
        if has_valid_embedding(user["user_id"]):
            crud.set_embedding_status(user["user_id"], "ready")
        else:
            crud.set_embedding_status(user["user_id"], "incomplete")
            logging.getLogger("visl.startup").warning(
                "User %s has no voice embedding.", user["user_id"]
            )
    yield


app = FastAPI(
    title="Voice Integrity Security Layer — Backend",
    description=(
        "Member 2 (Systems Lead) backend: enrollment, analysis, and the "
        "context/risk fusion engine. AI model calls are mocked until "
        "Member 1's real spoof-detection and speaker-verification models "
        "are integrated (see app/services/mock_ai_service.py)."
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
        "endpoints": ["/enroll", "/enroll/speakers", "/users", "/analyze", "/docs"],
        # Not part of the Section 3 interface contract (that's /enroll,
        # /users, /analyze only) — added so the dashboard can show the
        # ACTUAL configured backend instead of a hardcoded guess. See
        # dashboard/streamlit_app.py's check_backend_health().
        "ai_backend": AI_BACKEND,
    }


app.include_router(enroll.router, tags=["enrollment"])
app.include_router(users.router, tags=["users"])
app.include_router(analyze.router, tags=["analysis"])
app.include_router(analysis.router, tags=["analysis-history"])
