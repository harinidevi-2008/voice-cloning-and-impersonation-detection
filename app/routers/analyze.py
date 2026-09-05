import logging
import os
import time
from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Response

from app.schemas import AnalyzeResponse
from app.storage.audio_store import save_upload_file
from app.services.audio_conversion import convert_to_standard_wav
from app.db import crud
from app.db import analysis_db
from app.services import ai_service
from app.services import context_engine
from app.services import risk_engine
from app.services import transcription_service
from app.services.entity_extraction import extract_amount
from app.services.urgency_detector import detect_urgency_detailed
from app.services.ai_models.exceptions import AudioDecodeError
from app.config import AI_BACKEND, URGENCY_RISK_MAP, KNOWN_CONTACT_SIMILARITY_THRESHOLD

router = APIRouter()
logger = logging.getLogger("visl.analyze")


def _parse_bool(value: str, field_name: str) -> bool:
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}
    normalized = str(value).strip().lower()
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise HTTPException(
        status_code=400,
        detail=f"'{field_name}' must be a boolean-like value (true/false), got '{value}'.",
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_call(
    response: Response,
    audio_file: UploadFile = File(..., description="Audio sample from the call/transaction"),
    claimed_user_id: Optional[int] = Form(
        None, description="Enrolled user_id the caller claims to be, or omit/null if unknown"
    ),
    transaction_value: Optional[float] = Form(
        None, ge=0,
        description="Monetary value of the transaction. Omit to auto-detect from speech "
                    "(dashboard refactor) — see app/services/entity_extraction.py.",
    ),
    urgency: Optional[str] = Form(
        None,
        description=f"One of: {list(URGENCY_RISK_MAP.keys())}. Omit to auto-detect from "
                    "speech — see app/services/urgency_detector.py.",
    ),
    caller_known: Optional[str] = Form(
        None,
        description="true/false. Omit to auto-derive from speaker similarity vs. "
                    "KNOWN_CONTACT_SIMILARITY_THRESHOLD (requires claimed_user_id).",
    ),
):
    """
    Runs the full pipeline for one call/transaction:
      1. Save + normalize uploaded audio to mono 16kHz WAV (accepts WAV/
         MP3/M4A/AAC/FLAC/OGG/MP4 — app/services/audio_conversion.py).
      2. Get spoof_score from the AI detection model (mock or real,
         selected by AI_BACKEND — see app/services/ai_service.py).
      3. If claimed_user_id is given, verify it exists, then get
         speaker_similarity from the speaker verification model.
      4. Transcribe the audio (app/services/transcription_service.py).
      5. For any of transaction_value / urgency / caller_known NOT
         explicitly provided, auto-derive it from the transcript/
         similarity instead — this is what lets the dashboard drop manual
         entry entirely while keeping this endpoint's request shape
         backward compatible (explicit values, if sent, are still
         honored — nothing here breaks an existing caller that still
         sends all three).
      6. Compute context_risk from the (explicit-or-derived) metadata.
      7. Fuse everything into impersonation_risk and a verdict.
      8. Log the call to analysis.db (Task 6).

    RESPONSE CONTRACT: the original 5 fields (spoof_score,
    speaker_similarity, context_risk, impersonation_risk, verdict) are
    unchanged. New fields (transcript, detected_amount, detected_urgency,
    known_contact, call_id) are additive and optional — existing clients
    reading only the original 5 fields are unaffected.

    Latency instrumentation (Day 2 "measure end-to-end latency" task):
    logged server-side and returned as the X-Processing-Time-Ms response
    header, not a JSON field.
    """
    pipeline_start = time.perf_counter()

    # --- validate explicit overrides, if given (unchanged behavior when provided) ---
    caller_known_explicit = None
    if caller_known is not None:
        caller_known_explicit = _parse_bool(caller_known, "caller_known")

    if urgency is not None and urgency.strip().lower() not in URGENCY_RISK_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"'urgency' must be one of {list(URGENCY_RISK_MAP.keys())}, got '{urgency}'.",
        )

    # --- validate claimed_user_id, if provided ---
    claimed_user = None
    if claimed_user_id is not None:
        claimed_user = crud.get_user(claimed_user_id)
        if claimed_user is None:
            raise HTTPException(
                status_code=404,
                detail=f"claimed_user_id {claimed_user_id} is not enrolled.",
            )

    # --- save + normalize audio ---
    raw_path = save_upload_file(audio_file, prefix="analyze")
    try:
        saved_path = convert_to_standard_wav(raw_path)
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio: {exc}",
        )
    finally:
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except OSError:
                logger.exception("Failed to remove pre-conversion upload %s", raw_path)

    # --- Member 1 interface calls (mock or real, per AI_BACKEND) ---
    ai_start = time.perf_counter()
    try:
        spoof_score = ai_service.get_spoof_score(saved_path)

        speaker_similarity = None
        if claimed_user_id is not None:
            speaker_similarity = ai_service.get_similarity(saved_path, claimed_user_id)
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio: {exc}",
        )
    ai_elapsed_ms = (time.perf_counter() - ai_start) * 1000

    # --- transcription + auto-extraction (Tasks 3, 4, 5) ---
    try:
        # Pass the ORIGINAL uploaded filename as the mock heuristic's hint
        # (not saved_path, whose UUID could otherwise confuse the mock's
        # digit-sequence detection — see transcription_service.py's
        # module docstring for the bug this fixes).
        transcript = transcription_service.transcribe(saved_path, filename_hint=audio_file.filename)
    except Exception:  # noqa: BLE001 — transcription failure must not sink the whole analysis
        logger.exception("Transcription failed for %s; continuing without transcript", saved_path)
        transcript = None

    detected_amount = extract_amount(transcript) if transcript else None
    urgency_details = detect_urgency_detailed(transcript) if transcript else {
        "urgency": "low", "confidence": 0.4, "matched_keywords": [],
    }
    detected_urgency = urgency_details["urgency"]

    # Known-contact: explicit value wins if given; otherwise derive from
    # similarity vs threshold (Task 5). An unclaimed/unverifiable identity
    # defaults to False (not known) — consistent with this project's
    # existing "unverifiable is a risk signal, not a safe default"
    # principle (see risk_engine.compute_identity_mismatch_risk).
    if caller_known_explicit is not None:
        known_contact = caller_known_explicit
    elif speaker_similarity is not None:
        known_contact = speaker_similarity >= KNOWN_CONTACT_SIMILARITY_THRESHOLD
    else:
        known_contact = False

    final_transaction_value = transaction_value if transaction_value is not None else (detected_amount or 0.0)
    final_urgency = urgency.strip().lower() if urgency is not None else detected_urgency

    # --- context risk ---
    context_result = context_engine.compute_context_risk(
        caller_known=known_contact,
        transaction_value=final_transaction_value,
        urgency=final_urgency,
    )
    context_risk = context_result["context_risk"]

    # --- fusion ---
    identity_mismatch_risk = risk_engine.compute_identity_mismatch_risk(speaker_similarity)
    impersonation_risk = risk_engine.compute_impersonation_risk(
        spoof_score=spoof_score,
        identity_mismatch_risk=identity_mismatch_risk,
        context_risk=context_risk,
    )
    verdict = risk_engine.get_verdict(impersonation_risk)

    # --- persist to call_logs (Task 6) ---
    try:
        call_id = analysis_db.save_analysis(
            transcript=transcript,
            spoof_score=spoof_score,
            similarity=speaker_similarity,
            amount=final_transaction_value,
            urgency=final_urgency,
            risk=verdict,
            speaker_name=(claimed_user["name"] if claimed_user else None),
        )
    except Exception:  # noqa: BLE001 — logging failure must not sink the analysis result
        logger.exception("Failed to persist call record to analysis.db")
        call_id = None

    total_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{total_elapsed_ms:.1f}"
    logger.info(
        "analyze: backend=%s claimed_user_id=%s ai_ms=%.1f total_ms=%.1f verdict=%s "
        "amount=%s urgency=%s known_contact=%s",
        AI_BACKEND, claimed_user_id, ai_elapsed_ms, total_elapsed_ms, verdict,
        final_transaction_value, final_urgency, known_contact,
    )

    return AnalyzeResponse(
        spoof_score=spoof_score,
        speaker_similarity=speaker_similarity,
        context_risk=context_risk,
        impersonation_risk=impersonation_risk,
        verdict=verdict,
        transcript=transcript,
        detected_amount=detected_amount,
        detected_urgency=detected_urgency,
        urgency_confidence=urgency_details["confidence"],
        urgency_keywords=urgency_details["matched_keywords"],
        known_contact=known_contact,
        call_id=call_id,
    )
