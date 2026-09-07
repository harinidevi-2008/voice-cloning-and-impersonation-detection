import logging
import os
import time
import tempfile
from typing import Optional
import soundfile as sf
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
from app.services import prosody_analyzer
from app.services.preventive_actions import generate_preventive_actions
from app.services.financial_entity_extractor import amount_to_inr, extract_financial_entities
from app.services.urgency_detector import detect_urgency_detailed
from app.services.ai_models.exceptions import (
    AudioDecodeError,
    AudioTooShortError,
    SpeakerEmbeddingMissingError,
)
from app.config import LIVE_ANALYSIS_WINDOW_SECONDS, RETAIN_RAW_AUDIO
from app.services.ai_models.embedding_store import has_valid_embedding, init_db as init_embedding_db
from app.config import AI_BACKEND, URGENCY_RISK_MAP, KNOWN_CONTACT_SIMILARITY_THRESHOLD

router = APIRouter()
logger = logging.getLogger("visl.analyze")


def _discard_analysis_audio(path: Optional[str], *, force: bool = False) -> None:
    """Remove transient normalized call audio unless explicitly retained."""
    # A live-window file is always ephemeral, even when a deployment has
    # deliberately opted to retain authoritative final-call recordings.
    live_window = bool(path and os.path.basename(path).startswith("live_window_"))
    if (RETAIN_RAW_AUDIO and not live_window and not force) or not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        logger.exception("Failed to remove transient analysis audio %s", path)


def _recent_live_window(audio_path: str) -> str:
    """Return a bounded recent WAV window for temporary live inference.

    The upload has already been normalized to 16 kHz mono PCM. Keeping the
    most recent 12 seconds gives each unchanged model enough speech context,
    including AASIST's fixed ~4-second input, without making each later live
    pass slower than the last. This file is deleted by the caller exactly as
    any other transient analysis audio is.
    """
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    max_samples = int(sample_rate * LIVE_ANALYSIS_WINDOW_SECONDS)
    if max_samples <= 0:
        max_samples = samples.shape[0]
    with tempfile.NamedTemporaryFile(
        prefix="live_window_", suffix=".wav", dir=os.path.dirname(audio_path), delete=False
    ) as temporary:
        window_path = temporary.name
    sf.write(window_path, samples[-max_samples:], sample_rate, subtype="PCM_16")
    return window_path


def _stage_error(stage: str, exc: Exception, status_code: int = 500) -> HTTPException:
    """Log diagnostics server-side while returning a safe stage-specific error."""
    logger.exception("%s failed", stage, exc_info=exc)
    return HTTPException(status_code=status_code, detail=f"{stage} failed.")


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


async def _run_analysis(
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
    *,
    persist: bool,
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
                detail="Claimed speaker not found",
            )
        # Check the secondary embedding store before model inference.  The
        # profile DB and embedding DB are intentionally separate, so an old
        # or interrupted enrollment can leave a valid profile without a
        # voiceprint.  That is a client-actionable 422, never a 500.
        init_embedding_db()
        if not has_valid_embedding(claimed_user_id):
            logger.info(
                "analyze: claimed_user_id=%s speaker_found=%s embedding_loaded=%s",
                claimed_user_id, True, False,
            )
            raise HTTPException(status_code=422, detail="Speaker has no enrolled voice sample")

    # --- save + normalize audio ---
    raw_path = save_upload_file(audio_file, prefix="analyze")
    saved_path = None
    try:
        saved_path = convert_to_standard_wav(raw_path)
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - model integration errors must be actionable
        raise _stage_error("Audio conversion", exc) from exc
    finally:
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except OSError:
                logger.exception("Failed to remove pre-conversion upload %s", raw_path)

    if not persist:
        try:
            live_window_path = _recent_live_window(saved_path)
        except Exception as exc:  # noqa: BLE001 - normalized input should be readable
            _discard_analysis_audio(saved_path, force=True)
            raise _stage_error("Live audio window preparation", exc) from exc
        if live_window_path != saved_path:
            _discard_analysis_audio(saved_path, force=True)
            saved_path = live_window_path

    # --- Member 1 interface calls (mock or real, per AI_BACKEND) ---
    ai_start = time.perf_counter()
    try:
        spoof_assessment = ai_service.get_spoof_assessment(saved_path)
        spoof_score = float(spoof_assessment["spoof_score"])
    except AudioTooShortError as exc:
        _discard_analysis_audio(saved_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioDecodeError as exc:
        _discard_analysis_audio(saved_path)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded audio: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - model failures must not become raw 500s
        _discard_analysis_audio(saved_path)
        raise _stage_error("AASIST spoof detection", exc) from exc

    speaker_similarity = None
    if claimed_user_id is not None:
        try:
            speaker_similarity = ai_service.get_similarity(saved_path, claimed_user_id)
        except AudioTooShortError as exc:
            _discard_analysis_audio(saved_path)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (AudioDecodeError, SpeakerEmbeddingMissingError) as exc:
            _discard_analysis_audio(saved_path)
            raise HTTPException(status_code=422, detail="ECAPA speaker verification failed.") from exc
        except Exception as exc:  # noqa: BLE001
            _discard_analysis_audio(saved_path)
            raise _stage_error("ECAPA speaker verification", exc) from exc
    ai_elapsed_ms = (time.perf_counter() - ai_start) * 1000

    # --- transcription + auto-extraction (Tasks 3, 4, 5) ---
    try:
        # Pass the original filename as the mock heuristic's hint. Real
        # Whisper inference always receives the normalized 16 kHz mono WAV.
        transcription = transcription_service.transcribe_detailed(
            saved_path, filename_hint=audio_file.filename
        )
    except Exception as exc:  # transcription is required evidence, never silently discarded
        _discard_analysis_audio(saved_path)
        raise _stage_error("Transcription", exc) from exc

    transcript = transcription["transcript"]
    if not transcript or not transcript.strip():
        transcript = None

    financial_entities = extract_financial_entities(transcript)
    primary_amount = financial_entities["primary"]
    detected_amount = primary_amount["amount"] if primary_amount else None
    urgency_details = detect_urgency_detailed(transcript) if transcript else {
        "urgency": "low", "confidence": 0.4, "matched_keywords": [],
    }
    detected_urgency = urgency_details["urgency"]

    try:
        prosody = prosody_analyzer.analyze_prosody(saved_path)
    except Exception as exc:  # noqa: BLE001
        _discard_analysis_audio(saved_path)
        raise _stage_error("Prosody analysis", exc) from exc

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

    # Apply INR thresholds only to INR or an explicitly configured FX basis;
    # unknown/unconfigured currencies contribute no false monetary risk.
    risk_amount = amount_to_inr(
        detected_amount, primary_amount["currency"] if primary_amount else None
    )
    final_transaction_value = transaction_value if transaction_value is not None else risk_amount
    final_urgency = urgency.strip().lower() if urgency is not None else detected_urgency

    # --- context risk ---
    try:
        context_result = context_engine.compute_context_risk(
            caller_known=known_contact,
            transaction_value=final_transaction_value,
            urgency=final_urgency,
        )
    except Exception as exc:  # noqa: BLE001
        _discard_analysis_audio(saved_path)
        raise _stage_error("Context/risk fusion", exc) from exc
    context_risk = context_result["context_risk"]

    # --- fusion ---
    try:
        impersonation_risk = risk_engine.compute_weighted_risk(
            spoof_score=spoof_score,
            speaker_similarity=speaker_similarity,
            context_risk=context_risk,
            prosody_risk=prosody["prosody_risk"],
            prosody_confidence=prosody["confidence"],
        )
        verdict = risk_engine.get_verdict(impersonation_risk)
    except Exception as exc:  # noqa: BLE001
        _discard_analysis_audio(saved_path)
        raise _stage_error("Context/risk fusion", exc) from exc

    preventive_actions = generate_preventive_actions(
        verdict=verdict,
        spoof_score=spoof_score,
        amount=final_transaction_value,
        urgency=final_urgency,
    )

    # Intermediate analysis deliberately reuses every model and risk signal
    # above, but never creates a call-history record. Only a stopped call's
    # complete recording is authoritative and persisted.
    call_id = None
    if persist:
        try:
            call_id = analysis_db.save_analysis(
                transcript=transcript,
                spoof_score=spoof_score,
                similarity=speaker_similarity,
                amount=detected_amount,
                urgency=final_urgency,
                risk=verdict,
                speaker_name=(claimed_user["name"] if claimed_user else None),
                speaker_user_id=claimed_user_id,
                preventive_actions=preventive_actions,
                currency=primary_amount["currency"] if primary_amount else None,
                display_amount=primary_amount["display_amount"] if primary_amount else None,
            )
        except Exception as exc:  # noqa: BLE001
            _discard_analysis_audio(saved_path)
            raise _stage_error("Database/history persistence", exc) from exc

    total_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{total_elapsed_ms:.1f}"
    logger.info(
        "analyze: backend=%s claimed_user_id=%s speaker_found=%s embedding_loaded=%s "
        "similarity=%s raw_spoof_evidence=%s logits=(%s,%s) duration=%s "
        "spoof_score=%.4f final_risk=%.4f verdict=%s amount=%s urgency=%s",
        AI_BACKEND, claimed_user_id, claimed_user is not None,
        speaker_similarity is not None, speaker_similarity,
        spoof_assessment.get("raw_spoof_evidence"), spoof_assessment.get("logit_spoof"),
        spoof_assessment.get("logit_bonafide"), spoof_assessment.get("normalized_duration_seconds"), spoof_score,
        impersonation_risk, verdict, final_transaction_value, final_urgency,
    )

    result = AnalyzeResponse(
        spoof_score=spoof_score,
        speaker_similarity=speaker_similarity,
        context_risk=context_risk,
        impersonation_risk=impersonation_risk,
        verdict=verdict,
        transcript=transcript,
        detected_amount=detected_amount,
        detected_currency=primary_amount["currency"] if primary_amount else None,
        currency_symbol=primary_amount["currency_symbol"] if primary_amount else None,
        display_amount=primary_amount["display_amount"] if primary_amount else None,
        amount_confidence=primary_amount["confidence"] if primary_amount else None,
        amount_source_text=primary_amount["source_text"] if primary_amount else None,
        detected_duration=financial_entities["durations"][0] if financial_entities["durations"] else None,
        detected_urgency=detected_urgency,
        urgency_confidence=urgency_details["confidence"],
        urgency_keywords=urgency_details["matched_keywords"],
        known_contact=known_contact,
        speaker_status=risk_engine.classify_speaker_similarity(speaker_similarity),
        spoof_category=risk_engine.classify_spoof_score(spoof_score),
        spoof_label=risk_engine.classify_spoof_score(spoof_score),
        detected_language=transcription.get("detected_language"),
        language_probability=transcription.get("language_probability"),
        selected_language=transcription.get("selected_language"),
        language_detection_method=transcription.get("language_detection_method"),
        transcription_model=transcription.get("model_size"),
        transcription_segments=transcription.get("segments") or [],
        prosody_risk=prosody["prosody_risk"],
        prosody_confidence=prosody["confidence"],
        recommended_action=risk_engine.get_recommended_action(impersonation_risk),
        preventive_actions=preventive_actions,
        call_id=call_id,
    )
    _discard_analysis_audio(saved_path)
    return result


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_call(
    response: Response,
    audio_file: UploadFile = File(..., description="Audio sample from the call/transaction"),
    claimed_user_id: Optional[int] = Form(None, description="Enrolled user_id the caller claims to be, or omit/null if unknown"),
    transaction_value: Optional[float] = Form(None, ge=0),
    urgency: Optional[str] = Form(None),
    caller_known: Optional[str] = Form(None),
):
    """Analyze a complete call and persist its one authoritative result."""
    return await _run_analysis(
        response, audio_file, claimed_user_id, transaction_value, urgency, caller_known, persist=True
    )


@router.post("/analyze/intermediate", response_model=AnalyzeResponse)
async def analyze_intermediate_call(
    response: Response,
    audio_file: UploadFile = File(..., description="Accumulated in-progress call audio"),
    claimed_user_id: Optional[int] = Form(None, description="Optional enrolled caller identity"),
):
    """Analyze in-progress audio without retaining audio, transcripts, or history."""
    return await _run_analysis(
        response, audio_file, claimed_user_id, None, None, None, persist=False
    )
