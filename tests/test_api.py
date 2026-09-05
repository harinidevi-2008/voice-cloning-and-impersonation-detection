import io
import os
import sys

# Ensure tests can run isolated against a scratch DB/audio dir.
os.environ.setdefault("PYTHONPATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import init_db
from app.db import analysis_db
from app.config import AI_BACKEND, AUDIO_UPLOAD_DIR
from app.services.ai_models.exceptions import AudioDecodeError

# TestClient's plain instantiation doesn't reliably fire FastAPI's startup
# event across versions, so make sure both DB schemas exist before any test
# runs (mirrors what actually happens when uvicorn starts the real app).
init_db()
analysis_db.init_db()

client = TestClient(app)


def _fake_wav_bytes() -> bytes:
    # Genuinely valid (if trivial) WAV audio — required now that every
    # upload goes through real ffmpeg conversion (Task 1:
    # app/services/audio_conversion.py) regardless of AI_BACKEND. A plain
    # byte string like b"RIFF....WAVEfmt " sailed through the old pipeline
    # because nothing actually decoded audio content in mock mode; now
    # something always does, so tests need real (if minimal) audio.
    # Uses the stdlib `wave` module rather than numpy/soundfile so the
    # test suite doesn't gain a new dependency for this.
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
    return buf.getvalue()


def test_enroll_and_list_users():
    files = {"audio_file": ("genuine_alice.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"name": "Alice", "role": "customer"}
    resp = client.post("/enroll", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "user_id" in body
    user_id = body["user_id"]

    resp_users = client.get("/users")
    assert resp_users.status_code == 200
    users = resp_users.json()
    assert any(u["user_id"] == user_id and u["name"] == "Alice" for u in users)


def test_analyze_without_claimed_user():
    files = {"audio_file": ("clone_bob.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {
        "transaction_value": "10000",
        "urgency": "high",
        "caller_known": "false",
    }
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["speaker_similarity"] is None
    assert 0.0 <= body["spoof_score"] <= 1.0
    assert 0.0 <= body["context_risk"] <= 1.0
    assert 0.0 <= body["impersonation_risk"] <= 1.0
    assert body["verdict"] in {
        "HIGH_RISK_LIKELY_IMPERSONATION",
        "MEDIUM_RISK_MANUAL_REVIEW",
        "LOW_RISK_LIKELY_GENUINE",
    }


def test_analyze_returns_latency_header():
    # Day-2 "measure end-to-end latency" requirement: exposed as a response
    # header (X-Processing-Time-Ms) rather than a JSON field, so the frozen
    # Section 3 response contract stays untouched. See app/routers/analyze.py.
    files = {"audio_file": ("genuine_latency.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"transaction_value": "500", "urgency": "low", "caller_known": "true"}
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 200
    assert "x-processing-time-ms" in {k.lower() for k in resp.headers.keys()}
    elapsed = float(resp.headers["x-processing-time-ms"])
    assert elapsed >= 0.0
    # The original Section 3 contract (5 fields) must always be present.
    # Additional fields (transcript, detected_amount, detected_urgency,
    # known_contact, call_id) are a deliberate, additive, backward-
    # compatible extension for the dashboard refactor — see
    # app/schemas.py's AnalyzeResponse docstring — so this checks for a
    # superset, not an exact match.
    body_keys = set(resp.json().keys())
    original_contract_fields = {
        "spoof_score", "speaker_similarity", "context_risk",
        "impersonation_risk", "verdict",
    }
    assert original_contract_fields.issubset(body_keys)


def test_analyze_with_unknown_claimed_user_returns_404():
    files = {"audio_file": ("genuine_carol.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {
        "claimed_user_id": "999999",
        "transaction_value": "500",
        "urgency": "low",
        "caller_known": "true",
    }
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 404


def test_analyze_rejects_bad_extension():
    files = {"audio_file": ("notaudio.txt", io.BytesIO(b"hello"), "text/plain")}
    data = {"transaction_value": "500", "urgency": "low", "caller_known": "true"}
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 400


def test_analyze_rejects_invalid_urgency():
    files = {"audio_file": ("genuine_dave.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"transaction_value": "500", "urgency": "asap!!", "caller_known": "true"}
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 400
    assert "urgency" in resp.json()["detail"]


def test_analyze_rejects_invalid_caller_known():
    files = {"audio_file": ("genuine_dave.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"transaction_value": "500", "urgency": "low", "caller_known": "maybe"}
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 400


def test_analyze_rejects_negative_transaction_value():
    files = {"audio_file": ("genuine_dave.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"transaction_value": "-500", "urgency": "low", "caller_known": "true"}
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 422  # Pydantic Form(ge=0) validation


def test_analyze_rejects_non_integer_claimed_user_id():
    files = {"audio_file": ("genuine_dave.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {
        "claimed_user_id": "not-a-number",
        "transaction_value": "500",
        "urgency": "low",
        "caller_known": "true",
    }
    resp = client.post("/analyze", data=data, files=files)
    assert resp.status_code == 422


def test_enroll_rejects_missing_audio_file():
    data = {"name": "Eve", "role": "customer"}
    resp = client.post("/enroll", data=data)
    assert resp.status_code == 422


def test_enroll_rejects_blank_name():
    files = {"audio_file": ("genuine_eve.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"name": "   ", "role": "customer"}
    resp = client.post("/enroll", data=data, files=files)
    assert resp.status_code == 400


def test_root_endpoint_reports_actual_ai_backend():
    # Task 6: the dashboard must show the REAL configured backend, not a
    # hardcoded guess. This confirms the backend side of that contract:
    # GET / must report the actual app.config.AI_BACKEND value.
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "ai_backend" in body
    assert body["ai_backend"] == AI_BACKEND


def test_enrollment_rolls_back_on_ai_service_failure(monkeypatch):
    # Task 3: if voiceprint registration fails after the user row was
    # already created, the user must NOT be left in the database as a
    # "ghost user" with no embedding, and the uploaded file must be cleaned
    # up. Simulated here via the mock backend so it runs without real AI
    # deps — app/routers/enroll.py wraps both the mock and real branches in
    # the same try/except/rollback, so this exercises the same code path.
    import app.services.mock_ai_service as mock_ai_service

    def _boom(name, role, audio_path):
        raise RuntimeError("simulated ECAPA failure")

    monkeypatch.setattr(mock_ai_service, "enroll_speaker", _boom)

    files_before = set(os.listdir(AUDIO_UPLOAD_DIR)) if os.path.isdir(AUDIO_UPLOAD_DIR) else set()
    users_before = client.get("/users").json()

    files = {"audio_file": ("genuine_ghost.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"name": "Ghost User", "role": "customer"}
    resp = client.post("/enroll", data=data, files=files)

    assert resp.status_code == 500

    users_after = client.get("/users").json()
    assert not any(u["name"] == "Ghost User" for u in users_after), (
        "Ghost user found in database after a simulated enrollment failure — rollback did not work"
    )
    assert len(users_after) == len(users_before), "User count changed despite the enrollment failing"

    files_after = set(os.listdir(AUDIO_UPLOAD_DIR)) if os.path.isdir(AUDIO_UPLOAD_DIR) else set()
    assert files_after == files_before, (
        "Uploaded audio file was not cleaned up after a simulated enrollment failure"
    )


def test_enrollment_returns_400_not_500_for_audio_decode_error(monkeypatch):
    # Task 4: a corrupt/non-audio file should be a clean 400, not a raw 500,
    # even though this specific failure mode only actually happens with the
    # real AI backend (mock never decodes audio). Simulated here by forcing
    # the mock enrollment call to raise AudioDecodeError directly, to test
    # the ROUTER's exception handling in isolation from whether librosa is
    # installed in this environment.
    import app.services.mock_ai_service as mock_ai_service

    def _bad_audio(name, role, audio_path):
        raise AudioDecodeError("simulated corrupt audio file")

    monkeypatch.setattr(mock_ai_service, "enroll_speaker", _bad_audio)

    users_before = client.get("/users").json()
    files = {"audio_file": ("genuine_corrupt.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"name": "Corrupt Audio User", "role": "customer"}
    resp = client.post("/enroll", data=data, files=files)

    assert resp.status_code == 400
    assert "audio" in resp.json()["detail"].lower()

    users_after = client.get("/users").json()
    assert not any(u["name"] == "Corrupt Audio User" for u in users_after), (
        "Ghost user found after a simulated AudioDecodeError — rollback did not work"
    )


def test_analyze_returns_400_not_500_for_audio_decode_error(monkeypatch):
    # Same as above, but for /analyze's AI-service call path.
    import app.services.ai_service as ai_service

    def _bad_audio(audio_path):
        raise AudioDecodeError("simulated corrupt audio file")

    monkeypatch.setattr(ai_service, "get_spoof_score", _bad_audio)

    files = {"audio_file": ("genuine_corrupt2.wav", io.BytesIO(_fake_wav_bytes()), "audio/wav")}
    data = {"transaction_value": "500", "urgency": "low", "caller_known": "true"}
    resp = client.post("/analyze", data=data, files=files)

    assert resp.status_code == 400
    assert "audio" in resp.json()["detail"].lower()
