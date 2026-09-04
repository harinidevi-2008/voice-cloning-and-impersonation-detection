"""
check_real_ai_integration.py
=============================
Run this on a machine that has requirements-real-ai.txt installed, to
verify the AI_BACKEND=real integration works end-to-end BEFORE relying on
it through the API/dashboard. Deliberately NOT named test_*.py — pytest's
default discovery would otherwise sweep it up, and its module-level
`os.environ["VISL_AI_BACKEND"] = "real"` would leak into the mock-mode test
suite and break it (this happened once during development; pytest.ini also
scopes discovery to tests/ as a second layer of defense).

    python check_real_ai_integration.py

It exercises exactly the same code path app/routers/enroll.py and
app/routers/analyze.py use: enroll a speaker, verify get_similarity finds
the right embedding (the ID-sync fix), and run real spoof detection.

Expects a real (not synthetic) short WAV/MP3 file. If you don't have one
handy, record a few seconds of yourself speaking and save it as
sample.wav in this directory, or pass a path as the first argument:

    python check_real_ai_integration.py path/to/your_voice.wav
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force real backend for this script regardless of the environment's
# current VISL_AI_BACKEND setting.
os.environ["VISL_AI_BACKEND"] = "real"


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    if not os.path.exists(audio_path):
        print(f"ERROR: audio file not found: {audio_path}")
        print("Record a few seconds of speech and save it there, or pass a path:")
        print("    python check_real_ai_integration.py path/to/your_voice.wav")
        sys.exit(1)

    print(f"Using audio file: {audio_path}\n")

    print("[1/4] Loading real_ai_service (this loads AASIST + downloads/loads")
    print("      ECAPA-TDNN on first run — may take a minute)...")
    from app.services import real_ai_service

    print("\n[2/4] Enrolling a test speaker at user_id=999 (enroll_speaker_at,")
    print("      the same function app/routers/enroll.py calls)...")
    returned_id = real_ai_service.enroll_speaker_at(
        user_id=999, name="Integration Test User", role="test", audio_path=audio_path
    )
    assert returned_id == 999, f"expected 999 back, got {returned_id}"
    print(f"      OK — embedding stored at user_id={returned_id}")

    print("\n[3/4] Checking get_similarity() finds that exact embedding again")
    print("      (this is the ID-sync fix — would fail before it)...")
    similarity = real_ai_service.get_similarity(audio_path, 999)
    print(f"      Similarity of the SAME audio to itself: {similarity:.4f}")
    if similarity < 0.9:
        print("      WARNING: expected this to be close to 1.0 for identical audio.")
    else:
        print("      OK — high self-similarity, as expected.")

    print("\n[4/4] Running real spoof detection (AASIST)...")
    spoof_score = real_ai_service.get_spoof_score(audio_path)
    print(f"      Spoof score: {spoof_score:.4f}  (0 = likely genuine, 1 = likely AI-generated)")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED")
    print("=" * 60)
    print("\nNext step: run the full backend with the real models:")
    print("    VISL_AI_BACKEND=real uvicorn app.main:app --reload --port 8000")
    print("(Windows PowerShell: $env:VISL_AI_BACKEND=\"real\"; uvicorn app.main:app --reload --port 8000)")


if __name__ == "__main__":
    main()
