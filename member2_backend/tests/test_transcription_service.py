"""
Tests for app/services/transcription_service.py's mock backend, including
a regression test for the filename_hint bug found during development:
searching the post-conversion audio_path (which embeds a UUID) for a
digit sequence could match digits from the UUID noise instead of the
meaningful number in the ORIGINAL uploaded filename.
"""

from app.services.transcription_service import transcribe


def test_mock_transcribe_uses_filename_hint_not_converted_path():
    # Simulates exactly the bug scenario: a converted path whose UUID
    # segment contains a spurious 4+ digit run, alongside the correct
    # original filename passed as filename_hint.
    converted_path = "/data/audio_uploads/analyze_5859205294_urgent_50000_call_converted.wav"
    original_filename = "urgent_50000_call.wav"

    transcript = transcribe(converted_path, filename_hint=original_filename)

    assert "50,000" in transcript
    assert "5,859,205,294" not in transcript  # the bug's symptom


def test_mock_transcribe_falls_back_to_audio_path_if_no_hint_given():
    # Still works (just less reliably) if a caller doesn't pass a hint.
    transcript = transcribe("/data/audio_uploads/genuine_alice.wav")
    assert "fifty thousand" in transcript.lower() or "rupees" in transcript.lower()


def test_mock_transcribe_urgency_flavors():
    high = transcribe("x.wav", filename_hint="urgent_call.wav")
    medium = transcribe("x.wav", filename_hint="clone_call.wav")
    low = transcribe("x.wav", filename_hint="genuine_call.wav")

    assert "immediately" in high.lower() or "don't tell anyone" in high.lower()
    assert "soon" in medium.lower()
    assert "checking in" in low.lower()
