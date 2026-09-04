from models.spoof_detector import get_spoof_score
from models.speaker_verifier import enroll_speaker, get_similarity

def analyze_voice(audio_path: str, user_id: int):
    return {
        "spoof_score": get_spoof_score(audio_path),
        "speaker_similarity": get_similarity(audio_path, user_id)
    }

def register_user(name: str, role: str, audio_path: str):
    return enroll_speaker(name, role, audio_path)