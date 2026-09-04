from models.spoof_detector import get_spoof_score
from models.speaker_verifier import enroll_speaker, get_similarity

# STEP 1 — Enroll the genuine user
user_id = enroll_speaker(
    name="Rahul Sharma",
    role="CEO",
    audio_path="audio/test.wav"
)

print(f"User ID : {user_id}")

# STEP 2 — Speaker verification
similarity = get_similarity(
    "audio/test.wav",
    user_id
)

print(f"Similarity : {similarity:.4f}")

# STEP 3 — Spoof detection
spoof = get_spoof_score("audio/test.wav")

print(f"Spoof Score : {spoof:.4f}")

# STEP 4 — Final AI outputs (Member 1 deliverable)
print("\nReturned Values")
print("-------------------------")
print({
    "spoof_score": round(spoof, 4),
    "speaker_similarity": round(similarity, 4)
})