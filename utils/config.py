from utils.config import SPOOF_THRESHOLD

if spoof_score >= SPOOF_THRESHOLD:
    verdict = "AI Generated"
else:
    verdict = "Genuine"