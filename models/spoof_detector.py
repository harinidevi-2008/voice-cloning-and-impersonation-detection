import os
import sys
import torch
import numpy as np

from utils.preprocess import preprocess

# ------------------------------------------------------------------
# Add the cloned AASIST repository to Python path
# ------------------------------------------------------------------
AASIST_ROOT = os.path.join(os.getcwd(), "checkpoints", "aasist")
sys.path.insert(0, AASIST_ROOT)

from models.AASIST import Model

# ------------------------------------------------------------------
# Official AASIST configuration
# ------------------------------------------------------------------
D_ARGS = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}


class SpoofDetector:

    def __init__(self):

        # Force CPU (AASIST has an unsupported Conv1D op on Apple MPS)
        self.device = torch.device("cpu")

        # Build model
        self.model = Model(D_ARGS).to(self.device)

        # Load pretrained weights
        weight_path = os.path.join(
            AASIST_ROOT,
            "models",
            "weights",
            "AASIST.pth"
        )

        state = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str):

        # Preprocess audio
        waveform, _ = preprocess(audio_path)

        # Convert to tensor
        x = torch.tensor(
            waveform,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Forward pass. AASIST returns (hidden, logits), and the spoof class
        # is stored in the second column of the 2-class output.
        output = self.model(x)
        logits = output[1] if isinstance(output, tuple) else output

        if logits.dim() > 1:
            logit = logits[0, 1]
        else:
            logit = logits[0]

        # Convert logits → probability
        score = torch.sigmoid(logit).item()

        return float(score)


# ---------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------
_detector = SpoofDetector()


def get_spoof_score(audio_path: str) -> float:
    """
    Public function used by Member 2's backend.

    Returns:
        float between 0.0 and 1.0
        Higher = more likely AI-generated
    """
    return _detector.predict(audio_path)