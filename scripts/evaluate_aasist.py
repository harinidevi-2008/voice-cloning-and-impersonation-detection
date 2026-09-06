"""Evaluate the real AASIST checkpoint on labelled local audio folders.

Usage: VISL_AI_BACKEND=real python scripts/evaluate_aasist.py evaluation_data
Expected folders: evaluation_data/genuine and evaluation_data/spoof.
The output is descriptive validation only; it is not production benchmarking.
"""

import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main(root: str) -> None:
    from app.services.ai_models.spoof_detector import get_spoof_assessment

    rows = []
    for expected, directory in (("genuine", "genuine"), ("spoof", "spoof")):
        path = os.path.join(root, directory)
        if not os.path.isdir(path):
            continue
        for filename in sorted(os.listdir(path)):
            if os.path.splitext(filename)[1].lower() not in {".wav", ".flac", ".mp3", ".m4a", ".webm"}:
                continue
            assessment = get_spoof_assessment(os.path.join(path, filename))
            predicted = "spoof" if assessment["spoof_score"] >= 0.5 else "genuine"
            rows.append({"filename": filename, "expected": expected, "predicted": predicted, **assessment})
    if not rows:
        raise SystemExit("No supported audio samples found under genuine/ and spoof/.")

    output = os.path.join(root, "aasist_evaluation.csv")
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    tp = sum(r["expected"] == r["predicted"] == "spoof" for r in rows)
    tn = sum(r["expected"] == r["predicted"] == "genuine" for r in rows)
    fp = sum(r["expected"] == "genuine" and r["predicted"] == "spoof" for r in rows)
    fn = sum(r["expected"] == "spoof" and r["predicted"] == "genuine" for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"threshold=0.50 samples={len(rows)} accuracy={(tp + tn) / len(rows):.3f}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} fpr={fp / (fp + tn) if fp + tn else 0:.3f} fnr={fn / (fn + tp) if fn + tp else 0:.3f}")
    print(f"confusion_matrix [[tn={tn}, fp={fp}], [fn={fn}, tp={tp}]]")
    print(f"wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation_data")
