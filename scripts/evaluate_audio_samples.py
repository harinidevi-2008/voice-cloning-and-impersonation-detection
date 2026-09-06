"""Evaluate audio folders with the configured VISL pipeline.

Layout:
    evaluation/genuine/*.wav
    evaluation/spoof/*.wav

This intentionally reports measurements only; it does not claim a quality
level until the supplied labelled samples have actually been evaluated.
"""

import argparse
import csv
import time
from pathlib import Path

from app.services import ai_service, risk_engine
from app.services.prosody_analyzer import analyze_prosody


SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


def _metric(rows, threshold=0.5):
    labelled = [r for r in rows if r["label"] in {"genuine", "spoof"}]
    if not labelled:
        return {}
    truth = [r["label"] == "spoof" for r in labelled]
    pred = [r["spoof_score"] >= threshold for r in labelled]
    tp = sum(t and p for t, p in zip(truth, pred))
    tn = sum(not t and not p for t, p in zip(truth, pred))
    fp = sum(not t and p for t, p in zip(truth, pred))
    fn = sum(t and not p for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labelled), "precision": precision,
        "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("evaluation_results.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in SUPPORTED):
        label = path.parent.name.lower()
        started = time.perf_counter()
        try:
            spoof = ai_service.get_spoof_score(str(path))
            prosody = analyze_prosody(str(path))
            risk = risk_engine.compute_impersonation_risk(
                spoof, 0.0, 0.0, prosody.get("prosody_score") or 0.0, prosody.get("confidence") or 0.0
            )
            rows.append({"file": str(path), "label": label, "spoof_score": spoof,
                         "prosody_score": prosody.get("prosody_score"), "final_risk": risk,
                         "verdict": risk_engine.get_verdict(risk),
                         "processing_time_ms": round((time.perf_counter() - started) * 1000, 1), "error": ""})
        except Exception as exc:  # preserve evaluation progress after one bad sample
            rows.append({"file": str(path), "label": label, "spoof_score": "", "prosody_score": "",
                         "final_risk": "", "verdict": "", "processing_time_ms": "", "error": str(exc)})
    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["file", "label", "spoof_score", "prosody_score", "final_risk", "verdict", "processing_time_ms", "error"])
        writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if isinstance(r["spoof_score"], float)]
    print(f"Wrote {len(rows)} rows to {args.csv}")
    for key, value in _metric(valid).items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
