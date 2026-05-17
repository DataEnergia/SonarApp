"""Blind eval on TTS synthetic dataset — track_c_dataset/audio_tts/"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK_A = REPO_ROOT / "track_a_desktop"
if str(TRACK_A) not in sys.path:
    sys.path.insert(0, str(TRACK_A))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import run_pipeline
from shared.schemas.classification import Language

AUDIO_DIR = Path(__file__).parent / "audio_tts"
GT_PATH = Path(__file__).parent / "ground_truth_tts.json"
OUTPUT_DIR = Path(__file__).parent / "eval_outputs_tts"
OUTPUT_DIR.mkdir(exist_ok=True)

gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))
SAMPLES = [
    {
        "file": s["audio_file"],
        "type": s["type"],
        "expected": s["final_risk_level"],
        "language": s["language"],
    }
    for s in gt_data["samples"]
]


def run_eval() -> list[dict]:
    results = []
    total = len(SAMPLES)
    print(f"\n{'='*60}")
    print(f"  SONAR — TTS Eval ({total} synthetic recordings)")
    print(f"{'='*60}\n")

    for i, sample in enumerate(SAMPLES, 1):
        audio_path = AUDIO_DIR / sample["file"]
        if not audio_path.exists():
            print(f"[{i:02d}/{total}] MISSING: {sample['file']}")
            results.append({**sample, "predicted": "MISSING", "correct": False, "elapsed_s": 0})
            continue

        stem = audio_path.stem
        output_path = OUTPUT_DIR / f"{stem}_report.json"
        language = Language(sample["language"])

        print(f"[{i:02d}/{total}] {sample['file']}", end=" ... ", flush=True)
        t0 = time.time()
        try:
            report = run_pipeline(
                audio_path=audio_path,
                language=language,
                output_path=output_path,
                whisper_model_size="small",
                allow_transcript_fallback=False,
            )
            predicted = report.final_state.overall_risk.value
            elapsed = round(time.time() - t0, 1)
        except Exception as exc:
            predicted = f"ERROR: {exc}"
            elapsed = round(time.time() - t0, 1)

        correct = _is_correct(sample["type"], predicted)
        mark = "✓" if correct else "✗"
        print(f"{mark}  predicted={predicted}  expected={sample['expected']}  ({elapsed}s)")
        results.append({**sample, "predicted": predicted, "correct": correct, "elapsed_s": elapsed})

    _print_metrics(results)
    out = OUTPUT_DIR / "eval_tts_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {out}\n")
    return results


def _is_correct(call_type: str, predicted: str) -> bool:
    if "ERROR" in str(predicted) or predicted == "MISSING":
        return False
    if call_type == "scam":
        return predicted in ("suspicious", "danger")
    else:
        return predicted != "danger"


def _print_metrics(results: list[dict]) -> None:
    scam = [r for r in results if r["type"] == "scam"]
    legit = [r for r in results if r["type"] == "legitimate"]
    tp = sum(1 for r in scam if r["predicted"] in ("suspicious", "danger"))
    fn = sum(1 for r in scam if r["predicted"] == "safe")
    tn = sum(1 for r in legit if r["predicted"] != "danger")
    fp = sum(1 for r in legit if r["predicted"] == "danger")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(results) if results else 0
    print(f"\n{'='*60}")
    print(f"  METRICS  (n={len(results)}: {len(scam)} scam / {len(legit)} legit)")
    print(f"{'='*60}")
    print(f"  Precision : {precision:.1%}  Recall : {recall:.1%}")
    print(f"  F1 Score  : {f1:.1%}  Accuracy : {accuracy:.1%}")
    print(f"  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_eval()
