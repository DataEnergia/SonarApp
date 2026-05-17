"""Blind eval on real human recordings — track_c_dataset/audio_real/audios_SONAr2026/"""

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

AUDIO_DIR = Path(__file__).parent / "audio_real" / "audios_SONAr2026"
OUTPUT_DIR = Path(__file__).parent / "audio_real" / "eval_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ground truth built from Testes-audios.docx (blind — Gemma never sees this)
GROUND_TRUTH = [
    {"file": "scam_banco_pix_001.wav.ogg",            "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_filho_acidente_002.wav.ogg",        "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_receita_federal_003.wav.ogg",       "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_suporte_remoto_004.wav.ogg",        "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_cartao_clonado_005.wav.ogg",        "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_whatsapp_empresa_006.wav.ogg",      "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_policia_multa_007.wav.ogg",         "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_bitcoin_008.wav.ogg",               "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_inss_009.wav.ogg",                  "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_codigo_sms_010.wav.ogg",            "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_sequestro_011.wav.ogg",             "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_beneficio_cancelado_012.wav.ogg",   "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_marketplace_013.wav.ogg",           "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_giftcard_014.wav.ogg",              "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "scam_falso_banco_whatsapp_015.wav.ogg",  "type": "scam",       "expected": "danger",     "language": "pt-BR"},
    {"file": "legit_clinica_016.wav.ogg",              "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_entrega_017.wav.ogg",              "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_banco_alerta_018.wav.ogg",         "type": "legitimate", "expected": "suspicious", "language": "pt-BR"},
    {"file": "legit_escola_019.wav.ogg",               "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_suporte_internet_020.wav.ogg",     "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_hotel_021.wav.ogg",                "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_farmacia_022.wav.ogg",             "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_condominio_023.wav.ogg",           "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_empresa_rh_024.wav.ogg",           "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_amigo_025.wav.ogg",                "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_banco_cartao_026.wav.ogg",         "type": "legitimate", "expected": "suspicious", "language": "pt-BR"},
    {"file": "legit_laboratorio_027.wav.ogg",          "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_prefeitura_028.wav.ogg",           "type": "legitimate", "expected": "suspicious", "language": "pt-BR"},
    {"file": "legit_oficina_029.wav.ogg",              "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
    {"file": "legit_streaming_030.wav.ogg",            "type": "legitimate", "expected": "safe",       "language": "pt-BR"},
]


def run_eval() -> None:
    results = []
    total = len(GROUND_TRUTH)

    print(f"\n{'='*60}")
    print(f"  SONAR — Blind Eval ({total} real recordings)")
    print(f"{'='*60}\n")

    for i, sample in enumerate(GROUND_TRUTH, 1):
        audio_path = AUDIO_DIR / sample["file"]
        if not audio_path.exists():
            print(f"[{i:02d}/{total}] MISSING: {sample['file']}")
            results.append({**sample, "predicted": "MISSING", "correct": False, "elapsed_s": 0})
            continue

        stem = audio_path.stem.replace(".wav", "")
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

        correct = _is_correct(sample["type"], sample["expected"], predicted)
        mark = "✓" if correct else "✗"
        print(f"{mark}  predicted={predicted}  expected={sample['expected']}  ({elapsed}s)")

        results.append({**sample, "predicted": predicted, "correct": correct, "elapsed_s": elapsed})

    _print_metrics(results)
    _save_results(results)


def _is_correct(call_type: str, expected: str, predicted: str) -> bool:
    if "ERROR" in str(predicted) or predicted == "MISSING":
        return False
    if call_type == "scam":
        # scam is detected if flagged as suspicious or danger
        return predicted in ("suspicious", "danger")
    else:
        # legitimate is correct if NOT flagged as danger
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

    danger_rate = sum(1 for r in scam if r["predicted"] == "danger") / len(scam) if scam else 0

    print(f"\n{'='*60}")
    print(f"  METRICS  (n={len(results)}: {len(scam)} scam / {len(legit)} legit)")
    print(f"{'='*60}")
    print(f"  Precision   : {precision:.1%}")
    print(f"  Recall      : {recall:.1%}")
    print(f"  F1 Score    : {f1:.1%}")
    print(f"  Accuracy    : {accuracy:.1%}")
    print(f"  Danger rate (scam→danger) : {danger_rate:.1%}")
    print(f"  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
    print(f"{'='*60}\n")


def _save_results(results: list[dict]) -> None:
    out = OUTPUT_DIR / "eval_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {out}\n")


if __name__ == "__main__":
    run_eval()
