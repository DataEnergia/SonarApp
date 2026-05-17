# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""One-command desktop demo runner for the Senti prototype."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.schemas.classification import Language  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from preflight import run_preflight  # noqa: E402
from render_report import render_call_report  # noqa: E402


def run_demo(
    audio_path: Path,
    language: Language,
    output_dir: Path,
    model_name: str = "google/gemma-4-e2b",
    whisper_model: str = "tiny",
    prompt_path: Path | None = None,
    require_real_stt: bool = True,
) -> dict[str, Any]:
    """Run preflight, full pipeline, and Markdown rendering for one audio file."""

    prompt_path = prompt_path or REPO_ROOT / "shared" / "prompts" / "classifier_v3.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    preflight_path = output_dir / f"{stem}_preflight.json"
    call_report_path = output_dir / f"{stem}_call_report.json"
    markdown_report_path = output_dir / f"{stem}_call_report.md"
    summary_path = output_dir / f"{stem}_demo_summary.json"

    started_at = time.perf_counter()
    preflight = run_preflight(
        audio_path=audio_path,
        language=language,
        model_name=model_name,
        whisper_model=whisper_model,
        prompt_path=prompt_path,
        require_real_stt=require_real_stt,
    )
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    if not preflight["ok"]:
        summary = {
            "ok": False,
            "stage": "preflight",
            "audio_file": str(audio_path),
            "preflight_report": str(preflight_path),
            "failed_checks": [check for check in preflight["checks"] if check["status"] == "fail"],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    report = run_pipeline(
        audio_path=audio_path,
        language=language,
        output_path=call_report_path,
        model_name=model_name,
        whisper_model_size=whisper_model,
        prompt_path=prompt_path,
        allow_transcript_fallback=not require_real_stt,
    )
    render_call_report(call_report_path, markdown_report_path)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        "ok": True,
        "stage": "complete",
        "audio_file": str(audio_path),
        "language": language.value,
        "model_name": model_name,
        "whisper_model": whisper_model,
        "elapsed_seconds": elapsed_seconds,
        "preflight_report": str(preflight_path),
        "call_report": str(call_report_path),
        "markdown_report": str(markdown_report_path),
        "final_risk": report.final_state.overall_risk.value,
        "alert_level": report.final_state.alert_level.value,
        "top_signals": [signal.value for signal in report.final_state.top_signals],
        "should_play_audio_alert": report.final_state.should_play_audio_alert,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full Senti desktop demo pipeline")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", default="pt-BR", choices=[item.value for item in Language])
    parser.add_argument("--output-dir", default=Path("outputs/demo"), type=Path)
    parser.add_argument("--model", default="google/gemma-4-e2b")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--prompt", default=None, type=Path)
    parser.add_argument("--allow-transcript-fallback", action="store_true", help="Allow sidecar/env transcript fallback for deterministic demos")
    args = parser.parse_args(argv)

    summary = run_demo(
        audio_path=args.audio,
        language=Language(args.language),
        output_dir=args.output_dir,
        model_name=args.model,
        whisper_model=args.whisper_model,
        prompt_path=args.prompt,
        require_real_stt=not args.allow_transcript_fallback,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
