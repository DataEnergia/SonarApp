# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Track A desktop pipeline: STT -> classifier -> decision engine -> report."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import structlog

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.schemas.classification import (  # noqa: E402
    AlertLevel,
    CallReport,
    CallSegmentClassification,
    CallSegmentInput,
    CallState,
    Language,
    RiskLevel,
)

from classifier import GemmaClassifier  # noqa: E402
from decision_engine import DecisionEngine  # noqa: E402
from stt_module import get_audio_duration_seconds, transcribe_audio  # noqa: E402

LOGGER = structlog.get_logger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure structured JSON logs for CLI runs."""

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_log_level_number(log_level)),
        cache_logger_on_first_use=True,
    )


def run_pipeline(
    audio_path: Path,
    language: Language,
    output_path: Path,
    model_name: str = "google/gemma-4-e2b",
    whisper_model_size: str = "small",
    prompt_path: Path | None = None,
    allow_transcript_fallback: bool = True,
) -> CallReport:
    """Run the local desktop pipeline and write a validated CallReport JSON."""

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    stt_language = {"pt-BR": "pt", "en-US": "en", "es-419": "es"}.get(language.value, "pt")
    transcript_segments = transcribe_audio(
        audio_path,
        language=stt_language,
        model_size=whisper_model_size,
        allow_transcript_fallback=allow_transcript_fallback,
    )
    classifier = GemmaClassifier(model_name=model_name, prompt_path=prompt_path)
    decision_engine = DecisionEngine(REPO_ROOT / "shared" / "signals_taxonomy.yaml")
    decision_engine.begin_call(call_id)

    classifications: list[CallSegmentClassification] = []
    history_summary: str | None = None
    for index, transcript_segment in enumerate(transcript_segments, start=1):
        segment_input = CallSegmentInput(
            segment_id=f"{call_id}_seg_{index:03d}",
            transcript=transcript_segment.text,
            history_summary=history_summary,
            language=language,
        )
        classification = classifier.classify_segment(segment_input)
        classifications.append(classification)
        decision_engine.update(classification)
        history_summary = _build_history_summary(classifications)

    final_state = decision_engine.end_call() if classifications else _empty_call_state(call_id)
    report = CallReport(
        call_id=call_id,
        language=language,
        duration_seconds=get_audio_duration_seconds(audio_path),
        segments=classifications,
        final_state=final_state,
        model_versions={"classifier": model_name, "stt": f"whisper-{whisper_model_size}"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info(
        "pipeline_report_written",
        call_id=call_id,
        output_path=str(output_path),
        segment_count=len(classifications),
        final_risk=report.final_state.overall_risk.value,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Senti Track A desktop pipeline.")
    parser.add_argument("--audio", required=True, type=Path, help="Input WAV/MP3 path")
    parser.add_argument("--output", required=True, type=Path, help="Output CallReport JSON path")
    parser.add_argument("--language", default="pt-BR", choices=[item.value for item in Language])
    parser.add_argument("--model", default="google/gemma-4-e2b", help="LM Studio model name")
    parser.add_argument("--prompt", default=None, type=Path, help="Classifier prompt file path")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument("--no-transcript-fallback", action="store_true", help="Force faster-whisper instead of sidecar/env transcript")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    report = run_pipeline(
        audio_path=args.audio,
        language=Language(args.language),
        output_path=args.output,
        model_name=args.model,
        whisper_model_size=args.whisper_model,
        prompt_path=args.prompt,
        allow_transcript_fallback=not args.no_transcript_fallback,
    )
    return 0 if isinstance(report, CallReport) else 1


def _build_history_summary(classifications: list[CallSegmentClassification]) -> str:
    recent = classifications[-3:]
    risks = ", ".join(c.risk_level.value for c in recent)
    signals = sorted({s.value for c in recent for s in c.signals_detected})
    # Include last two transcript excerpts so the LLM can resolve phrases split across segment boundaries
    excerpts = " | ".join(c.transcript_excerpt[:80] for c in recent[-2:])
    return f"Recent risks: {risks}. Signals so far: {signals}. Prior transcript: {excerpts}"


def _empty_call_state(call_id: str) -> CallState:
    return CallState(
        call_id=call_id,
        overall_risk=RiskLevel.SAFE,
        top_signals=[],
        alert_level=AlertLevel.NONE,
        should_notify_family=False,
        should_play_audio_alert=False,
        rationale_for_user="Nenhum trecho de fala foi transcrito para análise.",
        rationale_for_audit_log="No transcript segments were produced by STT.",
    )


def _log_level_number(log_level: str) -> int:
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    return levels.get(log_level.upper(), 20)


if __name__ == "__main__":
    raise SystemExit(main())
