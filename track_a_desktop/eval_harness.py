# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Evaluation harness for Track A pipeline outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import structlog

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.schemas.classification import CallReport, Language, RiskLevel  # noqa: E402

from pipeline import configure_logging, run_pipeline  # noqa: E402

LOGGER = structlog.get_logger(__name__)
RISK_LABELS = [RiskLevel.SAFE.value, RiskLevel.SUSPICIOUS.value, RiskLevel.DANGER.value]

PipelineRunner = Callable[[Path, Language, Path, str, Path | None], CallReport]


def run_evaluation(
    dataset_dir: Path,
    ground_truth_path: Path,
    output_path: Path,
    model_name: str = "gemma4:e2b",
    limit: int | None = None,
    split: str | None = None,
    prompt_path: Path | None = None,
    sample_list_path: Path | None = None,
    disable_transcript_fallback: bool = False,
    whisper_model_size: str = "small",
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Run Track A over a labeled dataset and write aggregate metrics."""

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    samples = _filter_samples(
        ground_truth["samples"],
        limit=limit,
        split=split,
        sample_names=_read_sample_list(sample_list_path),
    )
    runner = pipeline_runner or _default_runner
    report_dir = output_path.parent / "pipeline_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for sample in samples:
        sample_started_at = time.perf_counter()
        audio_path = dataset_dir / sample["audio_file"]
        sample_report_path = report_dir / f"{Path(sample['audio_file']).stem}_report.json"
        try:
            report = runner(
                audio_path,
                Language(sample["language"]),
                sample_report_path,
                model_name,
                prompt_path,
                disable_transcript_fallback,
                whisper_model_size,
            )
            predicted_risk = report.final_state.overall_risk.value
            predicted_signals = sorted({signal.value for segment in report.segments for signal in segment.signals_detected})
            error = None
        except Exception as exc:  # evaluation must record failures instead of hiding them.
            predicted_risk = "error"
            predicted_signals = []
            error = str(exc)

        results.append(
            {
                "audio_file": sample["audio_file"],
                "split": sample.get("split", "unknown"),
                "language": sample["language"],
                "type": sample["type"],
                "expected_risk": sample["final_risk_level"],
                "predicted_risk": predicted_risk,
                "expected_signals": sample.get("all_signals", []),
                "predicted_signals": predicted_signals,
                "correct_risk": predicted_risk == sample["final_risk_level"],
                "latency_seconds": round(time.perf_counter() - sample_started_at, 3),
                "error": error,
            }
        )
        LOGGER.info(
            "eval_sample_completed",
            audio_file=sample["audio_file"],
            expected_risk=sample["final_risk_level"],
            predicted_risk=predicted_risk,
            error=error,
        )

    evaluation = {
        "version": "0.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "prompt_path": str(prompt_path) if prompt_path else None,
        "sample_list_path": str(sample_list_path) if sample_list_path else None,
        "transcript_fallback_disabled": disable_transcript_fallback,
        "whisper_model_size": whisper_model_size,
        "dataset_dir": str(dataset_dir),
        "ground_truth_path": str(ground_truth_path),
        "sample_count": len(results),
        "metrics": compute_metrics(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown_summary(evaluation), encoding="utf-8")
    return evaluation


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute risk-level metrics, confusion matrix, latency, and signal metrics."""

    evaluated = [result for result in results if result["predicted_risk"] in RISK_LABELS]
    total = len(results)
    correct = sum(1 for result in evaluated if result["correct_risk"])
    confusion = _confusion_matrix(evaluated)
    per_risk = {label: _precision_recall_f1(evaluated, label) for label in RISK_LABELS}
    signal_metrics = _signal_metrics(evaluated)
    legitimate = [result for result in evaluated if result["type"] == "legitimate"]
    false_danger = sum(1 for result in legitimate if result["predicted_risk"] == RiskLevel.DANGER.value)
    latencies = [float(result["latency_seconds"]) for result in evaluated]
    return {
        "total_samples": total,
        "evaluated_samples": len(evaluated),
        "error_count": total - len(evaluated),
        "accuracy": correct / len(evaluated) if evaluated else 0.0,
        "false_danger_rate_on_legitimate": false_danger / len(legitimate) if legitimate else 0.0,
        "confusion_matrix": confusion,
        "per_risk": per_risk,
        "per_signal": signal_metrics,
        "latency_seconds": _latency_summary(latencies),
    }


def render_markdown_summary(evaluation: dict[str, Any]) -> str:
    """Render a concise human-readable evaluation summary."""

    metrics = evaluation["metrics"]
    lines = [
        f"# Eval Run — {evaluation['generated_at']}",
        "",
        f"- Model: `{evaluation['model_name']}`",
        f"- Samples: {metrics['evaluated_samples']} evaluated / {metrics['total_samples']} total",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- False danger rate on legitimate: {metrics['false_danger_rate_on_legitimate']:.3f}",
        f"- Latency mean: {metrics['latency_seconds']['mean']:.3f}s",
        "",
        "## Per-Risk Metrics",
        "",
        "| Risk | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for risk in RISK_LABELS:
        item = metrics["per_risk"][risk]
        lines.append(f"| {risk} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    lines.extend(["", "## Confusion Matrix", "", "```json", json.dumps(metrics["confusion_matrix"], indent=2), "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Track A pipeline over Track C dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--prompt", default=None, type=Path)
    parser.add_argument("--sample-list", default=None, type=Path)
    parser.add_argument("--no-transcript-fallback", action="store_true")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "holdout"], default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    run_evaluation(
        args.dataset,
        args.ground_truth,
        args.output,
        args.model,
        args.limit,
        args.split,
        args.prompt,
        args.sample_list,
        args.no_transcript_fallback,
        args.whisper_model,
    )
    return 0


def _default_runner(
    audio_path: Path,
    language: Language,
    output_path: Path,
    model_name: str,
    prompt_path: Path | None,
    disable_transcript_fallback: bool,
    whisper_model_size: str = "small",
) -> CallReport:
    return run_pipeline(
        audio_path=audio_path,
        language=language,
        output_path=output_path,
        model_name=model_name,
        whisper_model_size=whisper_model_size,
        prompt_path=prompt_path,
        allow_transcript_fallback=not disable_transcript_fallback,
    )


def _filter_samples(
    samples: list[dict[str, Any]],
    limit: int | None,
    split: str | None,
    sample_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        sample for sample in samples
        if (split is None or sample.get("split") == split)
        and (sample_names is None or sample.get("audio_file") in sample_names)
    ]
    return filtered[:limit] if limit is not None else filtered


def _read_sample_list(sample_list_path: Path | None) -> set[str] | None:
    if sample_list_path is None:
        return None
    return {
        line.strip()
        for line in sample_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in RISK_LABELS} for actual in RISK_LABELS}
    for result in results:
        matrix[result["expected_risk"]][result["predicted_risk"]] += 1
    return matrix


def _precision_recall_f1(results: list[dict[str, Any]], label: str) -> dict[str, float]:
    true_positive = sum(1 for result in results if result["expected_risk"] == label and result["predicted_risk"] == label)
    false_positive = sum(1 for result in results if result["expected_risk"] != label and result["predicted_risk"] == label)
    false_negative = sum(1 for result in results if result["expected_risk"] == label and result["predicted_risk"] != label)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _signal_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    labels = sorted({signal for result in results for signal in result["expected_signals"] + result["predicted_signals"]})
    return {label: _signal_precision_recall_f1(results, label) for label in labels}


def _signal_precision_recall_f1(results: list[dict[str, Any]], label: str) -> dict[str, float]:
    true_positive = sum(1 for result in results if label in result["expected_signals"] and label in result["predicted_signals"])
    false_positive = sum(1 for result in results if label not in result["expected_signals"] and label in result["predicted_signals"])
    false_negative = sum(1 for result in results if label in result["expected_signals"] and label not in result["predicted_signals"])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(latencies)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": max(ordered),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
