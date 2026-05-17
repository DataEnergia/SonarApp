# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Local persistence helpers for the desktop test UI."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def new_recording_id() -> str:
    """Return a stable local recording id for user-provided audio."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"rec_{timestamp}_{uuid.uuid4().hex[:8]}"


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    """Append one JSON object to a local JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL records, newest first."""

    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.reverse()
    return rows[:limit] if limit is not None else rows


def save_feedback(path: Path, feedback: dict[str, Any]) -> dict[str, Any]:
    """Persist human feedback for one analyzed recording."""

    record = {
        "feedback_id": f"fb_{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now(UTC).isoformat(),
        **feedback,
    }
    append_jsonl(path, record)
    return record


def summarize_call_report(report_path: Path) -> dict[str, Any]:
    """Extract UI summary fields from a CallReport JSON file."""

    data = json.loads(report_path.read_text(encoding="utf-8"))
    final_state = data.get("final_state", {})
    segments = data.get("segments", [])
    return {
        "call_id": data.get("call_id"),
        "language": data.get("language"),
        "duration_seconds": data.get("duration_seconds"),
        "final_risk": final_state.get("overall_risk"),
        "alert_level": final_state.get("alert_level"),
        "should_play_audio_alert": final_state.get("should_play_audio_alert"),
        "rationale_for_user": final_state.get("rationale_for_user"),
        "top_signals": final_state.get("top_signals", []),
        "segments": [
            {
                "segment_id": segment.get("segment_id"),
                "risk_level": segment.get("risk_level"),
                "confidence": segment.get("confidence"),
                "signals_detected": segment.get("signals_detected", []),
                "transcript_excerpt": segment.get("transcript_excerpt"),
                "suggested_action_for_user": segment.get("suggested_action_for_user"),
            }
            for segment in segments
        ],
    }


def add_to_dataset(
    audio_src: Path,
    report_path: Path | None,
    dataset_audio_dir: Path,
    ground_truth_path: Path,
    recording_id: str,
    user_label: str,
    user_corrected_risk: str,
    language: str,
    notes: str = "",
) -> dict[str, Any]:
    """Copy user audio to the TTS dataset and append a ground-truth entry.

    Args:
        audio_src: path to the uploaded/recorded audio file.
        report_path: path to the call_report.json (may be None).
        dataset_audio_dir: target directory for dataset audio files.
        ground_truth_path: path to ground_truth_tts.json.
        recording_id: the recording session id.
        user_label: "scam" | "legitimate" | "uncertain".
        user_corrected_risk: "safe" | "suspicious" | "danger".
        language: "pt-BR" | "en-US".
        notes: free-text annotation from the user.

    Returns:
        Dict describing the new dataset entry added.
    """
    dataset_audio_dir.mkdir(parents=True, exist_ok=True)
    short_id = recording_id.replace("rec_", "")[:16]
    prefix = "user_scam" if user_label == "scam" else "user_legit" if user_label == "legitimate" else "user_unk"
    dest_name = f"{prefix}_{short_id}{audio_src.suffix}"
    dest_path = dataset_audio_dir / dest_name

    # Convert to WAV 16kHz mono if not already WAV — use faster-whisper's decoder.
    if audio_src.suffix.lower() != ".wav":
        _convert_to_wav(audio_src, dest_path.with_suffix(".wav"))
        dest_path = dest_path.with_suffix(".wav")
    else:
        shutil.copy2(audio_src, dest_path)

    # Read existing ground truth.
    if ground_truth_path.exists():
        gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    else:
        gt = {"version": "0.1-user", "samples": [], "splits": {"train": [], "holdout": []}}

    # Extract signals from report if available.
    detected_signals: list[str] = []
    duration_seconds = 0.0
    if report_path and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        duration_seconds = report.get("duration_seconds", 0.0)
        final_state = report.get("final_state", {})
        detected_signals = final_state.get("top_signals", [])

    entry: dict[str, Any] = {
        "audio_file": dest_path.name,
        "source": "user_recording",
        "recording_id": recording_id,
        "language": language,
        "type": "scam" if user_label == "scam" else "legitimate",
        "primary_signal": detected_signals[0] if detected_signals else "unknown",
        "all_signals": detected_signals,
        "final_risk_level": user_corrected_risk,
        "duration_seconds": duration_seconds,
        "user_notes": notes,
        "added_at": datetime.now(UTC).isoformat(),
        "split": "train",
    }

    samples: list[dict[str, Any]] = gt.get("samples", [])
    # Avoid duplicate entries for the same recording_id.
    samples = [s for s in samples if s.get("recording_id") != recording_id]
    samples.append(entry)
    gt["samples"] = samples
    gt["updated_at"] = datetime.now(UTC).isoformat()

    ground_truth_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def get_library_entries(output_dir: Path, feedback_path: Path) -> list[dict[str, Any]]:
    """Return saved call reports, newest first, with confirmed labels."""
    labels: dict[str, str] = {}
    if feedback_path.exists():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("recording_id") or rec.get("call_id", "")
                lbl = rec.get("user_label")
                if rid and lbl:
                    labels[rid] = lbl
            except json.JSONDecodeError:
                continue
    entries = []
    for report_path in sorted(output_dir.glob("*/*_call_report.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            final = data.get("final_state", {})
            recording_id = report_path.parent.name
            entries.append({
                "recording_id": recording_id,
                "call_id": data.get("call_id", ""),
                "language": data.get("language", "pt-BR"),
                "duration_seconds": round(data.get("duration_seconds", 0), 1),
                "overall_risk": final.get("overall_risk", "safe"),
                "alert_level": final.get("alert_level", "none"),
                "top_signals": final.get("top_signals", []),
                "segment_count": len(data.get("segments", [])),
                "label": labels.get(recording_id),
                "report_path": str(report_path),
                "timestamp": report_path.stat().st_mtime,
            })
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return entries


def get_few_shot_examples(output_dir: Path, feedback_path: Path, language: str, k: int = 2) -> str:
    """Build few-shot context from confirmed library examples for Gemma prompt injection."""
    entries = get_library_entries(output_dir, feedback_path)
    confirmed = [e for e in entries if e.get("label") in ("scam", "legitimate")]
    if not confirmed:
        return ""
    lines = ["# Exemplos verificados da sua biblioteca local\n"]
    for ex in confirmed[:k]:
        label_str = "GOLPE CONFIRMADO" if ex["label"] == "scam" else "CHAMADA LEGÍTIMA"
        signals = ", ".join(ex.get("top_signals", [])) or "nenhum sinal detectado"
        risk = ex.get("overall_risk", "safe")
        excerpt = ""
        try:
            data = json.loads(Path(ex["report_path"]).read_text(encoding="utf-8"))
            segs = data.get("segments", [])
            if segs:
                excerpt = segs[0].get("transcript_excerpt", "")[:120]
        except Exception:
            pass
        if excerpt:
            lines.append(f"## {label_str}:")
            lines.append(f'Trecho: "{excerpt}"')
            lines.append(f"Sinais: {signals} → {risk}\n")
    return "\n".join(lines) if len(lines) > 1 else ""


def _convert_to_wav(src: Path, dest: Path) -> None:
    """Convert audio (WebM, MP3, etc.) to 16kHz mono 16-bit WAV using PyAV."""
    try:
        import av
        import numpy as np
        import soundfile as sf

        samples_list: list[Any] = []
        target_sr = 16000
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        with av.open(str(src)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError(f"No audio stream in {src}")
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for rf in resampler.resample(frame):
                        samples_list.append(rf.to_ndarray().flatten())
        # Flush resampler
        for rf in resampler.resample(None):
            samples_list.append(rf.to_ndarray().flatten())

        if not samples_list:
            raise ValueError(f"No audio decoded from {src}")
        combined = np.concatenate(samples_list).astype(np.int16)
        sf.write(str(dest), combined, target_sr, subtype="PCM_16")
    except Exception as exc:
        # If conversion fails, copy as-is so the entry is still created.
        shutil.copy2(src, dest.with_suffix(src.suffix))
        raise RuntimeError(f"WAV conversion failed: {exc}") from exc
