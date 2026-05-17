# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Basic WAV quality checks for Track C generated audio."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any


def check_audio_file(audio_path: Path) -> dict[str, Any]:
    """Return basic format and duration checks for one WAV file."""

    result: dict[str, Any] = {"audio_file": audio_path.name, "exists": audio_path.exists(), "valid": False}
    if not audio_path.exists():
        result["error"] = "file_missing"
        return result
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            duration = frames / float(frame_rate) if frame_rate else 0.0
    except wave.Error as exc:
        result["error"] = str(exc)
        return result

    result.update(
        {
            "channels": channels,
            "sample_width_bytes": sample_width,
            "frame_rate": frame_rate,
            "duration_seconds": round(duration, 3),
            "valid": channels >= 1 and sample_width > 0 and frame_rate > 0 and duration > 0,
            "format_ok": channels == 1 and sample_width == 2 and frame_rate == 16000,
        }
    )
    return result


def check_dataset(dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    """Check every WAV in a dataset directory and write JSON report."""

    files = sorted(dataset_dir.glob("*.wav"))
    results = [check_audio_file(path) for path in files]
    report = {
        "dataset_dir": str(dataset_dir),
        "file_count": len(results),
        "valid_count": sum(1 for result in results if result["valid"]),
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Track C generated WAV files")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check_dataset(args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
