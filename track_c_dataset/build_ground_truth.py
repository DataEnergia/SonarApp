# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Build Track C ground_truth.json from scenario YAML files."""

from __future__ import annotations

import argparse
import json
import random
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def build_ground_truth(
    scenarios_dir: Path,
    audio_dir: Path,
    output_path: Path,
    holdout_fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Assemble canonical dataset annotations with reproducible holdout split."""

    scenario_paths = sorted(scenarios_dir.glob("*.yaml"))
    scenario_ids = [path.stem for path in scenario_paths]
    holdout_ids = _holdout_ids(scenario_ids, holdout_fraction, seed)

    samples: list[dict[str, Any]] = []
    for scenario_path in scenario_paths:
        scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        audio_file = f"{scenario['id']}.wav"
        audio_path = audio_dir / audio_file
        samples.append(
            {
                "audio_file": audio_file,
                "scenario_file": scenario_path.name,
                "language": scenario["language"],
                "type": scenario["type"],
                "primary_signal": scenario.get("primary_signal"),
                "all_signals": scenario.get("all_signals", []),
                "final_risk_level": scenario["final_risk_level"],
                "duration_seconds": _wav_duration_seconds(audio_path),
                "risk_evolution": scenario.get("risk_evolution", []),
                "voices": scenario.get("voices", {}),
                "split": "holdout" if scenario["id"] in holdout_ids else "train",
            }
        )

    ground_truth = {
        "version": "0.1-mini",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "samples": samples,
    }
    output_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track C ground_truth.json")
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios"))
    parser.add_argument("--audio", type=Path, default=Path("audio_dataset"))
    parser.add_argument("--output", type=Path, default=Path("ground_truth.json"))
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_ground_truth(args.scenarios, args.audio, args.output, args.holdout_fraction, args.seed)
    return 0


def _holdout_ids(ids: list[str], holdout_fraction: float, seed: int) -> set[str]:
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    count = max(1, round(len(shuffled) * holdout_fraction)) if shuffled else 0
    return set(shuffled[:count])


def _wav_duration_seconds(audio_path: Path) -> float:
    if not audio_path.exists():
        return 0.0
    with wave.open(str(audio_path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


if __name__ == "__main__":
    raise SystemExit(main())
