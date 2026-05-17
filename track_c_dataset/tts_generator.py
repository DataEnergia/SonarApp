# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Offline TTS generation for Track C scenarios using Windows SAPI voices."""

from __future__ import annotations

import argparse
import audioop
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SAMPLE_RATE = 16_000


class TTSError(RuntimeError):
    """Raised when local TTS generation fails."""


@dataclass(frozen=True)
class Scenario:
    """Minimal scenario fields needed by the TTS generator."""

    scenario_id: str
    language: str
    transcript: str


def generate_audio(scenario_path: Path, output_dir: Path) -> Path:
    """Generate a local WAV and transcript sidecar for a scenario."""

    scenario = _load_scenario(scenario_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{scenario.scenario_id}.wav"
    txt_path = output_dir / f"{scenario.scenario_id}.txt"
    plain_text = _plain_transcript(scenario.transcript)
    if not plain_text:
        raise TTSError(f"Scenario transcript is empty: {scenario_path}")

    _synthesize_with_pyttsx3(plain_text, wav_path, scenario.language)
    txt_path.write_text(plain_text, encoding="utf-8")
    return wav_path


def generate_batch(scenarios_dir: Path, output_dir: Path) -> list[Path]:
    """Generate WAV files for every YAML scenario in a directory."""

    return [generate_audio(path, output_dir) for path in sorted(scenarios_dir.glob("*.yaml"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline TTS audio for Track C scenarios")
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--batch", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("audio_dataset"))
    args = parser.parse_args()
    if args.scenario is None and args.batch is None:
        parser.error("Provide --scenario or --batch")
    if args.scenario is not None:
        generate_audio(args.scenario, args.output)
    if args.batch is not None:
        generate_batch(args.batch, args.output)
    return 0


def _load_scenario(scenario_path: Path) -> Scenario:
    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    return Scenario(
        scenario_id=str(data["id"]),
        language=str(data["language"]),
        transcript=str(data["transcript"]),
    )


def _plain_transcript(transcript: str) -> str:
    lines = []
    for line in transcript.splitlines():
        clean = re.sub(r"^\[[^\]]+\]\s*", "", line.strip())
        if clean:
            lines.append(clean)
    return " ".join(lines)


def _synthesize_with_pyttsx3(text: str, wav_path: Path, language: str) -> None:
    try:
        import pyttsx3
    except ImportError as exc:
        raise TTSError("pyttsx3 is not installed. Run: pip install pyttsx3") from exc

    engine = pyttsx3.init()
    _select_voice(engine, language)
    engine.setProperty("rate", 145)
    engine.setProperty("volume", 0.9)
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise TTSError(f"pyttsx3 did not create audio: {wav_path}")
    _convert_to_16khz_mono(wav_path)
    _validate_wav(wav_path)


def _select_voice(engine: object, language: str) -> None:
    voices = engine.getProperty("voices")
    preferred_terms = ["portuguese", "brazil", "portugal", "maria"] if language == "pt-BR" else ["english", "zira", "david"]
    for voice in voices:
        voice_text = " ".join(str(value).lower() for value in [getattr(voice, "id", ""), getattr(voice, "name", ""), getattr(voice, "languages", "")])
        if any(term in voice_text for term in preferred_terms):
            engine.setProperty("voice", voice.id)
            return


def _validate_wav(wav_path: Path) -> None:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() < 1:
                raise TTSError(f"Generated WAV has no channels: {wav_path}")
    except wave.Error as exc:
        raise TTSError(f"Generated file is not a readable WAV: {wav_path}") from exc


def _convert_to_16khz_mono(wav_path: Path) -> None:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    if frame_rate != SAMPLE_RATE:
        frames, _ = audioop.ratecv(frames, sample_width, channels, frame_rate, SAMPLE_RATE, None)
        frame_rate = SAMPLE_RATE

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(frames)


if __name__ == "__main__":
    raise SystemExit(main())
