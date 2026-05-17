# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import wave
from pathlib import Path

import preflight
from shared.schemas.classification import Language


def test_preflight_reports_missing_audio(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("classify", encoding="utf-8")
    monkeypatch.setattr(preflight, "_check_ollama", lambda *_args, **_kwargs: preflight._check("ollama_model", "pass", "ok"))

    report = preflight.run_preflight(
        audio_path=tmp_path / "missing.wav",
        language=Language.PT_BR,
        prompt_path=prompt_path,
        require_real_stt=False,
    )

    assert report["ok"] is False
    assert any(check["name"] == "audio_file" and check["status"] == "fail" for check in report["checks"])


def test_preflight_accepts_readable_audio_and_builds_command(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "call.wav"
    prompt_path = tmp_path / "prompt.txt"
    _write_silent_wav(audio_path)
    prompt_path.write_text("classify", encoding="utf-8")
    monkeypatch.setattr(preflight, "_check_ollama", lambda *_args, **_kwargs: preflight._check("ollama_model", "pass", "ok"))
    monkeypatch.setattr(preflight, "_check_faster_whisper", lambda *_args, **_kwargs: preflight._check("faster_whisper", "pass", "ok"))

    report = preflight.run_preflight(
        audio_path=audio_path,
        language=Language.PT_BR,
        model_name="gemma4:e2b",
        whisper_model="tiny",
        prompt_path=prompt_path,
        require_real_stt=True,
    )

    assert report["ok"] is True
    assert "--no-transcript-fallback" in report["recommended_command"]
    assert "gemma4:e2b" in report["recommended_command"]


def _write_silent_wav(audio_path: Path, seconds: int = 1) -> None:
    sample_rate = 16000
    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate * seconds)
