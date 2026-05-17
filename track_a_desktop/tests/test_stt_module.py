# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import wave
from pathlib import Path

from stt_module import get_audio_duration_seconds, transcribe_audio


def test_sidecar_transcript_produces_segment(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    audio_path.with_suffix(".txt").write_text("Banco urgente precisa do codigo SMS.", encoding="utf-8")

    segments = transcribe_audio(audio_path, language="pt", model_size="tiny")

    assert len(segments) == 1
    assert segments[0].start_time == 0.0
    assert segments[0].end_time > segments[0].start_time
    assert "Banco urgente" in segments[0].text


def test_sidecar_can_be_disabled(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    audio_path.with_suffix(".txt").write_text("texto local", encoding="utf-8")

    try:
        segments = transcribe_audio(audio_path, language="pt", model_size="tiny", allow_transcript_fallback=False)
    except Exception as exc:
        assert "faster-whisper" in str(exc) or "transcription failed" in str(exc)
    else:
        assert segments == []


def test_wav_duration_is_read(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path, seconds=2)
    assert get_audio_duration_seconds(audio_path) == 2.0


def _write_silent_wav(audio_path: Path, seconds: int = 1) -> None:
    sample_rate = 16000
    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate * seconds)
