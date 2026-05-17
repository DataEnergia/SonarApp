# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Speech-to-text adapter for the Track A desktop pipeline."""

from __future__ import annotations

import os
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import structlog

LOGGER = structlog.get_logger(__name__)
WINDOW_SECONDS = 5.0
OVERLAP_SECONDS = 1.0

# Module-level cache — prevents reloading Whisper on every chunk and avoids CUDA
# conflicts when LM Studio already holds the GPU.
_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


def _get_whisper_model(model_size: str) -> object:
    with _MODEL_LOCK:
        if model_size not in _MODEL_CACHE:
            from faster_whisper import WhisperModel
            LOGGER.info("stt_loading_model", model_size=model_size, device="cpu")
            _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        return _MODEL_CACHE[model_size]


class STTError(RuntimeError):
    """Raised when local transcription cannot complete."""


@dataclass(frozen=True)
class TranscriptSegment:
    """A timestamped transcript window emitted by STT."""

    start_time: float
    end_time: float
    text: str
    language_detected: str
    avg_logprob: float


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Return WAV duration in seconds, or 0.0 when unavailable."""

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return 0.0
            return frame_count / float(frame_rate)
    except (wave.Error, OSError):
        return 0.0


def transcribe_audio(
    audio_path: Path,
    language: str | None = "pt",
    model_size: str = "small",
    allow_transcript_fallback: bool = True,
) -> list[TranscriptSegment]:
    """Transcribe a local audio file with faster-whisper.

    For deterministic spike runs, `SENTI_TRANSCRIPT_TEXT` or a sidecar
    `<audio>.txt` file can provide a local transcript without cloud services.
    """

    if not audio_path.exists():
        raise STTError(f"Audio file not found: {audio_path}")

    fallback_text = None
    if allow_transcript_fallback:
        fallback_text = os.environ.get("SENTI_TRANSCRIPT_TEXT") or _read_sidecar_transcript(audio_path)
    if fallback_text:
        LOGGER.info("stt_using_local_transcript_fallback", audio_path=str(audio_path))
        return _window_text(fallback_text, get_audio_duration_seconds(audio_path), language)

    try:
        from faster_whisper import WhisperModel  # noqa: F401 — verify install
    except ImportError as exc:
        raise STTError("faster-whisper is not installed and no local transcript fallback was provided") from exc

    try:
        model = _get_whisper_model(model_size)
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 300},
            word_timestamps=False,
        )
        transcript_segments = _segments_to_windows(raw_segments, info.language or language)
    except Exception as exc:  # faster-whisper raises backend-specific exceptions.
        raise STTError(f"faster-whisper transcription failed: {exc}") from exc

    if not transcript_segments:
        LOGGER.warning("stt_no_speech_detected", audio_path=str(audio_path))
    return transcript_segments


def _read_sidecar_transcript(audio_path: Path) -> str | None:
    sidecar_path = audio_path.with_suffix(".txt")
    if not sidecar_path.exists():
        return None
    return sidecar_path.read_text(encoding="utf-8").strip()


def _window_text(text: str, duration_seconds: float, language: str) -> list[TranscriptSegment]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []
    effective_duration = max(duration_seconds, WINDOW_SECONDS)
    return [
        TranscriptSegment(
            start_time=0.0,
            end_time=min(effective_duration, WINDOW_SECONDS),
            text=clean_text,
            language_detected=language,
            avg_logprob=0.0,
        )
    ]


def _segments_to_windows(raw_segments: Iterable[object], language: str) -> list[TranscriptSegment]:
    windows: list[TranscriptSegment] = []
    current_text: list[str] = []
    current_start: float | None = None
    current_end = 0.0
    logprobs: list[float] = []

    for segment in raw_segments:
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        text = str(getattr(segment, "text", "")).strip()
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
        if not text:
            continue
        if current_start is None:
            current_start = start
        current_text.append(text)
        current_end = max(current_end, end)
        logprobs.append(avg_logprob)
        if current_end - current_start >= WINDOW_SECONDS:
            windows.append(_build_segment(current_start, current_end, current_text, language, logprobs))
            current_start = max(current_end - OVERLAP_SECONDS, current_start)
            current_text = []
            logprobs = []

    if current_start is not None and current_text:
        windows.append(_build_segment(current_start, current_end, current_text, language, logprobs))
    return windows


def _build_segment(
    start_time: float,
    end_time: float,
    text_parts: list[str],
    language: str,
    logprobs: list[float],
) -> TranscriptSegment:
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
    return TranscriptSegment(
        start_time=start_time,
        end_time=end_time,
        text=" ".join(text_parts),
        language_detected=language,
        avg_logprob=avg_logprob,
    )
