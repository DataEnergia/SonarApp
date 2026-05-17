# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Operational preflight checks for running Sonar on a real local audio file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.schemas.classification import Language  # noqa: E402
from stt_module import get_audio_duration_seconds  # noqa: E402


def run_preflight(
    audio_path: Path,
    language: Language,
    model_name: str = "google/gemma-4-e2b",
    whisper_model: str = "small",
    prompt_path: Path | None = None,
    ollama_url: str = "http://localhost:1234",
    require_real_stt: bool = False,
) -> dict[str, Any]:
    """Run fast local checks and return a machine-readable status report."""

    prompt_path = prompt_path or REPO_ROOT / "shared" / "prompts" / "classifier_v3.txt"
    checks = [
        _check_audio(audio_path),
        _check_prompt(prompt_path),
        _check_language(language),
        _check_faster_whisper(whisper_model, require_real_stt),
        _check_ollama(ollama_url, model_name),
    ]
    ok = not any(check["status"] == "fail" for check in checks)
    return {
        "ok": ok,
        "model_name": model_name,
        "language": language.value,
        "whisper_model": whisper_model,
        "checks": checks,
        "recommended_command": _recommended_command(audio_path, language, model_name, whisper_model, prompt_path, require_real_stt),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check if Sonar Track A can run on a real local audio file")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", default="pt-BR", choices=[item.value for item in Language])
    parser.add_argument("--model", default="google/gemma-4-e2b")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--prompt", default=None, type=Path)
    parser.add_argument("--ollama-url", default="http://localhost:1234")
    parser.add_argument("--require-real-stt", action="store_true", help="Fail if faster-whisper is unavailable")
    parser.add_argument("--output", default=None, type=Path, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    report = run_preflight(
        audio_path=args.audio,
        language=Language(args.language),
        model_name=args.model,
        whisper_model=args.whisper_model,
        prompt_path=args.prompt,
        ollama_url=args.ollama_url,
        require_real_stt=args.require_real_stt,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def _check_audio(audio_path: Path) -> dict[str, Any]:
    if not audio_path.exists():
        return _check("audio_file", "fail", f"Audio file not found: {audio_path}")
    duration = get_audio_duration_seconds(audio_path)
    suffix = audio_path.suffix.lower()
    if suffix != ".wav":
        return _check(
            "audio_file",
            "warn",
            f"File exists but duration could not be validated as WAV. faster-whisper may still read it: {audio_path}",
            duration_seconds=duration,
        )
    if duration <= 0:
        return _check("audio_file", "fail", f"WAV exists but duration is zero or unreadable: {audio_path}", duration_seconds=duration)
    return _check("audio_file", "pass", f"Audio file is readable: {audio_path}", duration_seconds=round(duration, 3))


def _check_prompt(prompt_path: Path) -> dict[str, Any]:
    if not prompt_path.exists():
        return _check("prompt", "fail", f"Prompt file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        return _check("prompt", "fail", f"Prompt file is empty: {prompt_path}")
    return _check("prompt", "pass", f"Prompt file loaded: {prompt_path}", bytes=len(text.encode("utf-8")))


def _check_language(language: Language) -> dict[str, Any]:
    return _check("language", "pass", f"Language accepted: {language.value}")


def _check_faster_whisper(whisper_model: str, require_real_stt: bool) -> dict[str, Any]:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        status = "fail" if require_real_stt else "warn"
        return _check("faster_whisper", status, "faster-whisper is not installed; only sidecar/env transcript fallback can run")
    return _check("faster_whisper", "pass", f"faster-whisper import ok; requested model={whisper_model}")


def _check_ollama(ollama_url: str, model_name: str) -> dict[str, Any]:
    base = ollama_url.rstrip("/")
    try:
        response = requests.get(f"{base}/v1/models", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        return _check("lmstudio", "fail", f"LM Studio is not reachable at {ollama_url}: {exc}")
    data = response.json()
    models = data.get("data") or data.get("models", [])
    names = {str(m.get("id", m.get("name", ""))) for m in models}
    if names and model_name not in names:
        return _check("lmstudio_model", "pass", f"LM Studio reachable; loaded model: {sorted(names)}")
    return _check("lmstudio_model", "pass", f"LM Studio is reachable at {ollama_url}")


def _recommended_command(
    audio_path: Path,
    language: Language,
    model_name: str,
    whisper_model: str,
    prompt_path: Path,
    require_real_stt: bool,
) -> str:
    output = CURRENT_DIR / "outputs" / f"{audio_path.stem}_call_report.json"
    parts = [
        "python pipeline.py",
        f'--audio "{audio_path}"',
        f"--language {language.value}",
        f'--output "{output}"',
        f"--model {model_name}",
        f"--whisper-model {whisper_model}",
        f'--prompt "{prompt_path}"',
        "--log-level INFO",
    ]
    if require_real_stt:
        parts.append("--no-transcript-fallback")
    return " ".join(parts)


def _check(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, **extra}


if __name__ == "__main__":
    raise SystemExit(main())
