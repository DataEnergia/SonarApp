# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Local browser UI for Sonar — real-time scam call detection."""

from __future__ import annotations

import argparse
import cgi
import json
import sys
import tempfile
import queue
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CURRENT_DIR = Path(__file__).resolve().parent
TRACK_A_DIR = CURRENT_DIR.parent
REPO_ROOT = TRACK_A_DIR.parent
if str(TRACK_A_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_A_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier import GemmaClassifier, correct_stt_transcript  # noqa: E402
from decision_engine import DecisionEngine  # noqa: E402
from preflight import run_preflight  # noqa: E402
from shared.schemas.classification import (  # noqa: E402
    CallReport,
    CallSegmentInput,
    Language,
)
from stt_module import get_audio_duration_seconds, transcribe_audio  # noqa: E402
from ui_local.store import (  # noqa: E402
    add_to_dataset,
    get_few_shot_examples,
    get_library_entries,
    new_recording_id,
    read_jsonl,
    save_feedback,
)
from ui_local.contacts_store import (  # noqa: E402
    add_contact,
    load_contacts,
    remove_contact,
)
from alert_sender import load_smtp_config, save_smtp_config, send_alert  # noqa: E402

_WHISPER_LANG: dict[str, str] = {"pt-BR": "pt", "en-US": "en", "es-419": "es"}
_WHISPER_TO_LANG: dict[str, Language] = {"pt": Language.PT_BR, "en": Language.EN_US, "es": Language.ES_419}

INPUT_DIR = TRACK_A_DIR / "inputs" / "user_recordings"
CONTACTS_PATH = TRACK_A_DIR / "config" / "contacts.json"
SONAR_CONFIG_PATH = TRACK_A_DIR / "config" / "sonar_config.json"
OUTPUT_DIR = TRACK_A_DIR / "outputs" / "user_tests"
FEEDBACK_PATH = TRACK_A_DIR / "feedback" / "user_feedback.jsonl"
PROMPT_PATH = REPO_ROOT / "shared" / "prompts" / "classifier_v3.txt"
DATASET_AUDIO_DIR = REPO_ROOT / "track_c_dataset" / "audio_tts"
GROUND_TRUTH_PATH = REPO_ROOT / "track_c_dataset" / "ground_truth_tts.json"

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_LAST_JOB_ID: str | None = None
_ALERT_LOG: list[dict[str, Any]] = []
_ALERT_LOG_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSIONS_LOCK = threading.Lock()
_INCOMING_CALL: dict[str, Any] | None = None
_INCOMING_CALL_LOCK = threading.Lock()


def _run_analysis_job(
    job_id: str,
    recording_id: str,
    audio_path: Path,
    audio_suffix: str,
    language: Language,
    model_name: str,
    whisper_model: str,
    allow_fallback: bool,
) -> None:
    def _patch(update: dict) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id].update(update)

    try:
        _patch({"stage": "preflight"})
        preflight = run_preflight(
            audio_path=audio_path,
            language=language,
            model_name=model_name,
            whisper_model=whisper_model,
            prompt_path=PROMPT_PATH,
            require_real_stt=not allow_fallback,
        )
        if not preflight["ok"]:
            failed = [c["check"] for c in preflight.get("checks", []) if c.get("status") == "fail"]
            _patch({"status": "error", "error": f"Preflight falhou: {', '.join(failed)}"})
            return

        _patch({"stage": "stt"})
        stt_lang = _WHISPER_LANG.get(language.value, None)
        transcript_segments = transcribe_audio(
            audio_path,
            language=stt_lang,
            model_size=whisper_model,
            allow_transcript_fallback=allow_fallback,
        )
        _patch({"stage": "classifying", "total": len(transcript_segments)})

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        classifier = GemmaClassifier(model_name=model_name, prompt_path=PROMPT_PATH)
        engine = DecisionEngine(REPO_ROOT / "shared" / "signals_taxonomy.yaml")
        engine.begin_call(call_id)

        classifications: list = []
        history_summary: str | None = None

        for index, ts in enumerate(transcript_segments, start=1):
            seg_input = CallSegmentInput(
                segment_id=f"{call_id}_seg_{index:03d}",
                transcript=ts.text,
                history_summary=history_summary,
                language=language,
            )
            classification = classifier.classify_segment(seg_input)
            classifications.append(classification)
            call_state = engine.update(classification)
            history_summary = _history_summary(classifications)

            seg_entry: dict[str, Any] = {
                "classification": {
                    "segment_id": classification.segment_id,
                    "transcript_excerpt": classification.transcript_excerpt,
                    "signals_detected": [s.value for s in classification.signals_detected],
                    "risk_level": classification.risk_level.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "suggested_action_for_user": classification.suggested_action_for_user,
                    "needs_deeper_analysis": classification.needs_deeper_analysis,
                },
                "call_state": {
                    "overall_risk": call_state.overall_risk.value,
                    "alert_level": call_state.alert_level.value,
                    "top_signals": [s.value for s in call_state.top_signals],
                    "rationale_for_user": call_state.rationale_for_user,
                    "should_play_audio_alert": call_state.should_play_audio_alert,
                },
            }
            with _JOBS_LOCK:
                _JOBS[job_id]["segments"].append(seg_entry)

        final_state = engine.end_call()
        final: dict[str, Any] = {
            "overall_risk": final_state.overall_risk.value,
            "alert_level": final_state.alert_level.value,
            "top_signals": [s.value for s in final_state.top_signals],
            "rationale_for_user": final_state.rationale_for_user,
            "should_notify_family": final_state.should_notify_family,
            "should_play_audio_alert": final_state.should_play_audio_alert,
        }

        report = CallReport(
            call_id=call_id,
            language=language,
            duration_seconds=get_audio_duration_seconds(audio_path),
            segments=classifications,
            final_state=final_state,
            model_versions={"classifier": model_name, "stt": f"whisper-{whisper_model}"},
        )
        call_report_path = OUTPUT_DIR / recording_id / f"{audio_path.stem}_call_report.json"
        call_report_path.parent.mkdir(parents=True, exist_ok=True)
        call_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        _patch({"status": "done", "stage": "done", "final": final})

    except Exception as exc:
        _patch({"status": "error", "error": str(exc)})


def _history_summary(classifications: list) -> str:
    recent = classifications[-3:]
    risks = ", ".join(c.risk_level.value for c in recent)
    signals = sorted({s.value for c in recent for s in c.signals_detected})
    excerpts = " | ".join(c.transcript_excerpt[:80] for c in recent[-2:])
    return f"Recent risks: {risks}. Signals so far: {signals}. Prior transcript: {excerpts}"


def _classification_worker(session: dict[str, Any]) -> None:
    """Background thread: drain transcript queue, classify with Gemma, update session."""
    q: queue.Queue = session["classification_queue"]
    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            if session.get("status") != "active":
                break
            continue
        if item is None:  # sentinel — call ended
            break
        seg_id, transcript, language = item
        try:
            transcript = correct_stt_transcript(transcript, language.value)
            seg_input = CallSegmentInput(
                segment_id=seg_id,
                transcript=transcript,
                history_summary=session.get("history_summary"),
                language=language,
            )
            classification = session["classifier"].classify_segment(seg_input)
            call_state = session["engine"].update(classification)
            session["classifications"].append(classification)
            session["history_summary"] = _history_summary(session["classifications"])
            seg_entry: dict[str, Any] = {
                "classification": {
                    "segment_id": classification.segment_id,
                    "transcript_excerpt": classification.transcript_excerpt,
                    "signals_detected": [s.value for s in classification.signals_detected],
                    "risk_level": classification.risk_level.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "suggested_action_for_user": classification.suggested_action_for_user,
                },
                "call_state": {
                    "overall_risk": call_state.overall_risk.value,
                    "alert_level": call_state.alert_level.value,
                    "top_signals": [s.value for s in call_state.top_signals],
                    "should_play_audio_alert": call_state.should_play_audio_alert,
                    "rationale_for_user": call_state.rationale_for_user,
                },
            }
            session["segments"].append(seg_entry)
            # ── Emergency alert trigger ────────────────────────────────
            risk_value = call_state.overall_risk.value
            if risk_value in ("suspicious", "danger"):
                alerted: set[str] = session.setdefault("alerted_risks", set())
                if risk_value not in alerted:
                    alerted.add(risk_value)
                    alert_payload = {
                        "risk": risk_value,
                        "signals": [s.value for s in call_state.top_signals],
                        "excerpt": classification.transcript_excerpt or "",
                        "duration_seconds": time.time() - session["started_at"],
                    }
                    contacts = load_contacts(CONTACTS_PATH)
                    if contacts:
                        smtp_conf = load_smtp_config(SONAR_CONFIG_PATH)
                        threading.Thread(
                            target=send_alert,
                            args=(smtp_conf, contacts, alert_payload),
                            daemon=True,
                        ).start()
                        LOGGER.info("alert_triggered", risk=risk_value, contacts=len(contacts))
                    with _ALERT_LOG_LOCK:
                        _ALERT_LOG.append({
                            "id": uuid.uuid4().hex[:8],
                            "timestamp": time.time(),
                            **alert_payload,
                        })
                        if len(_ALERT_LOG) > 30:
                            _ALERT_LOG.pop(0)
        except Exception as exc:
            LOGGER.warning("classification_worker_error", segment_id=seg_id, error=str(exc))


def _process_call_chunk(session: dict[str, Any], audio_bytes: bytes, suffix: str) -> dict[str, Any] | None:
    """Transcribe one streaming chunk with Whisper and queue it for background Gemma classification.

    Returns None immediately — results arrive via /api/status/{session_id} polling.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        # In auto mode before detection, pass None so Whisper identifies the language
        if session.get("language_auto") and not session.get("language_locked"):
            stt_lang = None
        else:
            stt_lang = _WHISPER_LANG.get(session["language"].value, None)
        transcript_segs = transcribe_audio(
            tmp_path,
            language=stt_lang,
            model_size=session["whisper_model"],
            allow_transcript_fallback=False,
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    if not transcript_segs:
        return None
    combined_text = " ".join(s.text for s in transcript_segs).strip()
    if not combined_text:
        return None

    # Lock language after first successful auto-detection
    if session.get("language_auto") and not session.get("language_locked"):
        detected = transcript_segs[0].language_detected or ""
        mapped = _WHISPER_TO_LANG.get(detected)
        if mapped:
            session["language"] = mapped
            session["language_locked"] = True
            LOGGER.info("language_auto_detected", detected=detected, mapped=mapped.value)

    session["segment_count"] += 1
    seg_id = f"{session['call_id']}_seg_{session['segment_count']:03d}"
    session["transcript_log"].append({"seg_id": seg_id, "text": combined_text})
    session["classification_queue"].put((seg_id, combined_text, session["language"]))
    return {"queued": True}  # signals successful transcription, not silence


class SentiUiHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local-only desktop UI."""

    server_version = "SonarLocalUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path == "/":
            self._send_html(_index_html())
            return
        if parsed.path == "/api/history":
            self._send_json({"items": read_jsonl(FEEDBACK_PATH, limit=50)})
            return
        if parsed.path == "/monitor":
            self._send_html(_monitor_html())
            return
        if parsed.path == "/api/latest_job":
            # Prefer active live session over file-upload job
            with _SESSIONS_LOCK:
                active_sessions = [s for s in _SESSIONS.values() if s.get("status") == "active"]
                latest_session = max(active_sessions, key=lambda s: s["started_at"], default=None)
            if latest_session:
                sid = latest_session["session_id"]
                segs = list(latest_session.get("segments", []))
                self._send_json({
                    "ok": True,
                    "job_id": sid,
                    "status": "running",
                    "stage": "classifying",
                    "recording_id": latest_session.get("recording_id", ""),
                    "model_name": latest_session.get("model_name", ""),
                    "whisper_model": latest_session.get("whisper_model", ""),
                    "language": str(latest_session.get("language", "pt-BR").value if hasattr(latest_session.get("language"), "value") else latest_session.get("language", "")),
                    "total": latest_session.get("segment_count", 0),
                    "segment_count": len(segs),
                    "final": latest_session.get("final"),
                    "transcript_log": list(latest_session.get("transcript_log", [])),
                })
                return
            with _JOBS_LOCK:
                jid = _LAST_JOB_ID
                job: dict[str, Any] = dict(_JOBS.get(jid, {})) if jid else {}
                if job:
                    job["segments"] = list(job.get("segments", []))
            if not jid or not job:
                self._send_json({"ok": False, "job_id": None})
                return
            self._send_json({
                "ok": True,
                "job_id": jid,
                "status": job.get("status", "running"),
                "stage": job.get("stage", ""),
                "recording_id": job.get("recording_id", ""),
                "model_name": job.get("model_name", ""),
                "whisper_model": job.get("whisper_model", ""),
                "language": job.get("language", ""),
                "total": job.get("total", 0),
                "segment_count": len(job.get("segments", [])),
                "final": job.get("final"),
            })
            return
        if parsed.path.startswith("/api/status/"):
            entity_id = parsed.path[len("/api/status/"):]
            # Check sessions first (live calls use sess_ prefix)
            if entity_id.startswith("sess_"):
                with _SESSIONS_LOCK:
                    session = dict(_SESSIONS.get(entity_id, {}))
                if session:
                    segs = list(session.get("segments", []))
                    self._send_json({
                        "ok": True,
                        "status": "running" if session.get("status") == "active" else "done",
                        "stage": "classifying" if session.get("status") == "active" else "done",
                        "total": session.get("segment_count", 0),
                        "segments": segs,
                        "final": session.get("final"),
                        "recording_id": session.get("recording_id", ""),
                        "audio_suffix": ".webm",
                        "error": None,
                        "transcript_log": list(session.get("transcript_log", [])),
                        "language": session.get("language", Language.PT_BR).value if hasattr(session.get("language"), "value") else str(session.get("language", "")),
                    })
                    return
            with _JOBS_LOCK:
                job = dict(_JOBS.get(entity_id, {}))
                if job:
                    job["segments"] = list(job.get("segments", []))
            if not job:
                self._send_json({"ok": False, "error": "job_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({
                "ok": True,
                "status": job.get("status", "running"),
                "stage": job.get("stage", ""),
                "total": job.get("total", 0),
                "segments": job.get("segments", []),
                "final": job.get("final"),
                "recording_id": job.get("recording_id", ""),
                "audio_suffix": job.get("audio_suffix", ""),
                "error": job.get("error"),
            })
            return
        if parsed.path == "/contato":
            self._send_html(_contato_html())
            return
        if parsed.path == "/api/alert-log":
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            since = float(qs.get("since", ["0"])[0])
            with _ALERT_LOG_LOCK:
                entries = [a for a in _ALERT_LOG if a["timestamp"] > since]
            self._send_json({"ok": True, "alerts": entries})
            return
        if parsed.path == "/api/incoming-call":
            with _INCOMING_CALL_LOCK:
                ic = dict(_INCOMING_CALL) if _INCOMING_CALL else None
            if ic:
                self._send_json({"ringing": True, "caller_id": ic.get("caller_id", "Número desconhecido"), "config": ic.get("config", {})})
            else:
                self._send_json({"ringing": False})
            return
        if parsed.path == "/api/library":
            entries = get_library_entries(OUTPUT_DIR, FEEDBACK_PATH)
            self._send_json({"ok": True, "entries": entries})
            return
        if parsed.path.startswith("/api/export/"):
            recording_id = parsed.path[len("/api/export/"):]
            report_candidates = list((OUTPUT_DIR / recording_id).glob("*_call_report.json")) if (OUTPUT_DIR / recording_id).exists() else []
            if not report_candidates:
                self._send_json({"ok": False, "error": "report_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            data = json.loads(report_candidates[0].read_text(encoding="utf-8"))
            self._send_json(data)
            return
        if parsed.path == "/api/contacts":
            self._send_json({"ok": True, "contacts": load_contacts(CONTACTS_PATH)})
            return
        if parsed.path == "/api/smtp-config":
            cfg = load_smtp_config(SONAR_CONFIG_PATH)
            safe = {k: v for k, v in cfg.items() if k != "password"}
            safe["password_set"] = bool(cfg.get("password"))
            self._send_json({"ok": True, "smtp": safe})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze()
            return
        if parsed.path == "/api/feedback":
            self._handle_feedback()
            return
        if parsed.path == "/api/add_to_dataset":
            self._handle_add_to_dataset()
            return
        if parsed.path == "/api/call/start":
            self._handle_call_start()
            return
        if parsed.path.startswith("/api/call/") and parsed.path.endswith("/chunk"):
            session_id = parsed.path.split("/")[3]
            self._handle_call_chunk(session_id)
            return
        if parsed.path.startswith("/api/call/") and parsed.path.endswith("/end"):
            session_id = parsed.path.split("/")[3]
            self._handle_call_end(session_id)
            return
        if parsed.path == "/api/test/ring":
            self._handle_test_ring()
            return
        if parsed.path == "/api/call/answer":
            self._handle_call_answer()
            return
        if parsed.path == "/api/call/reject":
            self._handle_call_reject()
            return
        if parsed.path == "/api/test/alert":
            self._handle_test_alert()
            return
        if parsed.path == "/api/contacts":
            self._handle_add_contact()
            return
        if parsed.path == "/api/smtp-config":
            self._handle_save_smtp()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        # DELETE /api/contacts/{id}
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "contacts":
            contacts = remove_contact(CONTACTS_PATH, parts[2])
            self._send_json({"ok": True, "contacts": contacts})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        # /api/library/{recording_id}/label
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "library" and parts[3] == "label":
            self._handle_library_label(parts[2])
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[sonar-ui] " + fmt % args + "\n")

    def _handle_analyze(self) -> None:
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
        audio_item = form["audio"] if "audio" in form else None
        if audio_item is None or not getattr(audio_item, "filename", None):
            self._send_json({"ok": False, "error": "missing_audio_file"}, status=HTTPStatus.BAD_REQUEST)
            return

        language_str = _field_value(form, "language", "pt-BR")
        model = _field_value(form, "model", "google/gemma-4-e2b")
        whisper_model = _field_value(form, "whisper_model", "tiny")
        allow_fallback = _field_value(form, "allow_transcript_fallback", "false") == "true"

        recording_id = new_recording_id()
        raw_filename = str(audio_item.filename or "recording.webm")
        suffix = Path(raw_filename).suffix.lower() or ".webm"
        audio_path = INPUT_DIR / f"{recording_id}{suffix}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio_item.file.read())

        job_id = f"job_{uuid.uuid4().hex[:16]}"
        global _LAST_JOB_ID
        with _JOBS_LOCK:
            _LAST_JOB_ID = job_id
            _JOBS[job_id] = {
                "status": "running",
                "stage": "starting",
                "total": 0,
                "segments": [],
                "final": None,
                "error": None,
                "recording_id": recording_id,
                "audio_suffix": suffix,
                "model_name": model,
                "whisper_model": whisper_model,
                "language": language_str,
            }

        threading.Thread(
            target=_run_analysis_job,
            args=(job_id, recording_id, audio_path, suffix, Language(language_str), model, whisper_model, allow_fallback),
            daemon=True,
        ).start()

        self._send_json({"ok": True, "job_id": job_id, "recording_id": recording_id, "audio_suffix": suffix})

    def _handle_feedback(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return
        saved = save_feedback(FEEDBACK_PATH, payload)
        self._send_json({"ok": True, "feedback": saved})

    def _handle_add_to_dataset(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return

        recording_id = payload.get("recording_id", "")
        user_label = payload.get("user_label", "uncertain")
        user_corrected_risk = payload.get("user_corrected_risk", "safe")
        language = payload.get("language", "pt-BR")
        notes = payload.get("notes", "")

        audio_path: Path | None = None
        for ext in (".webm", ".wav", ".mp3", ".m4a", ".ogg"):
            candidate = INPUT_DIR / f"{recording_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if audio_path is None:
            self._send_json(
                {"ok": False, "error": f"audio not found for recording_id={recording_id}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        report_dir = OUTPUT_DIR / recording_id
        report_candidates = list(report_dir.glob("*_call_report.json")) if report_dir.exists() else []
        report_path = report_candidates[0] if report_candidates else None

        try:
            entry = add_to_dataset(
                audio_src=audio_path,
                report_path=report_path,
                dataset_audio_dir=DATASET_AUDIO_DIR,
                ground_truth_path=GROUND_TRUTH_PATH,
                recording_id=recording_id,
                user_label=user_label,
                user_corrected_risk=user_corrected_risk,
                language=language,
                notes=notes,
            )
            self._send_json({"ok": True, "entry": entry})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_call_start(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        data: dict[str, Any] = {}
        if length:
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                pass
        language_str = data.get("language", "pt-BR")
        model = data.get("model", "google/gemma-4-e2b")
        whisper_model_name = data.get("whisper_model", "tiny")
        language_auto = language_str == "auto"
        if language_auto:
            language = Language.PT_BR  # temporary default until Whisper detects
        else:
            try:
                language = Language(language_str)
            except ValueError:
                language = Language.PT_BR
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        recording_id = new_recording_id()
        few_shot = get_few_shot_examples(OUTPUT_DIR, FEEDBACK_PATH, language_str if not language_auto else "pt-BR")
        classifier_inst = GemmaClassifier(model_name=model, prompt_path=PROMPT_PATH, few_shot_context=few_shot)
        engine_inst = DecisionEngine(REPO_ROOT / "shared" / "signals_taxonomy.yaml")
        engine_inst.begin_call(call_id)
        cls_queue: queue.Queue = queue.Queue()
        session_data: dict[str, Any] = {
            "session_id": session_id,
            "call_id": call_id,
            "recording_id": recording_id,
            "language": language,
            "language_auto": language_auto,
            "language_locked": False,
            "model_name": model,
            "whisper_model": whisper_model_name,
            "classifier": classifier_inst,
            "engine": engine_inst,
            "classifications": [],
            "history_summary": None,
            "segment_count": 0,
            "status": "active",
            "started_at": time.time(),
            "segments": [],
            "final": None,
            "audio_chunks": [],
            "alerted_risks": set(),
            "classification_queue": cls_queue,
            "transcript_log": [],
        }
        with _SESSIONS_LOCK:
            _SESSIONS[session_id] = session_data
        threading.Thread(target=_classification_worker, args=(session_data,), daemon=True).start()
        self._send_json({"ok": True, "session_id": session_id, "call_id": call_id, "recording_id": recording_id})

    def _handle_call_chunk(self, session_id: str) -> None:
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(session_id)
        if not session or session.get("status") != "active":
            self._send_json({"ok": False, "error": "session_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
        audio_item = form["audio"] if "audio" in form else None
        if audio_item is None:
            self._send_json({"ok": True, "segment": None, "silence": True})
            return
        audio_bytes = audio_item.file.read()
        raw_filename = str(getattr(audio_item, "filename", None) or "chunk.webm")
        suffix = Path(raw_filename).suffix.lower() or ".webm"
        session["audio_chunks"].append(audio_bytes)
        try:
            segment = _process_call_chunk(session, audio_bytes, suffix)
        except Exception as exc:
            LOGGER.warning("call_chunk_error", session_id=session_id, error=str(exc))
            sys.stderr.write(f"[sonar-ui] chunk error: {exc}\n")
            self._send_json({"ok": True, "segment": None, "error": str(exc)})
            return
        silence = segment is None  # None = no speech; {"queued": True} = transcribed
        self._send_json({"ok": True, "segment": None, "silence": silence})

    def _handle_call_end(self, session_id: str) -> None:
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(session_id)
        if not session:
            self._send_json({"ok": False, "error": "session_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        session["status"] = "ended"
        cls_q = session.get("classification_queue")
        if cls_q is not None:
            cls_q.put(None)  # sentinel — stop the background worker
        if session["classifications"]:
            final_state = session["engine"].end_call()
        else:
            from shared.schemas.classification import AlertLevel, CallState, RiskLevel
            final_state = CallState(
                call_id=session["call_id"],
                overall_risk=RiskLevel.SAFE,
                top_signals=[],
                alert_level=AlertLevel.NONE,
                should_notify_family=False,
                should_play_audio_alert=False,
                rationale_for_user="Nenhum trecho foi transcrito para análise.",
                rationale_for_audit_log="No segments transcribed.",
            )
        final: dict[str, Any] = {
            "overall_risk": final_state.overall_risk.value,
            "alert_level": final_state.alert_level.value,
            "top_signals": [s.value for s in final_state.top_signals],
            "rationale_for_user": final_state.rationale_for_user,
            "should_notify_family": final_state.should_notify_family,
            "should_play_audio_alert": final_state.should_play_audio_alert,
        }
        session["final"] = final
        recording_id = session["recording_id"]
        call_report_path = None
        if session["classifications"]:
            report = CallReport(
                call_id=session["call_id"],
                language=session["language"],
                duration_seconds=round(time.time() - session["started_at"], 1),
                segments=session["classifications"],
                final_state=final_state,
                model_versions={"classifier": session["model_name"], "stt": f"whisper-{session['whisper_model']}"},
            )
            call_report_path = OUTPUT_DIR / recording_id / f"{session['call_id']}_call_report.json"
            call_report_path.parent.mkdir(parents=True, exist_ok=True)
            call_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            audio_chunks = session.get("audio_chunks", [])
            if audio_chunks:
                audio_save_path = INPUT_DIR / f"{recording_id}.webm"
                audio_save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(audio_save_path, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
        self._send_json({
            "ok": True,
            "recording_id": recording_id,
            "call_id": session["call_id"],
            "final": final,
            "segments": session["segments"],
            "report_saved": call_report_path is not None,
            "duration_seconds": round(time.time() - session["started_at"], 1),
        })

    def _handle_add_contact(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return
        name = data.get("name", "").strip()
        if not name:
            self._send_json({"ok": False, "error": "name required"}, status=HTTPStatus.BAD_REQUEST)
            return
        contacts = add_contact(CONTACTS_PATH, name, data.get("phone", ""), data.get("email", ""))
        self._send_json({"ok": True, "contacts": contacts})

    def _handle_save_smtp(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return
        save_smtp_config(SONAR_CONFIG_PATH, data)
        self._send_json({"ok": True})

    def _handle_test_ring(self) -> None:
        global _INCOMING_CALL
        length = int(self.headers.get("Content-Length", "0"))
        data: dict[str, Any] = {}
        if length:
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                pass
        with _INCOMING_CALL_LOCK:
            _INCOMING_CALL = {
                "caller_id": data.get("caller_id", "0800 555-1234"),
                "started_at": time.time(),
                "config": {
                    "language": data.get("language", "pt-BR"),
                    "model": data.get("model", "google/gemma-4-e2b"),
                    "whisper_model": data.get("whisper_model", "tiny"),
                },
            }
        self._send_json({"ok": True, "caller_id": _INCOMING_CALL["caller_id"]})

    def _handle_call_answer(self) -> None:
        global _INCOMING_CALL
        with _INCOMING_CALL_LOCK:
            ic = dict(_INCOMING_CALL) if _INCOMING_CALL else None
            _INCOMING_CALL = None
        config = (ic or {}).get("config", {})
        language_str = config.get("language", "pt-BR")
        model = config.get("model", "google/gemma-4-e2b")
        whisper_model_name = config.get("whisper_model", "tiny")
        language_auto = language_str == "auto"
        if language_auto:
            language = Language.PT_BR
        else:
            try:
                language = Language(language_str)
            except ValueError:
                language = Language.PT_BR
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        recording_id = new_recording_id()
        few_shot = get_few_shot_examples(OUTPUT_DIR, FEEDBACK_PATH, language_str if not language_auto else "pt-BR")
        classifier_inst = GemmaClassifier(model_name=model, prompt_path=PROMPT_PATH, few_shot_context=few_shot)
        engine_inst = DecisionEngine(REPO_ROOT / "shared" / "signals_taxonomy.yaml")
        engine_inst.begin_call(call_id)
        cls_queue: queue.Queue = queue.Queue()
        session_data: dict[str, Any] = {
            "session_id": session_id,
            "call_id": call_id,
            "recording_id": recording_id,
            "language": language,
            "language_auto": language_auto,
            "language_locked": False,
            "model_name": model,
            "whisper_model": whisper_model_name,
            "classifier": classifier_inst,
            "engine": engine_inst,
            "classifications": [],
            "history_summary": None,
            "segment_count": 0,
            "status": "active",
            "started_at": time.time(),
            "segments": [],
            "final": None,
            "audio_chunks": [],
            "alerted_risks": set(),
            "classification_queue": cls_queue,
            "transcript_log": [],
        }
        with _SESSIONS_LOCK:
            _SESSIONS[session_id] = session_data
        threading.Thread(target=_classification_worker, args=(session_data,), daemon=True).start()
        self._send_json({"ok": True, "session_id": session_id, "call_id": call_id, "recording_id": recording_id})

    def _handle_call_reject(self) -> None:
        global _INCOMING_CALL
        with _INCOMING_CALL_LOCK:
            _INCOMING_CALL = None
        self._send_json({"ok": True})

    def _handle_test_alert(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            data = {}
        risk = data.get("risk", "danger")
        signals = data.get("signals", ["financial_request", "urgency_pressure", "authority_claim"])
        excerpt = data.get("excerpt", "Me passou o código? Precisa ser agora, a conta vai bloquear em 5 minutos.")
        entry = {
            "id": uuid.uuid4().hex[:8],
            "timestamp": time.time(),
            "risk": risk,
            "signals": signals,
            "excerpt": excerpt,
            "duration_seconds": data.get("duration_seconds", 47),
        }
        with _ALERT_LOG_LOCK:
            _ALERT_LOG.append(entry)
            if len(_ALERT_LOG) > 30:
                _ALERT_LOG.pop(0)
        self._send_json({"ok": True, "alert": entry})

    def _handle_library_label(self, recording_id: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return
        label = data.get("label", "uncertain")
        saved = save_feedback(FEEDBACK_PATH, {"recording_id": recording_id, "user_label": label, "source": "library_label"})
        self._send_json({"ok": True, "feedback": saved})

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Sonar local desktop UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SentiUiHandler)
    print(f"Sonar UI          →  http://{args.host}:{args.port}")
    print(f"Pipeline Monitor  →  http://{args.host}:{args.port}/monitor")
    print(f"Demo Contato      →  http://{args.host}:{args.port}/contato")
    server.serve_forever()
    return 0


def _field_value(form: cgi.FieldStorage, name: str, default: str) -> str:
    value = form.getfirst(name, default)
    return str(value) if value is not None else default


def _index_html() -> str:
    return r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sonar</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  min-height:100vh;
  background:linear-gradient(160deg,#E8EDF4 0%,#DDE4EF 100%);
  display:flex;align-items:center;justify-content:center;
  padding:40px 20px;
  font-family:-apple-system,BlinkMacSystemFont,'Inter','SF Pro Text',system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;color:#101828;
}

/* ── Phone shell ─────────────────────────────────────────────── */
.phone{
  position:relative;width:390px;
  background:#1C1C1E;
  border-radius:54px;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.06),
    0 0 0 3px #0A0A0B,
    0 60px 120px rgba(0,0,0,.28),
    0 24px 48px rgba(0,0,0,.16);
  padding:14px;flex-shrink:0;
}
.phone::before{content:'';position:absolute;right:-4px;top:132px;width:4px;height:70px;background:#2C2C2E;border-radius:0 3px 3px 0}
.phone::after{content:'';position:absolute;left:-4px;top:92px;width:4px;height:38px;background:#2C2C2E;border-radius:3px 0 0 3px;box-shadow:0 56px 0 #2C2C2E}

/* ── Screen ──────────────────────────────────────────────────── */
.screen{
  background:#F2F2F7;border-radius:42px;overflow:hidden;
  height:800px;display:flex;flex-direction:column;position:relative;
}

/* ── Dynamic island ──────────────────────────────────────────── */
.island{position:absolute;top:12px;left:50%;transform:translateX(-50%);width:126px;height:37px;background:#1C1C1E;border-radius:999px;z-index:10}

/* ── Status bar ──────────────────────────────────────────────── */
.statusbar{height:58px;display:flex;align-items:flex-end;padding:0 26px 10px;flex-shrink:0}
.sb-time{font-size:15px;font-weight:600;color:#1C1C1E}
.sb-icons{margin-left:auto;display:flex;align-items:center;gap:5px}

/* ── App bar ─────────────────────────────────────────────────── */
.appbar{
  background:rgba(255,255,255,.85);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(0,0,0,.07);
  padding:10px 20px 12px;
  display:flex;align-items:center;gap:10px;flex-shrink:0;
}
.brand{display:flex;align-items:center;gap:9px}
.brand-name{font-size:18px;font-weight:700;letter-spacing:-.04em;color:#101828}
.brand-tagline{font-size:11px;color:#98A2B3;margin-top:1px}

/* ── Home indicator ──────────────────────────────────────────── */
.home-ind{height:28px;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:#F2F2F7}
.home-pill{width:134px;height:5px;background:rgba(0,0,0,.14);border-radius:999px}

/* ── Main content area ───────────────────────────────────────── */
#main-content{flex:1;overflow:hidden;display:flex;flex-direction:column;position:relative}
.view-panel{flex:1;overflow-y:auto;display:flex;flex-direction:column;scrollbar-width:none}
.view-panel::-webkit-scrollbar{display:none}

/* ── Bottom tab bar ──────────────────────────────────────────── */
#tab-bar{display:flex;border-top:1px solid #EAECF0;background:#fff;flex-shrink:0}
.tab-btn{flex:1;padding:10px 0 8px;font-size:10px;color:#6B7280;background:none;border:none;cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:3px;font-family:inherit}
.tab-btn.active{color:#F97316}
.tab-btn svg{width:20px;height:20px}

/* ── Cards ───────────────────────────────────────────────────── */
.card{background:white;border-radius:14px;padding:14px;border:1px solid #EAECF0;box-shadow:0 1px 2px rgba(16,24,40,.04)}

/* ── Risk bar ────────────────────────────────────────────────── */
.risk-bar{padding:10px 16px;border-radius:10px;font-weight:700;font-size:13px;letter-spacing:.5px;text-align:center;transition:all .3s;}
.risk-bar.safe{background:#ECFDF3;color:#166534;}
.risk-bar.suspicious{background:#FFFAEB;color:#92400E;}
.risk-bar.danger{background:#FEF2F2;color:#991B1B;animation:pulse-danger 1s infinite;}
@keyframes pulse-danger{0%,100%{opacity:1}50%{opacity:.7}}

/* ── Waveform ────────────────────────────────────────────────── */
#waveform{width:100%;height:80px;background:#F9FAFB;border-radius:10px;display:block;}

/* ── Segment cards ───────────────────────────────────────────── */
.seg-card{background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:8px;border-left:3px solid #ccc;animation:slideIn .3s cubic-bezier(.22,.68,0,1.2)}
.seg-card.seg-safe{border-left-color:#17B26A;}
.seg-card.seg-suspicious{border-left-color:#F79009;}
.seg-card.seg-danger{border-left-color:#F04438;}
@keyframes slideIn{from{transform:translateX(16px);opacity:0}to{transform:translateX(0);opacity:1}}
.seg-header{display:flex;align-items:flex-start;gap:8px;margin-bottom:4px;}
.seg-risk-badge{font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;white-space:nowrap;flex-shrink:0;}
.risk-safe{background:#ECFDF3;color:#166534;}
.risk-suspicious{background:#FFFAEB;color:#92400E;}
.risk-danger{background:#FEF2F2;color:#991B1B;}
.seg-excerpt{font-size:12px;color:#374151;line-height:1.4;}
.seg-signals{font-size:11px;color:#6B7280;margin-bottom:3px;}
.seg-action{font-size:11px;color:#F97316;font-style:italic;}

/* ── Library ─────────────────────────────────────────────────── */
.lib-entry{background:#fff;border-radius:12px;padding:14px;margin-bottom:10px;}
.lib-meta{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;}
.lib-date{font-size:12px;color:#6B7280;}
.lib-dur{font-size:12px;color:#9CA3AF;}
.risk-badge-sm{font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;}
.lib-signals{font-size:11px;color:#6B7280;margin-bottom:8px;}
.lib-actions{display:flex;gap:6px;flex-wrap:wrap;}
.label-btn{font-size:11px;padding:4px 10px;border-radius:6px;border:1.5px solid #E5E7EB;background:#fff;color:#374151;cursor:pointer;font-family:inherit;}
.label-btn.active-label{background:#F97316;color:#fff;border-color:#F97316;}
.export-btn{border-color:#F97316;color:#F97316;}
.lib-empty{color:#9CA3AF;font-size:13px;text-align:center;padding:30px;}

/* ── Timer row ───────────────────────────────────────────────── */
#call-timer-row{display:flex;align-items:center;gap:8px;padding:8px 16px;}
.call-live-dot{width:8px;height:8px;border-radius:50%;background:#EF4444;animation:blink 1s infinite;flex-shrink:0;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
#call-timer{font-size:14px;font-weight:700;color:#101828;font-variant-numeric:tabular-nums;}

/* ── Pre-call ────────────────────────────────────────────────── */
#pre-call{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:20px;padding:30px 20px;text-align:center;}
.tagline{font-size:13px;color:#6B7280;line-height:1.5;max-width:200px;}
.btn-primary-call{background:#F97316;color:#fff;border:none;border-radius:14px;padding:16px 32px;font-size:16px;font-weight:600;cursor:pointer;width:100%;font-family:inherit;}
.btn-secondary-link{font-size:12px;color:#F97316;background:none;border:none;cursor:pointer;text-decoration:underline;font-family:inherit;}

/* ── Incoming call overlay ───────────────────────────────────── */
#incoming-overlay{position:absolute;inset:0;background:#101828;z-index:50;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;padding:30px;border-radius:42px;}
.inc-label{font-size:11px;font-weight:700;letter-spacing:.12em;color:#6B7280;text-transform:uppercase;}
.inc-caller{font-size:26px;font-weight:700;color:#fff;font-variant-numeric:tabular-nums;text-align:center;}
.inc-sub{font-size:12px;color:#4B5563;margin-top:-18px;}
.ring-wrap{position:relative;width:96px;height:96px;display:flex;align-items:center;justify-content:center;}
.ring-pulse{position:absolute;border-radius:50%;border:2px solid rgba(249,115,22,.5);animation:rpulse 2s ease-out infinite;}
.rp1{width:96px;height:96px;animation-delay:0s;}
.rp2{width:96px;height:96px;animation-delay:.65s;}
.rp3{width:96px;height:96px;animation-delay:1.3s;}
@keyframes rpulse{0%{opacity:.7;transform:scale(.85)}100%{opacity:0;transform:scale(2.1)}}
.ring-center{position:absolute;width:72px;height:72px;border-radius:50%;background:#F97316;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 4px rgba(249,115,22,.2);}
.inc-btns{display:flex;gap:36px;margin-top:8px;}
.inc-btn-wrap{display:flex;flex-direction:column;align-items:center;gap:8px;}
.inc-btn{width:68px;height:68px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:transform .15s,opacity .15s;}
.inc-btn:active{transform:scale(.92);opacity:.8}
.btn-reject{background:#EF4444;}
.btn-answer{background:#F97316;}
.inc-btn-lbl{font-size:12px;color:#9CA3AF;}

/* ── Active call ─────────────────────────────────────────────── */
#active-call{display:flex;flex-direction:column;height:100%;}
.call-header{padding:12px 16px;display:flex;align-items:center;justify-content:space-between;}
.call-info{font-size:12px;color:#6B7280;}
.call-number{font-size:15px;font-weight:600;color:#111827;}
#live-segments{flex:1;overflow-y:auto;padding:8px 16px;max-height:300px;scrollbar-width:thin;scrollbar-color:#EAECF0 transparent;}
.call-footer{padding:12px 16px;}
.btn-end-call{background:#EF4444;color:#fff;border:none;border-radius:14px;padding:14px;font-size:15px;font-weight:600;cursor:pointer;width:100%;font-family:inherit;}

/* ── File upload ─────────────────────────────────────────────── */
.upload-panel{display:grid;gap:10px;padding:14px 0 8px;}
.file-label{display:block;border:1.5px dashed #D0D5DD;border-radius:10px;padding:22px;text-align:center;font-size:12px;color:#98A2B3;cursor:pointer;transition:.15s;}
.file-label:hover{border-color:#F97316;color:#F97316}
.file-label svg{display:block;margin:0 auto 8px}
.btn-primary{width:100%;border:0;border-radius:10px;background:#F97316;color:white;padding:12px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s;display:flex;align-items:center;justify-content:center;gap:7px;font-family:inherit;}
.btn-primary:hover:not(:disabled){background:#EA580C}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-secondary{width:100%;border:1.5px solid #F97316;color:#C2410C;background:transparent;border-radius:8px;padding:9px;font:inherit;font-weight:600;cursor:pointer;font-size:12px;margin-top:6px;display:flex;align-items:center;justify-content:center;gap:6px;transition:.15s;}
.btn-secondary:hover{background:#FFF7ED}

/* ── Status ──────────────────────────────────────────────────── */
.status-wrap{display:flex;flex-direction:column;gap:8px}
.status-label{font-size:13px;font-weight:500;color:#344054}
.progress-bar{height:3px;background:#EAECF0;border-radius:99px;overflow:hidden}
.progress-bar span{display:block;height:100%;width:35%;background:#F97316;border-radius:99px;animation:progAnim 1.1s infinite alternate}
@keyframes progAnim{from{margin-left:0}to{margin-left:65%}}

/* ── Segment stream (upload analysis) ───────────────────────── */
.seg-stream{display:flex;flex-direction:column;gap:8px}
.seg-hdr{display:flex;align-items:center;gap:6px;margin-bottom:8px}
.seg-num{font-size:10px;font-weight:700;color:#98A2B3;text-transform:uppercase;letter-spacing:.06em}
.risk-tag{font-size:10px;font-weight:700;text-transform:uppercase;padding:2px 8px;border-radius:999px;letter-spacing:.04em}
.risk-tag.safe{background:#DCFCE7;color:#166534}
.risk-tag.suspicious{background:#FEF9C3;color:#713F12}
.risk-tag.danger{background:#FEE2E2;color:#991B1B}
.conf{font-size:10px;color:#D0D5DD;margin-left:auto;font-variant-numeric:tabular-nums}
.sig-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.sig-chip{font-size:10px;font-weight:600;padding:2px 7px;border-radius:999px}
.chip-critical{background:#FEE2E2;color:#9B1C1C}
.chip-high{background:#FFEDD5;color:#9A3412}
.chip-medium{background:#FEF9C3;color:#713F12}
.t-text{font-size:13px;color:#344054;line-height:1.6;word-break:break-word;margin-bottom:6px}
.a-tip{font-size:11px;color:#98A2B3;padding-top:7px;margin-top:4px;border-top:1px solid #F2F4F7;line-height:1.5}
mark.hl-critical{background:#FEE2E2;color:#9B1C1C;border-radius:3px;padding:0 2px;text-decoration:underline;text-decoration-color:#F87171;text-decoration-thickness:2px}
mark.hl-high{background:#FFEDD5;color:#9A3412;border-radius:3px;padding:0 2px;text-decoration:underline;text-decoration-color:#FB923C;text-decoration-thickness:2px}
mark.hl-medium{background:#FEF9C3;color:#713F12;border-radius:3px;padding:0 2px;text-decoration:underline;text-decoration-color:#FBBF24;text-decoration-thickness:2px}

/* ── Alert banner ────────────────────────────────────────────── */
.alert-banner{display:flex;align-items:center;gap:9px;padding:9px 20px;background:white;border-bottom:1px solid #EAECF0;transition:background .35s,border-color .35s;flex-shrink:0;min-height:40px;}
.alert-banner.safe{background:#ECFDF3;border-color:#ABEFC6}
.alert-banner.suspicious{background:#FFFAEB;border-color:#FEDF89}
.alert-banner.danger{background:#FEF3F2;border-color:#FEC9C7;animation:bPulse 2.5s infinite}
.alert-banner.running{background:#FFF7ED;border-color:#FED7AA}
@keyframes bPulse{0%,100%{background:#FEF3F2}50%{background:#FEE4E2}}
.a-dot{width:8px;height:8px;border-radius:50%;background:#D0D5DD;transition:background .35s;flex-shrink:0}
.alert-banner.safe .a-dot{background:#17B26A}
.alert-banner.suspicious .a-dot{background:#F79009}
.alert-banner.danger .a-dot{background:#F04438;animation:dPulse 1.2s infinite}
.alert-banner.running .a-dot{background:#F97316;animation:dPulse 1.6s infinite}
@keyframes dPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.55);opacity:.5}}
.a-text{font-size:12px;font-weight:500;color:#344054;line-height:1.4;flex:1}
.alert-banner.safe .a-text{color:#067647}
.alert-banner.suspicious .a-text{color:#B54708}
.alert-banner.danger .a-text{color:#B42318}
.alert-banner.running .a-text{color:#F97316}

/* ── Config form ─────────────────────────────────────────────── */
.cfg-body{display:grid;gap:10px;padding:10px 0 4px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
label{display:block;font-size:11px;font-weight:600;color:#344054;margin-bottom:4px}
select,input[type=text]{width:100%;background:white;border:1px solid #D0D5DD;border-radius:8px;padding:8px 10px;font:inherit;font-size:13px;color:#101828;outline:none;}
select:focus,input[type=text]:focus{border-color:#F97316;box-shadow:0 0 0 3px rgba(249,115,22,.1)}
.chk{display:flex;align-items:center;gap:7px;font-size:12px;color:#667085;cursor:pointer}
.chk input{accent-color:#F97316}
.sec-label{font-size:11px;font-weight:600;color:#667085;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;display:flex;align-items:center;gap:6px}

/* ── Feedback ────────────────────────────────────────────────── */
.fb-select,.fb-area{width:100%;background:white;border:1px solid #D0D5DD;border-radius:8px;padding:8px 10px;font:inherit;font-size:13px;color:#101828;margin-bottom:10px;outline:none;}
.fb-select:focus,.fb-area:focus{border-color:#F97316;box-shadow:0 0 0 3px rgba(249,115,22,.1)}
.fb-area{resize:vertical}
.fb-saved{font-size:12px;color:#067647;font-weight:600;text-align:center;padding:5px;display:flex;align-items:center;justify-content:center;gap:5px}
.dataset-ok{font-size:12px;color:#067647;font-weight:600;text-align:center;padding:5px}

/* ── History ─────────────────────────────────────────────────── */
.hist-list{max-height:150px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;margin-top:8px;scrollbar-width:thin;scrollbar-color:#EAECF0 transparent}
.hist-item{font-size:11px;border-left:2px solid #EAECF0;padding:5px 10px;color:#667085;background:#F9FAFB;border-radius:0 6px 6px 0;line-height:1.4}
.hist-item.scam{border-left-color:#F04438;background:#FEF3F2}
.hist-item.legitimate{border-left-color:#17B26A;background:#F6FEF9}

.hide{display:none!important}

/* ── View headers ────────────────────────────────────────────── */
.view-header{padding:14px 16px 8px;font-size:17px;font-weight:700;color:#101828;flex-shrink:0;}
.view-scroll{flex:1;overflow-y:auto;padding:0 14px 14px;display:flex;flex-direction:column;gap:10px;scrollbar-width:none;}
.view-scroll::-webkit-scrollbar{display:none}

/* ── Contacts ────────────────────────────────────────────────── */
.contact-row{display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid #F2F4F7;}
.contact-row:last-child{border-bottom:none}
.contact-info{flex:1;min-width:0}
.contact-name{font-size:13px;font-weight:600;color:#101828;}
.contact-detail{font-size:11px;color:#6B7280;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.btn-del-contact{background:none;border:none;cursor:pointer;color:#D1D5DB;padding:4px;border-radius:6px;line-height:0;transition:.15s;}
.btn-del-contact:hover{color:#EF4444;background:#FEF2F2;}
.contact-add-form{display:grid;gap:7px;margin-top:10px;padding-top:10px;border-top:1px solid #F2F4F7;}
.inp-sm{width:100%;background:white;border:1px solid #D0D5DD;border-radius:8px;padding:7px 9px;font:inherit;font-size:12px;color:#101828;outline:none;}
.inp-sm:focus{border-color:#F97316;box-shadow:0 0 0 3px rgba(249,115,22,.1)}
.btn-add-contact{width:100%;border:1.5px solid #F97316;color:#F97316;background:transparent;border-radius:8px;padding:8px;font:inherit;font-weight:600;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;gap:5px;transition:.15s;}
.btn-add-contact:hover{background:#FFF7ED}
.alert-log{max-height:100px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;margin-top:6px;scrollbar-width:thin;scrollbar-color:#EAECF0 transparent;}
.alert-log-item{font-size:11px;color:#667085;background:#F9FAFB;border-radius:6px;padding:4px 8px;border-left:2px solid #F97316;}
.smtp-row{display:grid;grid-template-columns:1fr 1fr;gap:7px;}
.smtp-status{font-size:11px;color:#6B7280;margin-top:2px;}
.smtp-status.ok{color:#067647;}
.lang-toggle{margin-left:auto;display:flex;gap:0;border:1.5px solid #E5E7EB;border-radius:8px;overflow:hidden}
.lang-btn{font-size:11px;font-weight:700;padding:4px 10px;border:none;background:transparent;color:#98A2B3;cursor:pointer;font-family:inherit;transition:.15s;letter-spacing:.04em}
.lang-btn.active{background:#F97316;color:#fff}
</style>
</head>
<body>
<div class="phone">
  <div class="screen">
    <div class="island"></div>

    <!-- Status bar -->
    <div class="statusbar">
      <span class="sb-time">9:41</span>
      <div class="sb-icons">
        <svg width="17" height="12" viewBox="0 0 17 12" fill="none">
          <rect x="0" y="4" width="3" height="8" rx="1" fill="#1C1C1E" opacity=".28"/>
          <rect x="4.5" y="2.5" width="3" height="9.5" rx="1" fill="#1C1C1E" opacity=".48"/>
          <rect x="9" y="0" width="3" height="12" rx="1" fill="#1C1C1E" opacity=".68"/>
          <rect x="13.5" y="0" width="3.5" height="12" rx="1" fill="#1C1C1E"/>
        </svg>
        <svg width="16" height="12" viewBox="0 0 16 12" fill="#1C1C1E" opacity=".8">
          <path d="M8 3C10.2 3 12.2 3.9 13.6 5.4L15.1 3.7C13.3 1.9 10.8.9 8 .9S2.7 1.9.9 3.7l1.5 1.7C3.8 3.9 5.8 3 8 3zM8 6.2c1.3 0 2.4.5 3.3 1.3l1.5-1.7C11.5 4.6 9.8 4 8 4S4.5 4.6 3.2 5.8L4.7 7.5C5.6 6.7 6.7 6.2 8 6.2zM8 9.5c.7 0 1.3.3 1.8.7L8 12 6.2 10.2C6.7 9.8 7.3 9.5 8 9.5z"/>
        </svg>
        <svg width="26" height="13" viewBox="0 0 26 13" fill="none">
          <rect x=".5" y=".5" width="22" height="12" rx="3.5" stroke="#1C1C1E" stroke-opacity=".3"/>
          <rect x="1.5" y="1.5" width="18" height="10" rx="2.5" fill="#1C1C1E"/>
          <path d="M24 4.5v4a2.3 2.3 0 000-4z" fill="#1C1C1E" opacity=".35"/>
        </svg>
      </div>
    </div>

    <!-- App bar -->
    <div class="appbar">
      <div class="brand">
        <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
          <circle cx="15" cy="15" r="13.5" stroke="#F97316" stroke-width="1.5" opacity=".15"/>
          <circle cx="15" cy="15" r="9" stroke="#F97316" stroke-width="1.5" opacity=".38"/>
          <circle cx="15" cy="15" r="4.5" stroke="#F97316" stroke-width="1.5" opacity=".7"/>
          <circle cx="15" cy="15" r="2" fill="#F97316"/>
        </svg>
        <div>
          <div class="brand-name">Sonar</div>
          <div class="brand-tagline" data-i18n="tagline">Detecção em tempo real</div>
        </div>
      </div>
      <div class="lang-toggle">
        <button id="lang-pt" class="lang-btn active" onclick="applyLang('pt')">PT</button>
        <button id="lang-en" class="lang-btn" onclick="applyLang('en')">EN</button>
      </div>
    </div>

    <!-- Incoming call overlay (hidden until ring event) -->
    <div id="incoming-overlay" class="hide">
      <div class="inc-label" data-i18n="inc_label">Ligação recebida</div>
      <div class="inc-caller" id="inc-caller-id">0800 555-1234</div>
      <div class="inc-sub" data-i18n="inc_sub">Número externo</div>
      <div class="ring-wrap">
        <div class="ring-pulse rp1"></div>
        <div class="ring-pulse rp2"></div>
        <div class="ring-pulse rp3"></div>
        <div class="ring-center">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="white"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1-9.4 0-17-7.6-17-17 0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1L6.6 10.8z"/></svg>
        </div>
      </div>
      <div class="inc-btns">
        <div class="inc-btn-wrap">
          <button class="inc-btn btn-reject" onclick="rejectCall()">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="white"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
          </button>
          <span class="inc-btn-lbl" data-i18n="reject">Rejeitar</span>
        </div>
        <div class="inc-btn-wrap">
          <button class="inc-btn btn-answer" onclick="answerCall()">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="white"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1-9.4 0-17-7.6-17-17 0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1L6.6 10.8z"/></svg>
          </button>
          <span class="inc-btn-lbl" data-i18n="answer">Atender</span>
        </div>
      </div>
    </div>

    <!-- Main content area -->
    <div id="main-content">

      <!-- ── VIEW: Chamada ──────────────────────────────────────── -->
      <div id="view-call" class="view-panel">

        <!-- Pre-call state -->
        <div id="pre-call">
          <svg width="64" height="64" viewBox="0 0 30 30" fill="none">
            <circle cx="15" cy="15" r="13.5" stroke="#F97316" stroke-width="1.5" opacity=".15"/>
            <circle cx="15" cy="15" r="9" stroke="#F97316" stroke-width="1.5" opacity=".38"/>
            <circle cx="15" cy="15" r="4.5" stroke="#F97316" stroke-width="1.5" opacity=".7"/>
            <circle cx="15" cy="15" r="2" fill="#F97316"/>
          </svg>
          <div>
            <div style="font-size:22px;font-weight:700;color:#101828;margin-bottom:6px;">Sonar</div>
            <div class="tagline" data-i18n="hero_tagline">Proteção ativa para chamadas suspeitas</div>
          </div>
          <button class="btn-primary-call" id="btn-start-call" data-i18n="start_call">Iniciar chamada</button>
          <button class="btn-secondary-link" id="btn-show-upload" data-i18n="or_upload">ou analisar gravação</button>
        </div>

        <!-- File upload state -->
        <div id="file-upload-view" class="hide" style="padding:14px;display:flex;flex-direction:column;gap:10px;flex:1;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <button class="btn-secondary-link" id="btn-back-precall" style="font-size:13px;" data-i18n="back">← Voltar</button>
            <span style="font-size:15px;font-weight:600;color:#101828;" data-i18n="analyze_title">Analisar Gravação</span>
          </div>
          <div class="card">
            <div class="upload-panel">
              <label class="file-label" for="file-input">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                <span data-i18n="select_audio">Selecione um arquivo de áudio</span><br>
                <span style="font-size:11px;color:#D0D5DD">WAV · MP3 · WebM · M4A</span>
              </label>
              <input type="file" id="file-input" accept="audio/*" style="display:none">
              <button class="btn-primary" id="upload-btn" onclick="analyzeUpload()" disabled>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                <span data-i18n="analyze_file">Analisar arquivo</span>
              </button>
            </div>
          </div>
          <!-- Alert banner (upload) -->
          <div class="alert-banner" id="alert-banner" style="border-radius:12px;border:1px solid #EAECF0;">
            <div class="a-dot"></div>
            <span class="a-text" id="alert-text" data-i18n="waiting">Aguardando análise</span>
          </div>
          <!-- Status card (upload) -->
          <div class="card hide" id="status-card">
            <div class="status-wrap">
              <div class="status-label" id="status-text" data-i18n="initializing">Inicializando...</div>
              <div class="progress-bar"><span></span></div>
            </div>
          </div>
          <!-- Segment stream (upload) -->
          <div class="seg-stream hide" id="seg-stream"></div>
          <!-- Feedback card (upload) -->
          <div class="card hide" id="feedback-card">
            <div class="sec-label">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#667085" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <span data-i18n="confirmation">Confirmação</span>
            </div>
            <label data-i18n="call_was">A chamada era:</label>
            <select class="fb-select" id="fb-label">
              <option value="scam">Era um golpe</option>
              <option value="legitimate">Era legítima</option>
              <option value="uncertain">Incerto</option>
            </select>
            <label data-i18n="correct_risk">Risco correto:</label>
            <select class="fb-select" id="fb-risk">
              <option value="safe">safe — sem sinais</option>
              <option value="suspicious">suspicious — suspeito</option>
              <option value="danger">danger — golpe provável</option>
            </select>
            <label data-i18n="notes_opt">Notas (opcional):</label>
            <textarea class="fb-area" id="fb-notes" rows="2" placeholder="Ex: banco pedindo código SMS" data-i18n-ph="fb_notes_ph"></textarea>
            <button class="btn-primary" onclick="saveFeedback()" data-i18n="save_fb">Salvar feedback</button>
            <div class="fb-saved hide" id="fb-saved">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#067647" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
              <span data-i18n="fb_saved">Feedback salvo</span>
            </div>
            <button class="btn-secondary hide" id="dataset-btn" onclick="addToDataset()">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
              <span data-i18n="add_dataset">Adicionar ao dataset</span>
            </button>
            <div class="dataset-ok hide" id="dataset-ok" data-i18n="added_dataset">Adicionado ao dataset</div>
          </div>
        </div>

        <!-- Active call state -->
        <div id="active-call" class="hide" style="display:flex;flex-direction:column;flex:1;padding:0;">
          <div id="call-timer-row">
            <div class="call-live-dot"></div>
            <span style="font-size:13px;color:#6B7280;" data-i18n="on_call_label">Em chamada •</span>
            <span id="call-timer">0:00</span>
          </div>
          <div style="padding:0 14px 8px;">
            <canvas id="waveform"></canvas>
          </div>
          <div style="padding:0 14px 8px;">
            <div class="risk-bar safe" id="risk-bar">SEGURO</div>
          </div>
          <div id="live-segments" style="flex:1;overflow-y:auto;padding:0 14px;max-height:280px;scrollbar-width:thin;scrollbar-color:#EAECF0 transparent;"></div>
          <div id="chunk-status" style="display:none;font-size:11px;color:#6B7280;text-align:center;padding:4px 14px;font-style:italic;"></div>
          <div class="call-footer" style="padding:10px 14px;">
            <button class="btn-end-call" id="btn-end-call" onclick="endCall()" data-i18n="end_call">Encerrar chamada</button>
          </div>
        </div>

      </div><!-- end view-call -->

      <!-- ── VIEW: Biblioteca ───────────────────────────────────── -->
      <div id="view-library" class="view-panel hide">
        <div class="view-header" data-i18n="my_calls">Minhas chamadas</div>
        <div class="view-scroll">
          <button class="btn-primary" style="margin-bottom:4px;" onclick="loadLibrary()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/></svg>
            <span data-i18n="load_btn">Carregar</span>
          </button>
          <div id="library-list">
            <p class="lib-empty">Toque em Carregar para ver as chamadas salvas.</p>
          </div>
        </div>
      </div><!-- end view-library -->

      <!-- ── VIEW: Config ───────────────────────────────────────── -->
      <div id="view-config" class="view-panel hide">
        <div class="view-header" data-i18n="settings_title">Configurações</div>
        <div class="view-scroll">
          <div class="card">
            <div class="cfg-body">
              <div class="row2">
                <div>
                  <label data-i18n="lang_label">Idioma</label>
                  <select id="cfg-language">
                    <option value="auto">Auto-detect</option>
                    <option value="pt-BR">Português (BR)</option>
                    <option value="en-US">English (US)</option>
                    <option value="es-419">Español (LA)</option>
                  </select>
                </div>
                <div>
                  <label>Whisper</label>
                  <select id="cfg-whisper">
                    <option>tiny</option><option>small</option><option>medium</option>
                  </select>
                </div>
              </div>
              <div>
                <label data-i18n="model_label">Modelo</label>
                <input type="text" id="cfg-model" value="google/gemma-4-e2b">
              </div>
              <label class="chk">
                <input type="checkbox" id="cfg-fallback">
                <span data-i18n="fallback_lbl">Fallback por sidecar transcript</span>
              </label>
            </div>
          </div>
          <!-- Trusted contacts -->
          <div class="card">
            <div class="sec-label">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#667085" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
              <span data-i18n="emergency_contacts">Contatos de Emergência</span>
            </div>
            <div style="font-size:11px;color:#9CA3AF;margin-bottom:8px;" data-i18n="auto_alerted">Alertados automaticamente ao detectar golpe</div>
            <div id="contact-list"><p style="font-size:12px;color:#D0D5DD">Nenhum contato</p></div>
            <div class="contact-add-form">
              <input class="inp-sm" id="ct-name" placeholder="Nome *" autocomplete="off" data-i18n-ph="ph_name">
              <input class="inp-sm" id="ct-phone" placeholder="Telefone (ex: +55 11 99999-0000)" autocomplete="off" data-i18n-ph="ph_phone">
              <input class="inp-sm" id="ct-email" placeholder="E-mail (para receber alertas)" autocomplete="off" type="email" data-i18n-ph="ph_email">
              <button class="btn-add-contact" onclick="addContact()">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
                <span data-i18n="add_contact">Adicionar contato</span>
              </button>
            </div>
          </div>

          <!-- Email alert config -->
          <div class="card">
            <div class="sec-label">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#667085" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
              <span data-i18n="email_notif">Notificação por E-mail</span>
            </div>
            <div style="font-size:11px;color:#9CA3AF;margin-bottom:10px;" data-i18n="use_gmail">Use Gmail com senha de app ou outro SMTP</div>
            <div class="cfg-body">
              <div class="smtp-row">
                <div>
                  <label data-i18n="smtp_server">Servidor SMTP</label>
                  <input class="inp-sm" id="smtp-host" placeholder="smtp.gmail.com">
                </div>
                <div>
                  <label data-i18n="smtp_port_lbl">Porta</label>
                  <input class="inp-sm" id="smtp-port" placeholder="587">
                </div>
              </div>
              <div>
                <label data-i18n="smtp_from">E-mail remetente</label>
                <input class="inp-sm" id="smtp-user" placeholder="seuemail@gmail.com" type="email">
              </div>
              <div>
                <label data-i18n="smtp_pass_lbl">Senha de app</label>
                <input class="inp-sm" id="smtp-pass" placeholder="••••••••••••" type="password" autocomplete="new-password">
              </div>
              <button class="btn-primary" onclick="saveSmtp()" style="margin-top:2px;" data-i18n="save_config">Salvar configuração</button>
              <div class="smtp-status" id="smtp-status"></div>
            </div>
          </div>

          <!-- History -->
          <div class="card">
            <div class="sec-label">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#667085" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>
              <span data-i18n="feedback_hist">Histórico de Feedback</span>
            </div>
            <div class="hist-list" id="history">
              <p style="font-size:12px;color:#D0D5DD;padding:2px 0">Carregando...</p>
            </div>
          </div>
        </div>
      </div><!-- end view-config -->

    </div><!-- end main-content -->

    <!-- Bottom tab bar -->
    <div id="tab-bar">
      <button class="tab-btn active" id="tbtn-call" onclick="showTab('call')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 12a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.6 1.07h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 8.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
        <span data-i18n="tab_call">Chamada</span>
      </button>
      <button class="tab-btn" id="tbtn-library" onclick="showTab('library')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
        <span data-i18n="tab_library">Biblioteca</span>
      </button>
      <button class="tab-btn" id="tbtn-config" onclick="showTab('config')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/></svg>
        <span data-i18n="tab_config">Config</span>
      </button>
    </div>

    <div class="home-ind"><div class="home-pill"></div></div>
  </div>
</div>

<script>
// ── Signal data ───────────────────────────────────────────────────────────────
const SEV={financial_request:'critical',personal_data_request:'critical',family_emergency_claim:'critical',
  unusual_payment_method:'critical',remote_access_request:'critical',authority_claim:'high',
  isolation_request:'high',secret_keeping_request:'high',urgency_pressure:'medium',emotional_manipulation:'medium'};
const SLBL={financial_request:'Pedido Financeiro',personal_data_request:'Dados Pessoais',
  family_emergency_claim:'Emerg. Familiar',unusual_payment_method:'Pag. Atípico',
  remote_access_request:'Acesso Remoto',authority_claim:'Falsa Autoridade',
  isolation_request:'Isolamento',secret_keeping_request:'Sigilo Forçado',
  urgency_pressure:'Pressão/Urgência',emotional_manipulation:'Manipulação'};
const SLBL_PT = {...SLBL};
const SLBL_EN = {financial_request:'Financial Request',personal_data_request:'Personal Data',
  family_emergency_claim:'Family Emergency',unusual_payment_method:'Unusual Payment',
  remote_access_request:'Remote Access',authority_claim:'False Authority',
  isolation_request:'Isolation Request',secret_keeping_request:'Forced Secrecy',
  urgency_pressure:'Urgency Pressure',emotional_manipulation:'Manipulation'};
const SKW={
  financial_request:['pix','mpix','transferência','transferencia','pagamento','pagar','depósito','deposito','cartão','cartao','crédito','credito','ted','boleto','dinheiro'],
  personal_data_request:['cpf','senha','código','codigo','sms','otp','pin','dados','confirmar','confirme','verificação','verificacao','informações','informacoes'],
  family_emergency_claim:['acidente','hospital','preso','sequestro','emergência','emergencia','filho','filha','neto','neta'],
  unusual_payment_method:['gift card','bitcoin','criptomoeda','crypto'],
  remote_access_request:['anydesk','teamviewer','instale','instalar','baixe','acesso remoto'],
  authority_claim:['banco','polícia','policia','federal','inss','receita','governo','central','bradesco','itaú','itau','santander','caixa','nubank'],
  isolation_request:['não desligue','nao desligue','fique na linha','não conte','nao conte'],
  secret_keeping_request:['segredo','esconda','não diga','nao diga'],
  urgency_pressure:['urgente','agora','imediatamente','rapidamente','minutos'],
  emotional_manipulation:['medo','perigo','grave','sério','serio','preocupada','preocupado']
};

// ── i18n ─────────────────────────────────────────────────────────────────────
let _lang = localStorage.getItem('sonar_lang') || 'pt';
const T = {
  pt: {
    tagline:'Detecção em tempo real', inc_label:'Ligação recebida', inc_sub:'Número externo',
    reject:'Rejeitar', answer:'Atender', hero_tagline:'Proteção ativa para chamadas suspeitas',
    start_call:'Iniciar chamada', or_upload:'ou analisar gravação', back:'← Voltar',
    analyze_title:'Analisar Gravação', select_audio:'Selecione um arquivo de áudio',
    analyze_file:'Analisar arquivo', waiting:'Aguardando análise', initializing:'Inicializando...',
    confirmation:'Confirmação', call_was:'A chamada era:', correct_risk:'Risco correto:',
    notes_opt:'Notas (opcional):', save_fb:'Salvar feedback', fb_saved:'Feedback salvo',
    fb_notes_ph:'Ex: banco pedindo código SMS',
    add_dataset:'Adicionar ao dataset', added_dataset:'Adicionado ao dataset',
    on_call_label:'Em chamada •', end_call:'Encerrar chamada',
    my_calls:'Minhas chamadas', load_btn:'Carregar', settings_title:'Configurações',
    lang_label:'Idioma', model_label:'Modelo', fallback_lbl:'Fallback por sidecar transcript',
    emergency_contacts:'Contatos de Emergência', auto_alerted:'Alertados automaticamente ao detectar golpe',
    ph_name:'Nome *', ph_phone:'Telefone (ex: +55 11 99999-0000)', ph_email:'E-mail (para receber alertas)',
    add_contact:'Adicionar contato', email_notif:'Notificação por E-mail',
    use_gmail:'Use Gmail com senha de app ou outro SMTP',
    smtp_server:'Servidor SMTP', smtp_port_lbl:'Porta', smtp_from:'E-mail remetente',
    smtp_pass_lbl:'Senha de app', save_config:'Salvar configuração',
    feedback_hist:'Histórico de Feedback', tab_call:'Chamada', tab_library:'Biblioteca', tab_config:'Config',
    fb_scam:'Era um golpe', fb_legit:'Era legítima', fb_uncertain:'Incerto',
    safe_risk:'SEGURO', suspicious_risk:'SUSPEITO', danger_risk:'⚠ PERIGO',
    lib_empty:'Toque em Carregar para ver as chamadas salvas.', lib_none:'Nenhuma chamada salva ainda.',
    no_contacts:'Nenhum contato', feedback_none:'Nenhum feedback salvo.',
    analyzing:'Analisando...', silence_detected:'Silêncio detectado...', conn_error:'Erro de conexão com servidor',
    sending_audio:'Enviando áudio para análise...', sending_audio2:'Enviando áudio...',
    analysis_done:'Análise concluída', starting_stage:'Iniciando...', preflight_stage:'Verificando sistema...',
    stt_stage:'Transcrevendo áudio...', classifying_stage:'Classificando segmentos...',
    analysis_error:'Erro na análise', smtp_saved:'Salvo.', smtp_email_required:'E-mail obrigatório.', smtp_save_error:'Erro ao salvar.', smtp_pass_set:'Senha configurada.',
    unknown_caller:'Número desconhecido', seg_label:'Seg.',
    save_evidence_confirm_pre:'Esta chamada foi classificada como ', save_evidence_confirm_post:'.\n\nDeseja salvar como evidência para denúncia e melhorar a detecção?',
    risk_danger_pt:'PERIGO', risk_suspicious_pt:'SUSPEITO',
    report_not_found:'Relatório não encontrado.',
    mic_unavailable:'Microfone indisponível: ', session_start_error:'Erro ao iniciar sessão: ', answer_error:'Erro ao atender: ',
    export_btn:'Exportar', start_analysis_error:'Falha ao iniciar análise',
  },
  en: {
    tagline:'Real-time detection', inc_label:'Incoming call', inc_sub:'External number',
    reject:'Decline', answer:'Answer', hero_tagline:'Active protection for suspicious calls',
    start_call:'Start call', or_upload:'or analyze recording', back:'← Back',
    analyze_title:'Analyze Recording', select_audio:'Select an audio file',
    analyze_file:'Analyze file', waiting:'Waiting for analysis', initializing:'Initializing...',
    confirmation:'Confirmation', call_was:'The call was:', correct_risk:'Correct risk:',
    notes_opt:'Notes (optional):', save_fb:'Save feedback', fb_saved:'Feedback saved',
    fb_notes_ph:'E.g. bank asking for SMS code',
    add_dataset:'Add to dataset', added_dataset:'Added to dataset',
    on_call_label:'On call •', end_call:'End call',
    my_calls:'My calls', load_btn:'Load', settings_title:'Settings',
    lang_label:'Language', model_label:'Model', fallback_lbl:'Sidecar transcript fallback',
    emergency_contacts:'Emergency Contacts', auto_alerted:'Automatically alerted when a scam is detected',
    ph_name:'Name *', ph_phone:'Phone (e.g. +1 555 000-0000)', ph_email:'Email (to receive alerts)',
    add_contact:'Add contact', email_notif:'Email Notification',
    use_gmail:'Use Gmail with app password or other SMTP',
    smtp_server:'SMTP Server', smtp_port_lbl:'Port', smtp_from:'Sender email',
    smtp_pass_lbl:'App password', save_config:'Save settings',
    feedback_hist:'Feedback History', tab_call:'Call', tab_library:'Library', tab_config:'Settings',
    fb_scam:'It was a scam', fb_legit:'It was legitimate', fb_uncertain:'Uncertain',
    safe_risk:'SAFE', suspicious_risk:'SUSPICIOUS', danger_risk:'⚠ DANGER',
    lib_empty:'Tap Load to see saved calls.', lib_none:'No saved calls yet.',
    no_contacts:'No contacts', feedback_none:'No feedback saved.',
    analyzing:'Analyzing...', silence_detected:'Silence detected...', conn_error:'Server connection error',
    sending_audio:'Sending audio for analysis...', sending_audio2:'Sending audio...',
    analysis_done:'Analysis complete', starting_stage:'Starting...', preflight_stage:'Checking system...',
    stt_stage:'Transcribing audio...', classifying_stage:'Classifying segments...',
    analysis_error:'Analysis error', smtp_saved:'Saved.', smtp_email_required:'Email required.', smtp_save_error:'Save error.', smtp_pass_set:'Password configured.',
    unknown_caller:'Unknown number', seg_label:'Seg.',
    save_evidence_confirm_pre:'This call was classified as ', save_evidence_confirm_post:'.\n\nDo you want to save it as evidence and improve detection?',
    risk_danger_pt:'DANGER', risk_suspicious_pt:'SUSPICIOUS',
    report_not_found:'Report not found.',
    mic_unavailable:'Microphone unavailable: ', session_start_error:'Error starting session: ', answer_error:'Error answering: ',
    export_btn:'Export', start_analysis_error:'Failed to start analysis',
  }
};
function t(key) { return (T[_lang] || T.pt)[key] || key; }
function applyLang(lang) {
  _lang = lang;
  localStorage.setItem('sonar_lang', lang);
  document.getElementById('lang-pt').classList.toggle('active', lang === 'pt');
  document.getElementById('lang-en').classList.toggle('active', lang === 'en');
  document.querySelectorAll('[data-i18n]').forEach(el => { const v = t(el.dataset.i18n); if (v) el.textContent = v; });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => { const v = t(el.dataset.i18nPh); if (v) el.placeholder = v; });
  const fbl = document.getElementById('fb-label');
  if (fbl) { fbl.options[0].text = t('fb_scam'); fbl.options[1].text = t('fb_legit'); fbl.options[2].text = t('fb_uncertain'); }
  Object.assign(SLBL, lang === 'en' ? SLBL_EN : SLBL_PT);
}

// ── Tab navigation ────────────────────────────────────────────────────────────
let _currentTab = 'call';
function getCurrentTab() { return _currentTab; }
function showTab(tab) {
  _currentTab = tab;
  document.getElementById('view-call').classList.toggle('hide', tab !== 'call');
  document.getElementById('view-library').classList.toggle('hide', tab !== 'library');
  document.getElementById('view-config').classList.toggle('hide', tab !== 'config');
  document.getElementById('tbtn-call').classList.toggle('active', tab === 'call');
  document.getElementById('tbtn-library').classList.toggle('active', tab === 'library');
  document.getElementById('tbtn-config').classList.toggle('active', tab === 'config');
  if (tab === 'library') loadLibrary();
  if (tab === 'config') { loadHistory(); loadContacts(); loadSmtpConfig(); }
}

// ── Pre-call / file toggle ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-show-upload').addEventListener('click', () => {
    document.getElementById('pre-call').classList.add('hide');
    document.getElementById('file-upload-view').classList.remove('hide');
    document.getElementById('file-upload-view').style.display = 'flex';
  });
  document.getElementById('btn-back-precall').addEventListener('click', () => {
    document.getElementById('file-upload-view').classList.add('hide');
    document.getElementById('pre-call').classList.remove('hide');
  });
  const fi = document.getElementById('file-input');
  if (fi) fi.addEventListener('change', () => { document.getElementById('upload-btn').disabled = !fi.files.length; });
  document.getElementById('btn-start-call').addEventListener('click', startCall);
  loadHistory();
  startRingPolling();
  applyLang(_lang);
});

// ── Live call state ───────────────────────────────────────────────────────────
let sessionId = null, callTimerInterval = null, callSeconds = 0;
let mediaRecorder = null, audioCtx = null, analyserNode = null, animFrame = null;
let chunkProcessing = false, pendingChunks = [];
let callActive = false, activeStream = null;
let seenSegIds = new Set(), liveCallPoll = null;

// ── Incoming call polling ─────────────────────────────────────────────────────
let _ringPollTimer = null;
let _ringCtx = null, _ringBeepTimer = null;

function startRingPolling() {
  if (_ringPollTimer) return;
  _ringPollTimer = setInterval(_checkIncoming, 1500);
  _checkIncoming();
}
function stopRingPolling() {
  if (_ringPollTimer) { clearInterval(_ringPollTimer); _ringPollTimer = null; }
}
async function _checkIncoming() {
  if (callActive) return;
  try {
    const d = await (await fetch('/api/incoming-call')).json();
    if (d.ringing) _showIncoming(d.caller_id);
    else _hideIncoming();
  } catch(e) {}
}
function _showIncoming(callerId) {
  document.getElementById('inc-caller-id').textContent = callerId || t('unknown_caller');
  const ov = document.getElementById('incoming-overlay');
  ov.classList.remove('hide');
  _startRingtone();
}
function _hideIncoming() {
  document.getElementById('incoming-overlay').classList.add('hide');
  _stopRingtone();
}
function _startRingtone() {
  if (_ringCtx) return;
  try {
    _ringCtx = new (window.AudioContext || window.webkitAudioContext)();
    function _beep() {
      const o = _ringCtx.createOscillator(), g = _ringCtx.createGain();
      o.connect(g); g.connect(_ringCtx.destination);
      o.frequency.value = 480;
      g.gain.setValueAtTime(0.25, _ringCtx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, _ringCtx.currentTime + 0.35);
      o.start(_ringCtx.currentTime); o.stop(_ringCtx.currentTime + 0.35);
    }
    _beep(); _ringBeepTimer = setInterval(_beep, 1800);
  } catch(e) {}
}
function _stopRingtone() {
  if (_ringBeepTimer) { clearInterval(_ringBeepTimer); _ringBeepTimer = null; }
  if (_ringCtx) { try { _ringCtx.close(); } catch(e) {} _ringCtx = null; }
}

async function rejectCall() {
  _stopRingtone();
  _hideIncoming();
  try { await fetch('/api/call/reject', {method:'POST'}); } catch(e) {}
}

async function answerCall() {
  stopRingPolling();
  _stopRingtone();
  _hideIncoming();
  let data;
  try {
    const res = await fetch('/api/call/answer', {method:'POST'});
    data = await res.json();
  } catch(err) { alert(t('answer_error') + err); startRingPolling(); return; }
  if (!data.ok) { alert(t('answer_error') + data.error); startRingPolling(); return; }
  await _beginActiveCall(data.session_id);
}

async function startCall() {
  const cfg = getConfig();
  let data;
  try {
    const res = await fetch('/api/call/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({language: cfg.language, model: cfg.model, whisper_model: cfg.whisperModel})
    });
    data = await res.json();
  } catch(err) { alert(t('session_start_error') + err); return; }
  if (!data.ok) { alert(t('session_start_error') + data.error); return; }
  await _beginActiveCall(data.session_id);
}

function startLiveResultPolling(sid) {
  seenSegIds.clear();
  liveCallPoll = setInterval(async () => {
    if (!callActive) return;
    try {
      const r = await fetch('/api/status/' + sid);
      const d = await r.json();
      if (d.ok && Array.isArray(d.segments)) {
        d.segments.forEach(seg => {
          const id = seg.classification && seg.classification.segment_id;
          if (id && !seenSegIds.has(id)) {
            seenSegIds.add(id);
            renderSegment(seg);
          }
        });
      }
    } catch(e) {}
  }, 3000);
}

function stopLiveResultPolling() {
  if (liveCallPoll) { clearInterval(liveCallPoll); liveCallPoll = null; }
  seenSegIds.clear();
}

async function _beginActiveCall(session_id) {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
  } catch(err) { alert(t('mic_unavailable') + err.message); return; }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyserNode);
  drawWaveform();
  sessionId = session_id;
  chunkProcessing = false;
  pendingChunks = [];
  callActive = true;
  activeStream = stream;
  startLiveResultPolling(session_id);
  startChunkCycle();
  document.getElementById('pre-call').classList.add('hide');
  document.getElementById('active-call').classList.remove('hide');
  document.getElementById('active-call').style.display = 'flex';
  const canvas = document.getElementById('waveform');
  canvas.width = canvas.offsetWidth || 340;
  callSeconds = 0;
  callTimerInterval = setInterval(() => {
    callSeconds++;
    const m = Math.floor(callSeconds / 60), s = callSeconds % 60;
    document.getElementById('call-timer').textContent = m + ':' + (s < 10 ? '0' : '') + s;
  }, 1000);
}

function startChunkCycle() {
  if (!callActive || !activeStream) return;
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
  mediaRecorder = new MediaRecorder(activeStream, {mimeType});
  mediaRecorder.ondataavailable = e => { if (callActive && e.data && e.data.size > 100) enqueueChunk(e.data); };
  mediaRecorder.onstop = () => { if (callActive) startChunkCycle(); };
  mediaRecorder.start();
  setTimeout(() => { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }, 6000);
}

async function enqueueChunk(blob) {
  pendingChunks.push(blob);
  if (!chunkProcessing) processNextChunk();
}

async function processNextChunk() {
  if (!pendingChunks.length || !sessionId) { chunkProcessing = false; return; }
  chunkProcessing = true;
  const blob = pendingChunks.shift();
  const fd = new FormData();
  fd.append('audio', blob, 'chunk.webm');
  showChunkStatus(t('analyzing'));
  try {
    const res = await fetch('/api/call/' + sessionId + '/chunk', {method: 'POST', body: fd});
    const data = await res.json();
    if (!data.ok) {
      showChunkStatus('Erro: ' + (data.error || '').slice(0, 80));
      console.warn('[sonar] chunk error:', data.error);
    } else if (data.silence) {
      showChunkStatus(t('silence_detected'));
    } else {
      showChunkStatus('');
    }
  } catch(e) {
    showChunkStatus(t('conn_error'));
    console.warn('chunk fetch error', e);
  }
  processNextChunk();
}

function showChunkStatus(msg) {
  const el = document.getElementById('chunk-status');
  if (el) { el.textContent = msg; el.style.display = msg ? 'block' : 'none'; }
}

async function endCall() {
  callActive = false;
  stopLiveResultPolling();
  pendingChunks = [];
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.onstop = null;
    mediaRecorder.stop();
  }
  if (activeStream) { activeStream.getTracks().forEach(t => t.stop()); activeStream = null; }
  clearInterval(callTimerInterval);
  cancelAnimationFrame(animFrame);
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  let data = {};
  try {
    const res = await fetch('/api/call/' + sessionId + '/end', {method: 'POST'});
    data = await res.json();
  } catch(e) { console.warn('end call error', e); }
  document.getElementById('active-call').classList.add('hide');
  document.getElementById('pre-call').classList.remove('hide');
  document.getElementById('live-segments').innerHTML = '';
  const bar = document.getElementById('risk-bar');
  bar.className = 'risk-bar safe';
  bar.textContent = t('safe_risk');
  if (data.ok && data.final && data.final.overall_risk !== 'safe') {
    showSaveDialog(data);
  }
  sessionId = null;
  startRingPolling();
}

function drawWaveform() {
  const canvas = document.getElementById('waveform');
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(analyserNode.frequencyBinCount);
  function frame() {
    animFrame = requestAnimationFrame(frame);
    analyserNode.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#F97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sliceW = canvas.width / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const y = (buf[i] / 128) * canvas.height / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }
  frame();
}

function renderSegment(seg) {
  const cl = seg.classification, cs = seg.call_state;
  const risk = cl.risk_level;
  const signals = cl.signals_detected.join(', ') || '—';
  const card = document.createElement('div');
  card.className = 'seg-card seg-' + risk;
  card.innerHTML =
    '<div class="seg-header">' +
      '<span class="seg-risk-badge risk-' + esc(risk) + '">' + esc(risk.toUpperCase()) + '</span>' +
      '<span class="seg-excerpt">' + esc(cl.transcript_excerpt) + '</span>' +
    '</div>' +
    (signals !== '—' ? '<div class="seg-signals">' + esc(signals) + '</div>' : '') +
    '<div class="seg-action">' + esc(cl.suggested_action_for_user) + '</div>';
  const list = document.getElementById('live-segments');
  list.appendChild(card);
  list.scrollTop = list.scrollHeight;
  const bar = document.getElementById('risk-bar');
  const alertLevel = cs.alert_level;
  if (alertLevel === 'red') {
    bar.className = 'risk-bar danger'; bar.textContent = t('danger_risk');
    if (cs.should_play_audio_alert) playBeep();
  } else if (alertLevel === 'yellow' || alertLevel === 'orange') {
    bar.className = 'risk-bar suspicious'; bar.textContent = t('suspicious_risk');
  }
}

function showSaveDialog(data) {
  const risk = data.final.overall_risk;
  const riskLabel = risk === 'danger' ? t('risk_danger_pt') : t('risk_suspicious_pt');
  if (!confirm(t('save_evidence_confirm_pre') + riskLabel + t('save_evidence_confirm_post'))) return;
  const label = risk === 'danger' ? 'scam' : 'uncertain';
  fetch('/api/library/' + data.recording_id + '/label', {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label})
  }).then(() => { if (getCurrentTab() === 'library') loadLibrary(); });
}

// ── Library ───────────────────────────────────────────────────────────────────
async function loadLibrary() {
  const res = await fetch('/api/library');
  const data = await res.json();
  const list = document.getElementById('library-list');
  if (!data.entries || !data.entries.length) {
    list.innerHTML = '<p class="lib-empty">' + t('lib_none') + '</p>';
    return;
  }
  list.innerHTML = data.entries.map(e => {
    const d = new Date(e.timestamp * 1000);
    const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});
    const dur = e.duration_seconds ? Math.round(e.duration_seconds) + 's' : '—';
    const riskClass = e.overall_risk;
    const signals = e.top_signals.slice(0, 3).join(', ') || '—';
    const labelBtns = ['scam', 'legitimate', 'uncertain'].map(l =>
      '<button class="label-btn ' + (e.label === l ? 'active-label' : '') + '" onclick="setLabel(\'' + esc(e.recording_id) + '\',\'' + l + '\',this)">' +
        (l === 'scam' ? t('fb_scam') : l === 'legitimate' ? t('fb_legit') : t('fb_uncertain')) +
      '</button>'
    ).join('');
    return '<div class="lib-entry">' +
      '<div class="lib-meta"><span class="lib-date">' + esc(dateStr) + '</span><span class="lib-dur">' + esc(dur) + '</span><span class="risk-badge-sm risk-' + esc(riskClass) + '">' + esc(e.overall_risk.toUpperCase()) + '</span></div>' +
      '<div class="lib-signals">' + esc(signals) + '</div>' +
      '<div class="lib-actions">' + labelBtns + '<button class="label-btn export-btn" onclick="exportEntry(\'' + esc(e.recording_id) + '\')">' + t('export_btn') + '</button></div>' +
    '</div>';
  }).join('');
}

async function setLabel(recordingId, label, btn) {
  await fetch('/api/library/' + recordingId + '/label', {
    method: 'PATCH', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label})
  });
  btn.closest('.lib-actions').querySelectorAll('.label-btn:not(.export-btn)').forEach(b => b.classList.remove('active-label'));
  btn.classList.add('active-label');
}

function exportEntry(recordingId) {
  fetch('/api/export/' + recordingId).then(r => r.json()).then(data => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = recordingId + '_relatorio.json';
    a.click();
  }).catch(() => alert(t('report_not_found')));
}

// ── Config ────────────────────────────────────────────────────────────────────
function getConfig() {
  return {
    model: document.getElementById('cfg-model').value || 'google/gemma-4-e2b',
    whisperModel: document.getElementById('cfg-whisper').value || 'tiny',
    language: document.getElementById('cfg-language').value || 'pt-BR',
  };
}

// ── Upload analysis (file upload flow) ───────────────────────────────────────
let currentJob = null, pollInterval = null, seenSegments = 0, dangerAlerted = false, feedbackSaved = false;

function analyzeUpload() {
  const fi = document.getElementById('file-input');
  if (!fi.files.length) return;
  submitAnalysis(fi.files[0]);
}

async function submitAnalysis(file) {
  resetUploadUI();
  const fd = new FormData();
  fd.append('audio', file);
  const cfg = getConfig();
  fd.append('language', cfg.language);
  fd.append('model', cfg.model);
  fd.append('whisper_model', cfg.whisperModel);
  fd.append('allow_transcript_fallback', document.getElementById('cfg-fallback').checked ? 'true' : 'false');
  setAlertBanner('running', t('sending_audio'));
  document.getElementById('status-card').classList.remove('hide');
  document.getElementById('status-text').textContent = t('sending_audio2');
  try {
    const r = await fetch('/api/analyze', {method: 'POST', body: fd});
    const d = await r.json();
    if (!d.ok) { showError(d.error || t('start_analysis_error')); return; }
    currentJob = {job_id: d.job_id, recording_id: d.recording_id, audio_suffix: d.audio_suffix, final_risk: null};
    startPolling(d.job_id);
  } catch(err) { showError(String(err)); }
}

function startPolling(jid) { seenSegments = 0; pollInterval = setInterval(() => pollStatus(jid), 2000); pollStatus(jid); }
async function pollStatus(jid) {
  try {
    const d = await (await fetch('/api/status/' + jid)).json();
    if (!d.ok) { stopPolling(); showError(d.error || 'Erro no job'); return; }
    const stg = {starting:t('starting_stage'),preflight:t('preflight_stage'),stt:t('stt_stage'),classifying:t('classifying_stage'),done:t('analysis_done')};
    const lbl = stg[d.stage] || d.stage;
    const pct = d.total > 0 && d.segments.length > 0 ? ' (' + d.segments.length + '/' + d.total + ')' : '';
    document.getElementById('status-text').textContent = lbl + pct;
    const ns = d.segments.slice(seenSegments);
    for (const seg of ns) {
      seenSegments++;
      addSegmentCard(seg, seenSegments);
      const cs = seg.call_state;
      setAlertBanner(cs.overall_risk, cs.rationale_for_user);
      if (cs.should_play_audio_alert && !dangerAlerted) { dangerAlerted = true; playBeep(); }
    }
    if (d.status === 'done') {
      stopPolling();
      document.getElementById('status-text').textContent = t('analysis_done');
      if (d.final) {
        setAlertBanner(d.final.overall_risk, d.final.rationale_for_user);
        if (currentJob) currentJob.final_risk = d.final.overall_risk;
        const el = document.getElementById('fb-risk');
        if (el && d.final.overall_risk !== 'none') el.value = d.final.overall_risk;
      }
      document.getElementById('feedback-card').classList.remove('hide');
    } else if (d.status === 'error') { stopPolling(); showError(d.error || t('analysis_error')); }
  } catch(_) {}
}
function stopPolling() { if (pollInterval) { clearInterval(pollInterval); pollInterval = null; } }

function resetUploadUI() {
  stopPolling(); currentJob = null; seenSegments = 0; dangerAlerted = false; feedbackSaved = false;
  document.getElementById('seg-stream').innerHTML = '';
  document.getElementById('seg-stream').classList.add('hide');
  document.getElementById('feedback-card').classList.add('hide');
  document.getElementById('fb-saved').classList.add('hide');
  document.getElementById('dataset-btn').classList.add('hide');
  document.getElementById('dataset-ok').classList.add('hide');
  document.getElementById('status-card').classList.add('hide');
  setAlertBanner('', t('waiting'));
}
function setAlertBanner(risk, text) {
  const b = document.getElementById('alert-banner');
  b.className = 'alert-banner';
  if (['safe','suspicious','danger','running'].includes(risk)) b.classList.add(risk);
  document.getElementById('alert-text').textContent = text || '';
}
function showError(msg) {
  document.getElementById('status-card').classList.remove('hide');
  document.getElementById('status-text').textContent = msg;
  setAlertBanner('danger', t('analysis_error'));
}

// ── Segment cards (upload analysis) ──────────────────────────────────────────
function addSegmentCard(seg, num) {
  const c = seg.classification, risk = c.risk_level || 'safe', sigs = c.signals_detected || [];
  const conf = c.confidence !== undefined ? Math.round(c.confidence * 100) + '%' : '—';
  const chips = sigs.map(s => {
    const sv = SEV[s] || 'medium', lb = SLBL[s] || s.replace(/_/g, ' ');
    return '<span class="sig-chip chip-' + sv + '">' + esc(lb) + '</span>';
  }).join('');
  const card = document.createElement('div');
  card.className = 'seg-card ' + esc(risk);
  card.innerHTML =
    '<div class="seg-hdr">' +
      '<span class="seg-num">' + t('seg_label') + ' ' + num + '</span>' +
      '<span class="risk-tag ' + esc(risk) + '">' + esc(risk) + '</span>' +
      '<span class="conf">' + conf + '</span>' +
    '</div>' +
    (chips ? '<div class="sig-chips">' + chips + '</div>' : '') +
    '<div class="t-text">' + hlTranscript(c.transcript_excerpt || '', sigs) + '</div>' +
    (c.suggested_action_for_user ? '<div class="a-tip">' + esc(c.suggested_action_for_user) + '</div>' : '');
  const stream = document.getElementById('seg-stream');
  stream.classList.remove('hide');
  stream.appendChild(card);
}

// ── Transcript highlighting ───────────────────────────────────────────────────
function hlTranscript(text, signals) {
  if (!signals || !signals.length) return esc(text);
  const lt = text.toLowerCase(), ivs = [];
  for (const sig of signals) {
    const sv = SEV[sig] || 'medium';
    for (const kw of (SKW[sig] || [])) {
      let p = 0;
      while ((p = lt.indexOf(kw, p)) !== -1) { ivs.push({start: p, end: p + kw.length, cls: 'hl-' + sv}); p += kw.length; }
    }
  }
  if (!ivs.length) return esc(text);
  ivs.sort((a, b) => a.start - b.start);
  let out = '', cur = 0;
  for (const iv of ivs) {
    if (iv.start < cur) continue;
    out += esc(text.slice(cur, iv.start));
    out += '<mark class="' + iv.cls + '">' + esc(text.slice(iv.start, iv.end)) + '</mark>';
    cur = iv.end;
  }
  return out + esc(text.slice(cur));
}

// ── Feedback (upload analysis) ────────────────────────────────────────────────
async function saveFeedback() {
  if (!currentJob) return;
  const p = {recording_id: currentJob.recording_id, audio_suffix: currentJob.audio_suffix,
    user_label: document.getElementById('fb-label').value,
    user_corrected_risk: document.getElementById('fb-risk').value,
    user_notes: document.getElementById('fb-notes').value,
    predicted_risk: currentJob.final_risk || ''};
  try {
    const r = await fetch('/api/feedback', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)});
    if (r.ok) { feedbackSaved = true; document.getElementById('fb-saved').classList.remove('hide'); document.getElementById('dataset-btn').classList.remove('hide'); loadHistory(); }
  } catch(err) { alert('Erro: ' + err); }
}
async function addToDataset() {
  if (!currentJob || !feedbackSaved) return;
  const p = {recording_id: currentJob.recording_id, user_label: document.getElementById('fb-label').value,
    user_corrected_risk: document.getElementById('fb-risk').value,
    language: getConfig().language, notes: document.getElementById('fb-notes').value};
  try {
    const r = await fetch('/api/add_to_dataset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)});
    const d = await r.json();
    if (d.ok) { document.getElementById('dataset-btn').classList.add('hide'); document.getElementById('dataset-ok').classList.remove('hide'); }
    else alert('Erro: ' + (d.error || 'desconhecido'));
  } catch(err) { alert('Erro: ' + err); }
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const d = await (await fetch('/api/history')).json();
    const items = d.items || [];
    const el = document.getElementById('history');
    if (!items.length) { el.innerHTML = '<p style="font-size:12px;color:#D0D5DD;padding:2px 0">' + t('feedback_none') + '</p>'; return; }
    el.innerHTML = items.map(x =>
      '<div class="hist-item ' + esc(x.user_label) + '">' +
        '<strong>' + esc(x.user_label) + '</strong> → <strong>' + esc(x.user_corrected_risk) + '</strong>' +
        '<span style="color:#98A2B3"> (modelo: ' + esc(x.predicted_risk || '?') + ')</span>' +
        '<div style="font-size:10px;color:#D0D5DD;margin-top:1px;font-family:monospace">' + esc(x.recording_id) + '</div>' +
        (x.user_notes ? '<div style="margin-top:2px;color:#98A2B3">' + esc(x.user_notes) + '</div>' : '') +
      '</div>'
    ).join('');
  } catch(_) {}
}

// ── Contacts ──────────────────────────────────────────────────────────────────
async function loadContacts() {
  try {
    const d = await (await fetch('/api/contacts')).json();
    renderContacts(d.contacts || []);
  } catch(_) {}
}
function renderContacts(contacts) {
  const el = document.getElementById('contact-list');
  if (!contacts.length) { el.innerHTML = '<p style="font-size:12px;color:#D0D5DD">' + t('no_contacts') + '</p>'; return; }
  el.innerHTML = contacts.map(c => `
    <div class="contact-row">
      <div class="contact-info">
        <div class="contact-name">${esc(c.name)}</div>
        <div class="contact-detail">${[c.phone, c.email].filter(Boolean).map(esc).join(' · ')}</div>
      </div>
      <button class="btn-del-contact" onclick="deleteContact('${esc(c.id)}')" title="Remover">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
      </button>
    </div>`).join('');
}
async function addContact() {
  const name = document.getElementById('ct-name').value.trim();
  const phone = document.getElementById('ct-phone').value.trim();
  const email = document.getElementById('ct-email').value.trim();
  if (!name) { document.getElementById('ct-name').focus(); return; }
  try {
    const d = await (await fetch('/api/contacts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, phone, email})
    })).json();
    if (d.ok) { renderContacts(d.contacts); document.getElementById('ct-name').value = ''; document.getElementById('ct-phone').value = ''; document.getElementById('ct-email').value = ''; }
  } catch(_) {}
}
async function deleteContact(id) {
  try {
    const d = await (await fetch('/api/contacts/' + id, {method:'DELETE'})).json();
    if (d.ok) renderContacts(d.contacts);
  } catch(_) {}
}

// ── SMTP config ───────────────────────────────────────────────────────────────
async function saveSmtp() {
  const host = document.getElementById('smtp-host').value.trim() || 'smtp.gmail.com';
  const port = parseInt(document.getElementById('smtp-port').value) || 587;
  const username = document.getElementById('smtp-user').value.trim();
  const password = document.getElementById('smtp-pass').value;
  const st = document.getElementById('smtp-status');
  if (!username) { st.textContent = t('smtp_email_required'); return; }
  try {
    const d = await (await fetch('/api/smtp-config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({host, port, username, password: password || undefined})
    })).json();
    if (d.ok) { st.textContent = t('smtp_saved'); st.className = 'smtp-status ok'; document.getElementById('smtp-pass').value = ''; }
  } catch(_) { st.textContent = t('smtp_save_error'); }
}
async function loadSmtpConfig() {
  try {
    const d = await (await fetch('/api/smtp-config')).json();
    const s = d.smtp || {};
    if (s.host) document.getElementById('smtp-host').value = s.host;
    if (s.port) document.getElementById('smtp-port').value = s.port;
    if (s.username) document.getElementById('smtp-user').value = s.username;
    const st = document.getElementById('smtp-status');
    if (s.password_set) { st.textContent = t('smtp_pass_set'); st.className = 'smtp-status ok'; }
  } catch(_) {}
}

// ── Audio alert ───────────────────────────────────────────────────────────────
function playBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [0, .36, .72].forEach(delay => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.connect(g); g.connect(ctx.destination);
      o.frequency.setValueAtTime(880, ctx.currentTime + delay);
      g.gain.setValueAtTime(.25, ctx.currentTime + delay);
      g.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + delay + .28);
      o.start(ctx.currentTime + delay); o.stop(ctx.currentTime + delay + .28);
    });
  } catch(_) {}
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script>
</body></html>'''


def _monitor_html() -> str:
    return r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Sonar — Pipeline Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#060a10;color:#c9d1d9;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;
     min-height:100vh;display:flex;flex-direction:column;font-size:13px}
.hd{background:#0d1117;border-bottom:1px solid #1a2744;height:50px;padding:0 20px;
    display:flex;align-items:center;gap:18px;flex-shrink:0}
.hd-logo{display:flex;align-items:center;gap:10px}
.hd-name{font-size:13px;font-weight:800;color:#e2e8f0;letter-spacing:.06em}
.hd-div{width:1px;height:22px;background:#1a2744}
.hd-live{display:flex;align-items:center;gap:5px;font-size:10px;font-weight:800;color:#4ade80;letter-spacing:.1em}
.live-dot{width:7px;height:7px;border-radius:50%;background:#4ade80;animation:lp 1.5s infinite}
@keyframes lp{0%,100%{opacity:1}50%{opacity:.25}}
.hd-stat{font-size:11px;color:#374151}
.hd-stat span{color:#c9d1d9;font-weight:700}
.hd-link{font-size:11px;color:#38bdf8;text-decoration:none;
         border:1px solid #1a3a5f;border-radius:6px;padding:3px 10px;letter-spacing:.05em}
.hd-link:hover{background:rgba(56,189,248,.08)}
.hd-ring-wrap{margin-left:auto;display:flex;align-items:center;gap:8px}
.btn-ring{font-size:11px;font-weight:700;letter-spacing:.04em;cursor:pointer;border:1px solid #ea580c;border-radius:6px;padding:4px 12px;background:rgba(234,88,12,.12);color:#fb923c;font-family:inherit;transition:.2s}
.btn-ring:hover:not(:disabled){background:rgba(234,88,12,.22)}
.btn-ring:disabled{opacity:.45;cursor:not-allowed}
.btn-ring.ringing{border-color:#fbbf24;color:#fbbf24;background:rgba(251,191,36,.1);animation:ringblink 1.2s infinite}
@keyframes ringblink{0%,100%{opacity:1}50%{opacity:.55}}
.ring-st{font-size:10px;color:#374151}
.pipe-row{background:#0a0e18;border-bottom:1px solid #1a2744;padding:14px 20px;
          display:flex;align-items:center;flex-shrink:0}
.pn{background:#0d1117;border:1px solid #1a2744;border-radius:10px;
    padding:9px 14px;min-width:115px;text-align:center;transition:.3s}
.pn-title{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#374151}
.pn-sub{font-size:10px;color:#1f2937;margin-top:2px}
.pn-val{font-size:12px;font-weight:700;color:#1f2937;margin-top:3px}
.pn.active{background:rgba(56,189,248,.07);border-color:#38bdf8;box-shadow:0 0 18px rgba(56,189,248,.2)}
.pn.active .pn-title{color:#38bdf8}.pn.active .pn-val{color:#e2e8f0}.pn.active .pn-sub{color:#7dd3fc}
.pn.done{background:rgba(74,222,128,.06);border-color:#4ade80;box-shadow:0 0 10px rgba(74,222,128,.12)}
.pn.done .pn-title{color:#4ade80}.pn.done .pn-val{color:#e2e8f0}
.pn.warn{background:rgba(251,191,36,.06);border-color:#fbbf24;box-shadow:0 0 14px rgba(251,191,36,.15)}
.pn.warn .pn-title{color:#fbbf24}.pn.warn .pn-val{color:#fde68a}
.pn.dng{background:rgba(248,113,113,.08);border-color:#f87171;box-shadow:0 0 18px rgba(248,113,113,.25);animation:dnp 2s infinite}
.pn.dng .pn-title{color:#f87171}.pn.dng .pn-val{color:#f87171}
@keyframes dnp{0%,100%{box-shadow:0 0 18px rgba(248,113,113,.25)}50%{box-shadow:0 0 30px rgba(248,113,113,.55)}}
.pc{flex:1;height:2px;background:repeating-linear-gradient(90deg,#1a2744 0,#1a2744 6px,transparent 6px,transparent 12px)}
.pc.flow{background:repeating-linear-gradient(90deg,#38bdf8 0,#38bdf8 6px,transparent 6px,transparent 12px);animation:pf .4s linear infinite}
.pc.done-f{background:repeating-linear-gradient(90deg,#4ade80 0,#4ade80 6px,transparent 6px,transparent 12px);animation:pf .9s linear infinite}
@keyframes pf{from{background-position:0}to{background-position:12px}}
.mgrid{flex:1;display:grid;grid-template-columns:35% 27% 38%;min-height:0}
.mpanel{border-right:1px solid #1a2744;border-bottom:1px solid #1a2744;display:flex;flex-direction:column;overflow:hidden}
.mpanel:last-child{border-right:0}
.mp-title{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:#374151;
          padding:9px 14px 7px;border-bottom:1px solid #1a2744;flex-shrink:0;display:flex;align-items:center;gap:8px}
.badge{font-size:9px;padding:1px 7px;border-radius:999px;margin-left:auto}
.bs{background:rgba(74,222,128,.15);color:#4ade80}
.bw{background:rgba(251,191,36,.15);color:#fbbf24}
.bd{background:rgba(248,113,113,.15);color:#f87171}
.br{background:rgba(56,189,248,.15);color:#38bdf8}
.mp-body{flex:1;overflow:auto;padding:10px 14px;scrollbar-width:thin;scrollbar-color:#1a2744 transparent}
#risk-chart{width:100%;display:block}
.sr{display:flex;align-items:center;gap:7px;padding:4px 0;border-bottom:1px solid #0a0e18}
.s-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dc{background:#f87171}.dh{background:#fb923c}.dm{background:#fbbf24}
.s-name{font-size:10px;color:#4b5563;width:108px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.s-bw{flex:1;height:5px;background:#111827;border-radius:99px;overflow:hidden}
.s-bar{height:100%;border-radius:99px;transition:width .4s;min-width:0}
.bc{background:#f87171}.bh{background:#fb923c}.bm{background:#fbbf24}
.s-cnt{font-size:10px;color:#374151;width:16px;text-align:right;flex-shrink:0}
.json-pre{font-family:inherit;font-size:11px;line-height:1.6;color:#c9d1d9;white-space:pre-wrap;word-break:break-all;margin:0}
.jk{color:#7dd3fc}.js{color:#86efac}.jn{color:#fdba74}.jb{color:#c084fc}.jnull{color:#4b5563}
.tp{border-top:1px solid #1a2744;display:flex;flex-direction:column;max-height:170px;flex-shrink:0}
.tb{flex:1;overflow-y:auto;padding:8px 20px;display:flex;flex-direction:column;gap:5px;scrollbar-width:thin;scrollbar-color:#1a2744 transparent}
.ts{font-size:12px;line-height:1.6;color:#6b7280;border-left:3px solid #1a2744;padding:3px 9px}
.ts.safe{border-left-color:#2D8F4E}.ts.suspicious{border-left-color:#E8B028}.ts.danger{border-left-color:#F04438}
.ts.ni{animation:sil .3s ease-out}
@keyframes sil{from{transform:translateX(-8px);opacity:0}to{transform:translateX(0);opacity:1}}
mark.hc{background:rgba(248,113,113,.2);color:#fca5a5;text-decoration:underline;text-decoration-color:#ef4444;text-decoration-thickness:2px;border-radius:2px;padding:0 1px}
mark.hh{background:rgba(249,115,22,.15);color:#fdba74;text-decoration:underline;text-decoration-color:#f97316;text-decoration-thickness:2px;border-radius:2px;padding:0 1px}
mark.hm{background:rgba(234,179,8,.15);color:#fde68a;text-decoration:underline;text-decoration-color:#eab308;text-decoration-thickness:2px;border-radius:2px;padding:0 1px}
.nojob{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:#1f2937}
.nojob svg{opacity:.3}
.nojob-text{font-size:13px;color:#374151}
.nojob-sub{font-size:11px;color:#1f2937}
.hide{display:none!important}
.lang-toggle-m{display:flex;gap:0;border:1px solid #1a2744;border-radius:6px;overflow:hidden;margin-right:4px}
.lang-btn-m{font-size:10px;font-weight:700;padding:3px 8px;border:none;background:transparent;color:#374151;cursor:pointer;font-family:inherit;transition:.15s;letter-spacing:.04em}
.lang-btn-m.active{background:#ea580c;color:#fff}
</style>
</head>
<body>
<header class="hd">
  <div class="hd-logo">
    <svg width="20" height="20" viewBox="0 0 30 30" fill="none">
      <circle cx="15" cy="15" r="13.5" stroke="#F97316" stroke-width="1.5" opacity=".2"/>
      <circle cx="15" cy="15" r="9" stroke="#F97316" stroke-width="1.5" opacity=".45"/>
      <circle cx="15" cy="15" r="4.5" stroke="#F97316" stroke-width="1.5" opacity=".75"/>
      <circle cx="15" cy="15" r="2" fill="#F97316"/>
    </svg>
    <span class="hd-name">SONAR — PIPELINE MONITOR</span>
  </div>
  <div class="hd-div"></div>
  <div class="hd-live"><div class="live-dot"></div>LIVE</div>
  <div class="hd-div"></div>
  <div class="hd-stat">JOB <span id="h-job">—</span></div>
  <div class="hd-stat">SEGS <span id="h-segs">0</span></div>
  <div class="hd-stat">RISCO <span id="h-risk" style="color:#374151">—</span></div>
  <div class="hd-stat">MODELO <span id="h-model">—</span></div>
  <div class="hd-ring-wrap">
    <span class="ring-st" id="ring-st"></span>
    <div class="lang-toggle-m">
      <button id="mlang-pt" class="lang-btn-m active" onclick="setMonitorLang('pt')">PT</button>
      <button id="mlang-en" class="lang-btn-m" onclick="setMonitorLang('en')">EN</button>
    </div>
    <select id="ring-lang" style="font-size:11px;background:#0d1117;color:#c9d1d9;border:1px solid #1a2744;border-radius:4px;padding:2px 5px;font-family:inherit;">
      <option value="pt-BR">PT-BR</option>
      <option value="es-419">ES-419</option>
      <option value="en-US">EN-US</option>
    </select>
    <button class="btn-ring" id="btn-ring" onclick="simulateCall()">📞 Simular Ligação</button>
    <a class="hd-link" href="/" target="_blank" id="hd-link-ui">↗ ABRIR UI</a>
  </div>
</header>

<div class="pipe-row">
  <div class="pn" id="pn-audio"><div class="pn-title">AUDIO INPUT</div><div class="pn-sub" id="pns-audio">aguardando...</div></div>
  <div class="pc" id="pc1"></div>
  <div class="pn" id="pn-stt"><div class="pn-title">WHISPER STT</div><div class="pn-sub" id="pns-stt">—</div><div class="pn-val" id="pnv-stt"></div></div>
  <div class="pc" id="pc2"></div>
  <div class="pn" id="pn-gemma"><div class="pn-title">GEMMA 4</div><div class="pn-sub" id="pns-gemma">—</div><div class="pn-val" id="pnv-gemma"></div></div>
  <div class="pc" id="pc3"></div>
  <div class="pn" id="pn-eng"><div class="pn-title">DECISION ENGINE</div><div class="pn-sub" id="pns-eng">—</div><div class="pn-val" id="pnv-eng"></div></div>
  <div class="pc" id="pc4"></div>
  <div class="pn" id="pn-alert"><div class="pn-title">ALERT SYSTEM</div><div class="pn-sub" id="pns-alert">—</div><div class="pn-val" id="pnv-alert"></div></div>
</div>

<div class="nojob" id="nojob">
  <svg width="40" height="40" viewBox="0 0 30 30" fill="none"><circle cx="15" cy="15" r="13.5" stroke="#374151" stroke-width="1.5"/><circle cx="15" cy="15" r="9" stroke="#374151" stroke-width="1.5" opacity=".6"/><circle cx="15" cy="15" r="4.5" stroke="#374151" stroke-width="1.5" opacity=".3"/><circle cx="15" cy="15" r="2" fill="#374151" opacity=".5"/></svg>
  <div class="nojob-text">Aguardando análise...</div>
  <div class="nojob-sub">Inicie uma análise na UI principal para visualizar o pipeline ao vivo.</div>
</div>

<div class="mgrid hide" id="mgrid">
  <div class="mpanel">
    <div class="mp-title">RISCO POR SEGMENTO<span class="badge br" id="risk-badge">—</span></div>
    <div class="mp-body" style="padding:10px 12px"><canvas id="risk-chart" height="185"></canvas></div>
  </div>
  <div class="mpanel">
    <div class="mp-title">SINAIS DETECTADOS</div>
    <div class="mp-body" id="sig-matrix"></div>
  </div>
  <div class="mpanel">
    <div class="mp-title">ÚLTIMO OUTPUT GEMMA</div>
    <div class="mp-body" style="padding:8px 12px"><pre class="json-pre" id="json-out">—</pre></div>
  </div>
</div>

<div class="tp hide" id="tp">
  <div class="mp-title">TRANSCRIÇÃO AO VIVO</div>
  <div class="tb" id="tb"></div>
</div>

<div class="tp hide" id="wtp" style="margin-top:10px">
  <div class="mp-title" style="display:flex;align-items:center;gap:8px">
    TRANSCRIÇÃO COMPLETA WHISPER
    <span id="wtp-lang" style="font-size:9px;font-weight:400;color:#60a5fa;letter-spacing:.04em"></span>
    <span id="wtp-segs" style="font-size:9px;font-weight:400;color:#374151;margin-left:auto"></span>
  </div>
  <div id="wtb" style="font-family:monospace;font-size:11px;color:#c9d1d9;padding:10px 14px;line-height:1.6;max-height:320px;overflow-y:auto"></div>
</div>

<script>
const SEV={financial_request:'c',personal_data_request:'c',family_emergency_claim:'c',unusual_payment_method:'c',remote_access_request:'c',authority_claim:'h',isolation_request:'h',secret_keeping_request:'h',urgency_pressure:'m',emotional_manipulation:'m'};
const SLBL={financial_request:'financial_request',personal_data_request:'personal_data_req',family_emergency_claim:'family_emergency',unusual_payment_method:'unusual_payment',remote_access_request:'remote_access',authority_claim:'authority_claim',isolation_request:'isolation_request',secret_keeping_request:'secret_keeping',urgency_pressure:'urgency_pressure',emotional_manipulation:'emotional_manip'};
const SKW={financial_request:['pix','mpix','transferência','transferencia','pagamento','depósito','deposito','cartão','cartao','crédito','credito','ted','boleto'],personal_data_request:['cpf','senha','código','codigo','sms','otp','pin','dados','confirmar','verificação','verificacao'],family_emergency_claim:['acidente','hospital','preso','sequestro','emergência','emergencia','filho','filha'],unusual_payment_method:['gift card','bitcoin','criptomoeda'],remote_access_request:['anydesk','teamviewer','instale','instalar','acesso remoto'],authority_claim:['banco','polícia','policia','federal','inss','receita','governo','bradesco','itaú','itau','santander','caixa','nubank'],isolation_request:['não desligue','nao desligue','fique na linha','não conte','nao conte'],secret_keeping_request:['segredo','esconda','não diga','nao diga'],urgency_pressure:['urgente','agora','imediatamente','rapidamente','minutos'],emotional_manipulation:['medo','perigo','grave','sério','serio','preocupada','preocupado']};
const ALL_SIG=Object.keys(SEV);
const RV={safe:0,suspicious:1,danger:2};
const RC={safe:'#4ade80',suspicious:'#fbbf24',danger:'#f87171'};
let jobId=null,allSegs=[],seen=0,sigCounts={};
ALL_SIG.forEach(s=>sigCounts[s]=0);
initMatrix();
setInterval(checkLatest,3000);checkLatest();

async function checkLatest(){
  try{const d=await(await fetch('/api/latest_job')).json();if(!d.ok||!d.job_id)return;if(d.job_id!==jobId){jobId=d.job_id;resetMonitor(d);}}catch(_){}
}
function renderWhisperTranscript(log, language) {
  if (!log || !log.length) return;
  const wtb = document.getElementById('wtb');
  const wtp = document.getElementById('wtp');
  wtb.innerHTML = log.map((entry, i) => {
    const num = String(i + 1).padStart(2, '0');
    const txt = String(entry.text || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    return `<div style="margin-bottom:6px"><span style="color:#374151;user-select:none">[S${num}] </span>${txt}</div>`;
  }).join('');
  document.getElementById('wtp-lang').textContent = language ? language.toUpperCase() : '';
  document.getElementById('wtp-segs').textContent = log.length + ' segmento' + (log.length !== 1 ? 's' : '');
  wtp.classList.remove('hide');
}

function resetMonitor(meta){
  allSegs=[];seen=0;ALL_SIG.forEach(s=>sigCounts[s]=0);
  document.getElementById('nojob').classList.add('hide');
  document.getElementById('mgrid').classList.remove('hide');
  document.getElementById('tp').classList.remove('hide');
  document.getElementById('wtp').classList.add('hide');
  document.getElementById('wtb').innerHTML='';
  document.getElementById('tb').innerHTML='';document.getElementById('json-out').innerHTML='—';
  document.getElementById('h-job').textContent=jobId.slice(-8);
  document.getElementById('h-model').textContent=meta.model_name||'—';
  document.getElementById('h-segs').textContent='0';
  document.getElementById('h-risk').textContent='—';document.getElementById('h-risk').style.color='#374151';
  pipeReset();updateMatrix();drawChart();
  if(window._si)clearInterval(window._si);
  window._si=setInterval(pollStatus,2000);pollStatus();
}
async function pollStatus(){
  if(!jobId)return;
  try{
    const d=await(await fetch('/api/status/'+jobId)).json();
    if(!d.ok)return;
    document.getElementById('h-segs').textContent=d.segments.length;
    updatePipe(d);
    const ns=d.segments.slice(seen);
    for(const seg of ns){seen++;allSegs.push(seg);for(const s of(seg.classification.signals_detected||[]))if(s in sigCounts)sigCounts[s]++;addTSeg(seg,seen);updateJSON(seg);}
    if(ns.length){updateMatrix();drawChart();}
    const fr=d.final?.overall_risk||(d.segments.length?d.segments[d.segments.length-1].call_state.overall_risk:null);
    if(fr){document.getElementById('h-risk').textContent=fr.toUpperCase();document.getElementById('h-risk').style.color=RC[fr]||'#c9d1d9';const rb=document.getElementById('risk-badge');rb.textContent=fr.toUpperCase();rb.className='badge '+(fr==='danger'?'bd':fr==='suspicious'?'bw':'bs');}
    if(d.transcript_log && d.transcript_log.length) renderWhisperTranscript(d.transcript_log, d.language);
    if(d.status==='done'||d.status==='error')clearInterval(window._si);
  }catch(_){}
}
function updatePipe(d){
  const st=d.stage,segs=d.segments;
  ['pn-audio','pn-stt','pn-gemma','pn-eng','pn-alert'].forEach(id=>document.getElementById(id).className='pn');
  ['pc1','pc2','pc3','pc4'].forEach(id=>document.getElementById(id).className='pc');
  if(st==='starting'||st==='preflight'){AN('pn-audio');FC('pc1');document.getElementById('pns-audio').textContent=st==='preflight'?'verificando...':'carregando...';}
  else if(st==='stt'){DN('pn-audio');FC('pc1');AN('pn-stt');document.getElementById('pns-stt').textContent=d.whisper_model?'whisper-'+d.whisper_model:'—';}
  else if(st==='classifying'){
    DN('pn-audio');DN('pn-stt');document.getElementById('pns-stt').textContent=d.total+' seg.';
    DFC('pc1');FC('pc2');AN('pn-gemma');document.getElementById('pns-gemma').textContent=d.model_name||'—';document.getElementById('pnv-gemma').textContent='seg '+segs.length+'/'+d.total;
    if(segs.length>0){FC('pc3');const cr=segs[segs.length-1].call_state.overall_risk;document.getElementById('pns-eng').textContent='rolling window';document.getElementById('pnv-eng').textContent=cr.toUpperCase();if(cr==='danger'){NR('pn-eng','dng');FC('pc4');NR('pn-alert','dng');document.getElementById('pns-alert').textContent='ALERTA VERMELHO';}else if(cr==='suspicious'){NR('pn-eng','warn');document.getElementById('pns-alert').textContent='suspeito';}else{AN('pn-eng');document.getElementById('pns-alert').textContent='monitorando';}}
  }else if(st==='done'){
    ['pn-audio','pn-stt','pn-gemma','pn-eng'].forEach(id=>DN(id));['pc1','pc2','pc3','pc4'].forEach(id=>DFC(id));
    document.getElementById('pns-stt').textContent=d.total+' segmentos';document.getElementById('pns-gemma').textContent=segs.length+' classificações';
    const fr=d.final?.overall_risk||'safe';if(fr==='danger'){NR('pn-alert','dng');document.getElementById('pnv-alert').textContent='ALERTA VERMELHO';}else if(fr==='suspicious'){NR('pn-alert','warn');document.getElementById('pnv-alert').textContent=fr.toUpperCase();}else{DN('pn-alert');document.getElementById('pnv-alert').textContent=fr.toUpperCase();}
  }
}
function AN(id){document.getElementById(id).classList.add('active')}
function DN(id){document.getElementById(id).classList.add('done')}
function NR(id,cls){document.getElementById(id).className='pn '+cls}
function FC(id){document.getElementById(id).classList.add('flow')}
function DFC(id){document.getElementById(id).classList.add('done-f')}
function pipeReset(){AN('pn-audio');FC('pc1');document.getElementById('pns-audio').textContent='carregando...';['pn-stt','pn-gemma','pn-eng','pn-alert'].forEach(id=>{document.getElementById(id).className='pn';});['pc2','pc3','pc4'].forEach(id=>document.getElementById(id).className='pc');}
function initMatrix(){document.getElementById('sig-matrix').innerHTML=ALL_SIG.map(sig=>{const s=SEV[sig];return `<div class="sr" id="sr-${sig}" style="opacity:.3"><div class="s-dot d${s}"></div><div class="s-name">${SLBL[sig]||sig}</div><div class="s-bw"><div class="s-bar b${s}" id="sb-${sig}" style="width:0%"></div></div><div class="s-cnt" id="sc-${sig}">0</div></div>`;}).join('');}
function updateMatrix(){const mx=Math.max(1,...Object.values(sigCounts));for(const s of ALL_SIG){const cnt=sigCounts[s],pct=(cnt/mx)*100;const bar=document.getElementById('sb-'+s),lbl=document.getElementById('sc-'+s),row=document.getElementById('sr-'+s);if(bar)bar.style.width=pct+'%';if(lbl)lbl.textContent=cnt;if(row)row.style.opacity=cnt>0?'1':'0.25';}}
function drawChart(){
  const canvas=document.getElementById('risk-chart');if(!canvas)return;
  canvas.width=canvas.offsetWidth||420;
  const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
  const P={t:18,r:12,b:28,l:48},cW=W-P.l-P.r,cH=H-P.t-P.b;
  ctx.clearRect(0,0,W,H);
  const zH=cH/3;
  [[2,'rgba(248,113,113,.06)'],[1,'rgba(251,191,36,.06)'],[0,'rgba(74,222,128,.06)']].forEach(([v,c])=>{ctx.fillStyle=c;ctx.fillRect(P.l,P.t+(2-v)*zH,cW,zH);});
  [['DANGER',2,'#f87171'],['SUSP.',1,'#fbbf24'],['SAFE',0,'#4ade80']].forEach(([lbl,v,color])=>{const y=P.t+(2-v)*zH+zH/2;ctx.strokeStyle='#1a2744';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+cW,y);ctx.stroke();ctx.fillStyle=color;ctx.font='9px monospace';ctx.textAlign='right';ctx.fillText(lbl,P.l-5,y+3);});
  if(!allSegs.length)return;
  const n=allSegs.length,xs=n>1?cW/(n-1):cW/2;
  const pts=allSegs.map((seg,i)=>{const rv=RV[seg.call_state.overall_risk]||0;return{x:P.l+(n>1?i*xs:cW/2),y:P.t+(2-rv)*zH+zH/2,risk:seg.call_state.overall_risk};});
  ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.lineTo(pts[pts.length-1].x,P.t+cH);ctx.lineTo(P.l,P.t+cH);ctx.closePath();
  const g=ctx.createLinearGradient(0,P.t,0,P.t+cH);g.addColorStop(0,'rgba(248,113,113,.18)');g.addColorStop(.5,'rgba(251,191,36,.12)');g.addColorStop(1,'rgba(74,222,128,.08)');ctx.fillStyle=g;ctx.fill();
  ctx.beginPath();ctx.strokeStyle='#38bdf8';ctx.lineWidth=2;pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.stroke();
  pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fillStyle=RC[p.risk]||'#374151';ctx.fill();ctx.strokeStyle='#060a10';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#374151';ctx.font='9px monospace';ctx.textAlign='center';ctx.fillText(i+1,p.x,P.t+cH+16);});
}
function addTSeg(seg,num){const c=seg.classification,risk=c.risk_level||'safe',sigs=c.signals_detected||[];const div=document.createElement('div');div.className=`ts ${risk} ni`;div.innerHTML=`<strong style="color:#374151;font-size:9px">S${String(num).padStart(2,'0')}</strong> ${hlTxt(c.transcript_excerpt||'',sigs)}`;const tb=document.getElementById('tb');tb.appendChild(div);tb.scrollTop=tb.scrollHeight;}
function updateJSON(seg){const el=document.getElementById('json-out');if(!el)return;el.innerHTML=syntaxHL(seg.classification);}
function syntaxHL(obj){return JSON.stringify(obj,null,2).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"([^"]+)":/g,'<span class="jk">"$1":</span>').replace(/: "((?:[^"\\]|\\.)*)"/g,(_,v)=>`: <span class="js">"${v}"</span>`).replace(/: (-?\d+\.?\d*)/g,'<span>: </span><span class="jn">$1</span>').replace(/: (true|false)/g,'<span>: </span><span class="jb">$1</span>').replace(/: null/g,': <span class="jnull">null</span>');}
function hlTxt(text,signals){if(!signals||!signals.length)return escM(text);const lt=text.toLowerCase(),ivs=[];for(const sig of signals){const s=SEV[sig]||'m';const cls=s==='c'?'hc':s==='h'?'hh':'hm';for(const kw of(SKW[sig]||[])){let pos=0;while((pos=lt.indexOf(kw,pos))!==-1){ivs.push({start:pos,end:pos+kw.length,cls});pos+=kw.length;}}}if(!ivs.length)return escM(text);ivs.sort((a,b)=>a.start-b.start);let out='',cur=0;for(const iv of ivs){if(iv.start<cur)continue;out+=escM(text.slice(cur,iv.start));out+=`<mark class="${iv.cls}">${escM(text.slice(iv.start,iv.end))}</mark>`;cur=iv.end;}return out+escM(text.slice(cur));}
function escM(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

// ── Monitor i18n ──────────────────────────────────────────────────────────────
let _mlang = localStorage.getItem('sonar_lang') || 'pt';
let _mRingState = 'idle';
const MT = {
  pt: { simulate_call:'📞 Simular Ligação', open_ui:'↗ ABRIR UI' },
  en: { simulate_call:'📞 Simulate Call', open_ui:'↗ OPEN UI' }
};
function mt(key) { return (MT[_mlang] || MT.pt)[key] || key; }
function setMonitorLang(lang) {
  _mlang = lang;
  localStorage.setItem('sonar_lang', lang);
  document.getElementById('mlang-pt').classList.toggle('active', lang === 'pt');
  document.getElementById('mlang-en').classList.toggle('active', lang === 'en');
  const btn = document.getElementById('btn-ring');
  if (btn && _mRingState === 'idle') btn.textContent = mt('simulate_call');
  const lnk = document.getElementById('hd-link-ui');
  if (lnk) lnk.textContent = mt('open_ui');
}
(function() {
  const stored = localStorage.getItem('sonar_lang') || 'pt';
  if (stored === 'en') setMonitorLang('en');
})();

// ── Simulated incoming call (monitor side) ─────────────────────────────────
async function simulateCall() {
  const btn = document.getElementById('btn-ring');
  if (_mRingState === 'ringing') return;
  _setRingState('ringing');
  const lang = document.getElementById('ring-lang').value;
  try {
    const r = await fetch('/api/test/ring', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({caller_id:'0800 555-1234', language: lang, model:'google/gemma-4-e2b', whisper_model:'tiny'})
    });
    const d = await r.json();
    if (!d.ok) _setRingState('idle');
  } catch(e) { _setRingState('idle'); }
}
function _setRingState(state) {
  _mRingState = state;
  const btn = document.getElementById('btn-ring'), st = document.getElementById('ring-st');
  if (state === 'idle') {
    btn.disabled = false; btn.textContent = mt('simulate_call'); btn.className = 'btn-ring'; st.textContent = '';
  } else if (state === 'ringing') {
    btn.disabled = true; btn.textContent = '⏳ Chamando...'; btn.className = 'btn-ring ringing'; st.textContent = 'aguardando atendimento'; st.style.color = '#fbbf24';
  } else if (state === 'answered') {
    btn.disabled = false; btn.textContent = mt('simulate_call'); btn.className = 'btn-ring'; st.textContent = '✓ atendida'; st.style.color = '#4ade80';
    setTimeout(() => { if (_mRingState === 'answered') _setRingState('idle'); }, 3000);
  } else if (state === 'rejected') {
    btn.disabled = false; btn.textContent = mt('simulate_call'); btn.className = 'btn-ring'; st.textContent = '✗ rejeitada'; st.style.color = '#f87171';
    setTimeout(() => { if (_mRingState === 'rejected') _setRingState('idle'); }, 2500);
  }
}
setInterval(async () => {
  if (_mRingState !== 'ringing') return;
  try {
    const d = await (await fetch('/api/incoming-call')).json();
    if (!d.ringing) {
      const lj = await (await fetch('/api/latest_job')).json();
      _setRingState(lj.ok && lj.job_id ? 'answered' : 'rejected');
    }
  } catch(e) {}
}, 1500);
</script>
</body></html>'''


def _contato_html() -> str:
    return r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sonar — Alerta de Emergência</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#E5DDD5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:0}
.demo-banner{width:100%;background:#1a1a1a;color:#F97316;font-size:11px;font-weight:700;letter-spacing:.1em;text-align:center;padding:6px 16px;display:flex;align-items:center;justify-content:center;gap:8px;flex-shrink:0}
.demo-dot{width:7px;height:7px;border-radius:50%;background:#F97316;animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.phone-shell{width:100%;max-width:420px;flex:1;display:flex;flex-direction:column;background:#fff;min-height:calc(100vh - 30px);box-shadow:0 0 40px rgba(0,0,0,.25)}
.phone-header{background:#075E54;color:#fff;padding:12px 16px 10px;display:flex;align-items:center;gap:12px;flex-shrink:0}
.ph-back{font-size:20px;cursor:pointer;opacity:.85}
.ph-avatar{width:40px;height:40px;border-radius:50%;background:#25D366;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;flex-shrink:0}
.ph-info{flex:1}
.ph-name{font-size:15px;font-weight:600}
.ph-sub{font-size:12px;opacity:.75}
.ph-icons{display:flex;gap:18px;align-items:center}
.ph-icon{width:20px;height:20px;opacity:.85;cursor:pointer}
.chat-area{flex:1;overflow-y:auto;padding:10px 12px 12px;display:flex;flex-direction:column;gap:6px;background:#E5DDD5}
.chat-date{text-align:center;font-size:11px;color:#667781;background:rgba(255,255,255,.65);border-radius:8px;padding:3px 10px;align-self:center;margin:4px 0}
.msg-wrap{display:flex;flex-direction:column;max-width:82%}
.msg-wrap.system-msg{align-self:center;max-width:90%}
.msg-bubble{border-radius:8px;padding:8px 10px;font-size:13.5px;line-height:1.4;position:relative;word-break:break-word}
.msg-time{font-size:10px;color:rgba(0,0,0,.45);text-align:right;margin-top:2px;padding-right:2px}
.msg-wrap.incoming{align-self:flex-start}
.msg-wrap.incoming .msg-bubble{background:#fff;border-top-left-radius:2px}
.msg-wrap.outgoing{align-self:flex-end}
.msg-wrap.outgoing .msg-bubble{background:#DCF8C6;border-top-right-radius:2px}
.msg-wrap.system-msg .msg-bubble{background:rgba(255,255,255,.75);color:#555;font-size:12px;text-align:center;border-radius:8px}
.alert-bubble{background:#FFF3CD !important;border-left:3px solid #F97316 !important}
.alert-header{font-weight:700;font-size:13px;color:#c2410c;margin-bottom:4px}
.alert-danger .msg-bubble{background:#FFEBEE !important;border-left:3px solid #DC2626 !important}
.alert-danger .alert-header{color:#DC2626}
.signal-pill{display:inline-block;background:#F97316;color:#fff;font-size:10px;border-radius:4px;padding:1px 6px;margin:2px 2px 2px 0;font-weight:600}
.signal-pill.danger-pill{background:#DC2626}
.excerpt-text{font-size:12px;color:#374151;margin-top:5px;font-style:italic;border-left:2px solid #D1D5DB;padding-left:6px}
.read-ticks{color:#53bdeb;font-size:11px;margin-left:3px}
.demo-controls{background:#fff;border-top:1px solid #E5E7EB;padding:10px 12px;display:flex;gap:8px;flex-shrink:0}
.demo-btn{flex:1;padding:9px 10px;border-radius:8px;border:none;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:.15s}
.btn-fire{background:#F97316;color:#fff}
.btn-fire:hover{background:#EA580C}
.btn-fire.danger{background:#DC2626}
.btn-fire.danger:hover{background:#B91C1C}
.btn-clear{background:#F3F4F6;color:#6B7280;border:1px solid #E5E7EB}
.btn-clear:hover{background:#E5E7EB}
.empty-hint{align-self:center;color:#667781;font-size:13px;text-align:center;padding:40px 20px;line-height:1.6}
.typing-indicator{display:none;align-self:flex-start;background:#fff;border-radius:18px;padding:10px 14px;align-items:center;gap:4px}
.typing-indicator.show{display:flex}
.dot{width:7px;height:7px;border-radius:50%;background:#9CA3AF;animation:bounce 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg-wrap{animation:fadeIn .3s ease}
.lang-toggle-c{display:flex;gap:0;border:1px solid #374151;border-radius:6px;overflow:hidden;margin-left:8px}
.lang-btn-c{font-size:10px;font-weight:700;padding:3px 8px;border:none;background:transparent;color:#9CA3AF;cursor:pointer;font-family:inherit}
.lang-btn-c.active{background:#F97316;color:#fff}
</style>
</head>
<body>
<div class="demo-banner">
  <span class="demo-dot"></span>
  <span id="banner-text">MODO DEMONSTRAÇÃO — Sonar · Proteção Contra Golpes</span>
  <div class="lang-toggle-c" style="margin-left:auto">
    <button id="clang-pt" class="lang-btn-c active" onclick="setContactLang('pt')">PT</button>
    <button id="clang-en" class="lang-btn-c" onclick="setContactLang('en')">EN</button>
  </div>
</div>
<div class="phone-shell">
  <div class="phone-header">
    <span class="ph-back">←</span>
    <div class="ph-avatar">S</div>
    <div class="ph-info">
      <div class="ph-name">Sonar Proteção</div>
      <div class="ph-sub" id="status-sub">aguardando alertas...</div>
    </div>
    <div class="ph-icons">
      <svg class="ph-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.82 19.79 19.79 0 01.5 1.18 2 2 0 012.18 0h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.91 7.15a16 16 0 006.29 6.29l1.42-1.42a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/></svg>
      <svg class="ph-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/></svg>
    </div>
  </div>
  <div class="chat-area" id="chat-area">
    <div class="chat-date">Hoje</div>
    <div class="msg-wrap system-msg">
      <div class="msg-bubble">
        🔒 Esta conversa está simulando o canal de alertas do Sonar.<br>
        Os alertas aparecem aqui automaticamente quando o app detecta risco.
      </div>
    </div>
    <div class="empty-hint" id="empty-hint">
      Nenhum alerta ainda.<br>Use os botões abaixo para simular<br>ou inicie uma chamada na <a href="/" target="_blank" style="color:#F97316">UI principal</a>.
    </div>
    <div class="typing-indicator" id="typing">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
  </div>
  <div class="demo-controls">
    <button class="demo-btn btn-fire" onclick="fireAlert('suspicious')">⚠️ Simular Suspeita</button>
    <button class="demo-btn btn-fire danger" onclick="fireAlert('danger')">🚨 Simular Golpe</button>
    <button class="demo-btn btn-clear" onclick="clearChat()">✕ Limpar</button>
  </div>
</div>
<script>
const CT = {
  pt: {
    banner: 'MODO DEMONSTRAÇÃO — Sonar · Proteção Contra Golpes',
    status_sub: 'aguardando alertas...',
    alerted_now: 'alerta recebido agora',
    empty_hint: 'Nenhum alerta ainda.<br>Use os botões abaixo para simular<br>ou inicie uma chamada na <a href="/" target="_blank" style="color:#F97316">UI principal</a>.',
    btn_suspicious: '⚠️ Simular Suspeita', btn_danger: '🚨 Simular Golpe', btn_clear: '✕ Limpar',
    risk_suspicious_label: '⚠️ CHAMADA SUSPEITA DETECTADA', risk_danger_label: '🚨 GOLPE EM ANDAMENTO — AÇÃO URGENTE',
    action_title: '💡 O que fazer agora:', action_body: '1. Ligue imediatamente para confirmar que está bem<br>2. Oriente a desligar a chamada suspeita<br>3. Nenhum banco pede senha ou código por telefone',
    detected_msg: 'O Sonar detectou sinais de golpe em uma chamada ativa.',
    duration_suffix: 's de chamada detectados',
    signals: {
      financial_request: 'Pedido de transferência / PIX', personal_data_request: 'Solicitação de dados pessoais',
      authority_claim: 'Falsa autoridade (banco, governo, INSS)', urgency_pressure: 'Pressão de urgência',
      isolation_request: 'Pedido para não desligar', secret_keeping_request: 'Pedido de sigilo',
      family_emergency_claim: 'Falsa emergência familiar', unusual_payment_method: 'Método de pagamento incomum',
      remote_access_request: 'Pedido de acesso remoto', emotional_manipulation: 'Manipulação emocional',
    }
  },
  en: {
    banner: 'DEMO MODE — Sonar · Scam Protection',
    status_sub: 'waiting for alerts...',
    alerted_now: 'alert received now',
    empty_hint: 'No alerts yet.<br>Use the buttons below to simulate<br>or start a call in the <a href="/" target="_blank" style="color:#F97316">main UI</a>.',
    btn_suspicious: '⚠️ Simulate Suspicious', btn_danger: '🚨 Simulate Scam', btn_clear: '✕ Clear',
    risk_suspicious_label: '⚠️ SUSPICIOUS CALL DETECTED', risk_danger_label: '🚨 SCAM IN PROGRESS — URGENT ACTION',
    action_title: '💡 What to do now:', action_body: '1. Call immediately to confirm they are safe<br>2. Tell them to hang up the suspicious call<br>3. No bank ever asks for passwords or codes by phone',
    detected_msg: 'Sonar detected scam signals in an active call.',
    duration_suffix: 's of call detected',
    signals: {
      financial_request: 'Financial transfer / payment request', personal_data_request: 'Personal data request',
      authority_claim: 'False authority (bank, government, INSS)', urgency_pressure: 'Urgency pressure',
      isolation_request: 'Do not hang up request', secret_keeping_request: 'Secrecy request',
      family_emergency_claim: 'Fake family emergency', unusual_payment_method: 'Unusual payment method',
      remote_access_request: 'Remote access request', emotional_manipulation: 'Emotional manipulation',
    }
  }
};
let _clang = localStorage.getItem('sonar_lang') || 'pt';

function setContactLang(lang) {
  _clang = lang;
  localStorage.setItem('sonar_lang', lang);
  document.getElementById('clang-pt').classList.toggle('active', lang === 'pt');
  document.getElementById('clang-en').classList.toggle('active', lang === 'en');
  const L = CT[lang] || CT.pt;
  document.getElementById('banner-text').textContent = L.banner;
  document.getElementById('status-sub').textContent = L.status_sub;
  document.getElementById('empty-hint').innerHTML = L.empty_hint;
  const btns = document.querySelectorAll('.demo-btn');
  if (btns[0]) btns[0].textContent = L.btn_suspicious;
  if (btns[1]) btns[1].textContent = L.btn_danger;
  if (btns[2]) btns[2].textContent = L.btn_clear;
}

function ct(key) { return (CT[_clang] || CT.pt)[key] || key; }
function ct_signal(sig) { return ((CT[_clang] || CT.pt).signals || {})[sig] || sig; }

const _SIGNAL_LABELS = {
  financial_request: 'Pedido de transferência / PIX',
  personal_data_request: 'Solicitação de dados pessoais',
  authority_claim: 'Falsa autoridade (banco, governo, INSS)',
  urgency_pressure: 'Pressão de urgência',
  isolation_request: 'Pedido para não desligar',
  secret_keeping_request: 'Pedido de sigilo',
  family_emergency_claim: 'Falsa emergência familiar',
  unusual_payment_method: 'Método de pagamento incomum',
  remote_access_request: 'Pedido de acesso remoto',
  emotional_manipulation: 'Manipulação emocional',
};

let _lastTs = 0;
let _rendered = new Set();

function fmt_time(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('pt-BR', {hour:'2-digit', minute:'2-digit'});
}

function risk_label(risk) {
  return risk === 'danger' ? ct('risk_danger_label') : ct('risk_suspicious_label');
}

function render_alert(a) {
  const isDanger = a.risk === 'danger';
  const pills = (a.signals || []).map(s =>
    `<span class="signal-pill${isDanger ? ' danger-pill' : ''}">${ct_signal(s)}</span>`
  ).join('');
  const excerpt = a.excerpt ? `<div class="excerpt-text">"${esc(a.excerpt)}"</div>` : '';
  const dur = a.duration_seconds ? `<div style="font-size:11px;color:#6B7280;margin-top:4px">⏱ ${Math.round(a.duration_seconds)}${ct('duration_suffix')}</div>` : '';
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap incoming${isDanger ? ' alert-danger' : ''}`;
  wrap.dataset.alertId = a.id;
  wrap.innerHTML = `
    <div class="msg-bubble alert-bubble">
      <div class="alert-header">${esc(risk_label(a.risk))}</div>
      <div style="font-size:12px;color:#374151;margin-bottom:4px;">${esc(ct('detected_msg'))}</div>
      <div>${pills}</div>
      ${excerpt}
      ${dur}
      <div style="margin-top:8px;font-size:12px;font-weight:600;color:#111;">${esc(ct('action_title'))}</div>
      <div style="font-size:12px;color:#374151;margin-top:2px;line-height:1.5;">
        ${ct('action_body')}
      </div>
    </div>
    <div class="msg-time">${fmt_time(a.timestamp)} <span class="read-ticks">✓✓</span></div>`;
  return wrap;
}

function show_typing() {
  document.getElementById('typing').classList.add('show');
  scroll_bottom();
}
function hide_typing() {
  document.getElementById('typing').classList.remove('show');
}

function scroll_bottom() {
  const ca = document.getElementById('chat-area');
  ca.scrollTop = ca.scrollHeight;
}

function add_alert(a) {
  if (_rendered.has(a.id)) return;
  _rendered.add(a.id);
  document.getElementById('empty-hint').style.display = 'none';
  show_typing();
  setTimeout(() => {
    hide_typing();
    const ca = document.getElementById('chat-area');
    const typing = document.getElementById('typing');
    ca.insertBefore(render_alert(a), typing);
    scroll_bottom();
    document.getElementById('status-sub').textContent = ct('alerted_now');
  }, 900);
}

async function poll() {
  try {
    const r = await fetch(`/api/alert-log?since=${_lastTs}`);
    const d = await r.json();
    if (d.ok && d.alerts && d.alerts.length) {
      d.alerts.forEach(a => {
        if (a.timestamp > _lastTs) _lastTs = a.timestamp;
        add_alert(a);
      });
    }
  } catch(e) {}
}

async function fireAlert(risk) {
  const presets = {
    suspicious: {
      risk: 'suspicious',
      signals: ['urgency_pressure', 'authority_claim'],
      excerpt: 'Senhor, o senhor precisa confirmar sua senha agora para não bloquear sua conta.',
      duration_seconds: 38,
    },
    danger: {
      risk: 'danger',
      signals: ['financial_request', 'urgency_pressure', 'authority_claim', 'personal_data_request'],
      excerpt: 'Me passou o código? Precisa ser agora, a conta vai bloquear em 5 minutos.',
      duration_seconds: 72,
    },
  };
  try {
    const payload = presets[risk] || presets.danger;
    await fetch('/api/test/alert', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
  } catch(e) {}
}

function clearChat() {
  _rendered.clear();
  _lastTs = 0;
  const ca = document.getElementById('chat-area');
  const keep = ca.querySelectorAll('.chat-date, .system-msg, #empty-hint, #typing');
  [...ca.children].forEach(c => { if (![...keep].includes(c)) c.remove(); });
  document.getElementById('empty-hint').style.display = '';
  document.getElementById('status-sub').textContent = ct('status_sub');
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

poll();
setInterval(poll, 2000);
setContactLang(_clang);
</script>
</body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
