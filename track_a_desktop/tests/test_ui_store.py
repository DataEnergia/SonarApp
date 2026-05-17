# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from ui_local.store import add_to_dataset, append_jsonl, read_jsonl, save_feedback, summarize_call_report


def test_jsonl_history_reads_newest_first(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"

    append_jsonl(path, {"id": 1})
    append_jsonl(path, {"id": 2})

    assert read_jsonl(path) == [{"id": 2}, {"id": 1}]
    assert read_jsonl(path, limit=1) == [{"id": 2}]


def test_save_feedback_adds_metadata(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"

    saved = save_feedback(path, {"recording_id": "rec_1", "user_label": "scam"})

    assert saved["feedback_id"].startswith("fb_")
    assert saved["created_at"]
    assert read_jsonl(path)[0]["recording_id"] == "rec_1"


def test_summarize_call_report_extracts_ui_fields(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "call_id": "call_1",
                "language": "pt-BR",
                "duration_seconds": 5.0,
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "risk_level": "danger",
                        "confidence": 0.9,
                        "signals_detected": ["financial_request"],
                        "transcript_excerpt": "Faca um Pix.",
                        "suggested_action_for_user": "Pare.",
                    }
                ],
                "final_state": {
                    "overall_risk": "danger",
                    "alert_level": "red",
                    "should_play_audio_alert": True,
                    "rationale_for_user": "Sinais fortes.",
                    "top_signals": ["financial_request"],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_call_report(report_path)

    assert summary["final_risk"] == "danger"
    assert summary["alert_level"] == "red"
    assert summary["segments"][0]["signals_detected"] == ["financial_request"]


def test_add_to_dataset_creates_wav_entry(tmp_path: Path) -> None:
    audio_src = tmp_path / "rec_20260513_000001_abcd1234.wav"
    audio_src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")  # minimal placeholder

    dataset_audio_dir = tmp_path / "audio"
    ground_truth_path = tmp_path / "ground_truth.json"

    entry = add_to_dataset(
        audio_src=audio_src,
        report_path=None,
        dataset_audio_dir=dataset_audio_dir,
        ground_truth_path=ground_truth_path,
        recording_id="rec_20260513_000001_abcd1234",
        user_label="scam",
        user_corrected_risk="danger",
        language="pt-BR",
        notes="test note",
    )

    assert entry["type"] == "scam"
    assert entry["final_risk_level"] == "danger"
    assert entry["language"] == "pt-BR"
    assert entry["user_notes"] == "test note"
    assert entry["source"] == "user_recording"
    assert entry["split"] == "train"

    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    assert len(gt["samples"]) == 1
    assert gt["samples"][0]["recording_id"] == "rec_20260513_000001_abcd1234"
    assert (dataset_audio_dir / entry["audio_file"]).exists()


def test_add_to_dataset_deduplicates_recording_id(tmp_path: Path) -> None:
    audio_src = tmp_path / "rec_20260513_000002_bbbb0000.wav"
    audio_src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    dataset_audio_dir = tmp_path / "audio"
    ground_truth_path = tmp_path / "ground_truth.json"

    kwargs = dict(
        audio_src=audio_src,
        report_path=None,
        dataset_audio_dir=dataset_audio_dir,
        ground_truth_path=ground_truth_path,
        recording_id="rec_20260513_000002_bbbb0000",
        user_label="legitimate",
        user_corrected_risk="safe",
        language="pt-BR",
        notes="",
    )
    add_to_dataset(**kwargs)
    add_to_dataset(**kwargs)  # second call should replace, not duplicate

    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    assert len(gt["samples"]) == 1


def test_add_to_dataset_with_report_extracts_signals(tmp_path: Path) -> None:
    audio_src = tmp_path / "rec_20260513_000003_cccc1111.wav"
    audio_src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    report_path = tmp_path / "call_report.json"
    report_path.write_text(
        json.dumps(
            {
                "call_id": "call_x",
                "duration_seconds": 12.0,
                "final_state": {"top_signals": ["authority_claim", "urgency_pressure"]},
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    dataset_audio_dir = tmp_path / "audio"
    ground_truth_path = tmp_path / "ground_truth.json"

    entry = add_to_dataset(
        audio_src=audio_src,
        report_path=report_path,
        dataset_audio_dir=dataset_audio_dir,
        ground_truth_path=ground_truth_path,
        recording_id="rec_20260513_000003_cccc1111",
        user_label="scam",
        user_corrected_risk="danger",
        language="pt-BR",
    )

    assert entry["primary_signal"] == "authority_claim"
    assert entry["all_signals"] == ["authority_claim", "urgency_pressure"]
    assert entry["duration_seconds"] == 12.0
