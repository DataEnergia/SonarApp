# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from pathlib import Path

import demo_run
from shared.schemas.classification import AlertLevel, CallReport, CallState, Language, RiskLevel, ScamSignal


def test_run_demo_stops_when_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "missing.wav"

    monkeypatch.setattr(
        demo_run,
        "run_preflight",
        lambda **_kwargs: {"ok": False, "checks": [{"name": "audio_file", "status": "fail", "message": "missing"}]},
    )

    summary = demo_run.run_demo(audio_path, Language.PT_BR, tmp_path / "out")

    assert summary["ok"] is False
    assert summary["stage"] == "preflight"
    assert Path(summary["preflight_report"]).exists()


def test_run_demo_writes_summary_after_pipeline(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"fake")

    def fake_pipeline(**kwargs):
        output_path = kwargs["output_path"]
        report = CallReport(
            call_id="call_test",
            language=Language.PT_BR,
            duration_seconds=1.0,
            segments=[],
            final_state=CallState(
                call_id="call_test",
                overall_risk=RiskLevel.DANGER,
                top_signals=[ScamSignal.FINANCIAL_REQUEST],
                alert_level=AlertLevel.RED,
                should_play_audio_alert=True,
                rationale_for_user="Sinais fortes.",
                rationale_for_audit_log="risk=danger",
            ),
            model_versions={"classifier": "gemma4:e2b"},
        )
        output_path.write_text(report.model_dump_json(), encoding="utf-8")
        return report

    monkeypatch.setattr(demo_run, "run_preflight", lambda **_kwargs: {"ok": True, "checks": []})
    monkeypatch.setattr(demo_run, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(demo_run, "render_call_report", lambda _report, output: output.write_text("# report", encoding="utf-8"))

    summary = demo_run.run_demo(audio_path, Language.PT_BR, tmp_path / "out")

    assert summary["ok"] is True
    assert summary["final_risk"] == "danger"
    assert summary["alert_level"] == "red"
    assert Path(summary["call_report"]).exists()
    assert Path(summary["markdown_report"]).exists()
