# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from render_report import render_call_report


def test_render_call_report_outputs_decision_and_segments(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "report.md"
    report_path.write_text(
        json.dumps(
            {
                "call_id": "call_test",
                "language": "pt-BR",
                "duration_seconds": 12.5,
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "transcript_excerpt": "Faca um Pix agora.",
                        "signals_detected": ["financial_request", "urgency_pressure"],
                        "risk_level": "danger",
                        "confidence": 0.9,
                        "reasoning": "Pedido financeiro urgente.",
                        "suggested_action_for_user": "Pare e confirme pelo banco.",
                        "needs_deeper_analysis": False,
                    }
                ],
                "final_state": {
                    "call_id": "call_test",
                    "overall_risk": "danger",
                    "top_signals": ["financial_request", "urgency_pressure"],
                    "alert_level": "red",
                    "should_notify_family": False,
                    "should_play_audio_alert": True,
                    "rationale_for_user": "Sinais fortes de golpe.",
                    "rationale_for_audit_log": "risk=danger",
                },
                "model_versions": {"classifier": "gemma4:e2b", "stt": "whisper-tiny"},
            }
        ),
        encoding="utf-8",
    )

    markdown = render_call_report(report_path, output_path)

    assert "Final risk: **DANGER**" in markdown
    assert "Alerta vermelho" in markdown
    assert "financial_request" in markdown
    assert output_path.exists()


def test_render_call_report_repairs_display_language(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "call_id": "call_test",
                "language": "pt-BR",
                "duration_seconds": 8.0,
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "transcript_excerpt": "Banco falso.",
                        "signals_detected": ["authority_claim"],
                        "risk_level": "suspicious",
                        "confidence": 0.8,
                        "reasoning": "Sinal suspeito.",
                        "suggested_action_for_user": "Do not provide financial information. Verify with the bank.",
                        "needs_deeper_analysis": False,
                    }
                ],
                "final_state": {
                    "call_id": "call_test",
                    "overall_risk": "suspicious",
                    "top_signals": ["authority_claim"],
                    "alert_level": "yellow",
                    "should_notify_family": False,
                    "should_play_audio_alert": False,
                    "rationale_for_user": "Ha sinais suspeitos.",
                    "rationale_for_audit_log": "risk=suspicious",
                },
                "model_versions": {},
            }
        ),
        encoding="utf-8",
    )

    markdown = render_call_report(report_path)

    assert "Do not provide" not in markdown
    assert "Pause a conversa" in markdown
