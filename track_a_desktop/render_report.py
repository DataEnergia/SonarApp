# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Render a Sonar CallReport JSON into a human-readable Markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.schemas.classification import AlertLevel, CallReport, RiskLevel  # noqa: E402


def render_call_report(report_path: Path, output_path: Path | None = None) -> str:
    """Load a CallReport JSON and render a concise Markdown decision report."""

    report = CallReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    markdown = _render_markdown(report, report_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    return markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Sonar CallReport JSON as Markdown")
    parser.add_argument("--report", required=True, type=Path, help="Input CallReport JSON path")
    parser.add_argument("--output", default=None, type=Path, help="Optional Markdown output path")
    args = parser.parse_args(argv)

    markdown = render_call_report(args.report, args.output)
    if args.output is None:
        print(markdown)
    return 0


def _render_markdown(report: CallReport, report_path: Path) -> str:
    final = report.final_state
    lines = [
        "# Sonar Call Report",
        "",
        "## Decision",
        "",
        f"- Source JSON: `{report_path}`",
        f"- Call ID: `{report.call_id}`",
        f"- Language: `{report.language.value}`",
        f"- Duration: `{report.duration_seconds:.3f}s`",
        f"- Final risk: **{final.overall_risk.value.upper()}**",
        f"- Alert level: **{final.alert_level.value.upper()}**",
        f"- Play audio alert: `{str(final.should_play_audio_alert).lower()}`",
        f"- Notify family: `{str(final.should_notify_family).lower()}`",
        "",
        "## User-Facing Guidance",
        "",
        _alert_sentence(final.overall_risk, final.alert_level),
        "",
        f"> {final.rationale_for_user}",
        "",
        "## Top Signals",
        "",
    ]
    if final.top_signals:
        lines.extend(f"- `{signal.value}`" for signal in final.top_signals)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Segment Evidence",
            "",
            "| # | Segment ID | Risk | Confidence | Signals | Transcript excerpt | Suggested action |",
            "|---:|---|---|---:|---|---|---|",
        ]
    )
    for index, segment in enumerate(report.segments, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{segment.segment_id}`",
                    f"**{segment.risk_level.value}**",
                    f"{segment.confidence:.2f}",
                    _signals_cell([signal.value for signal in segment.signals_detected]),
                    _cell(segment.transcript_excerpt),
                    _cell(_display_action(segment.suggested_action_for_user, report.language.value, segment.risk_level)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Audit Details",
            "",
            f"- Audit rationale: `{final.rationale_for_audit_log}`",
            f"- Model versions: `{_format_model_versions(report.model_versions)}`",
            "",
            "## Interpretation",
            "",
            _interpretation(report),
            "",
        ]
    )
    return "\n".join(lines)


def _alert_sentence(risk: RiskLevel, alert: AlertLevel) -> str:
    if risk == RiskLevel.DANGER or alert == AlertLevel.RED:
        return "**Alerta vermelho:** pause a conversa, nao informe dados, nao faca pagamentos e confirme por um canal oficial."
    if risk == RiskLevel.SUSPICIOUS or alert == AlertLevel.YELLOW:
        return "**Alerta amarelo:** ha sinais suspeitos; reduza a pressao e confirme a identidade do chamador."
    return "**Sem alerta:** nenhum sinal forte foi agregado, mas mantenha cautela com dados sensiveis."


def _interpretation(report: CallReport) -> str:
    if report.final_state.overall_risk == RiskLevel.DANGER:
        return "A decisao final chegou a `danger` porque ao menos um sinal critico ou combinacao forte foi agregado pela Decision Engine."
    if report.final_state.overall_risk == RiskLevel.SUSPICIOUS:
        return "A decisao final ficou em `suspicious`: ha sinais relevantes, mas nenhum gatilho critico suficiente foi agregado."
    return "A decisao final ficou em `safe`: os segmentos nao sustentaram sinais suficientes para alerta."


def _signals_cell(signals: list[str]) -> str:
    if not signals:
        return "none"
    return ", ".join(f"`{signal}`" for signal in signals)


def _display_action(action: str, language: str, risk: RiskLevel) -> str:
    """Avoid showing obvious language drift in user-facing demo reports."""

    if language == "pt-BR" and _looks_english(action):
        if risk == RiskLevel.DANGER:
            return "Pare a conversa, nao informe dados e confirme por um canal oficial."
        return "Pause a conversa e confirme por um canal oficial antes de agir."
    if language == "en-US" and _looks_portuguese(action):
        if risk == RiskLevel.DANGER:
            return "Stop the call, do not share data, and verify through an official channel."
        return "Pause the call and verify through an official channel before acting."
    return action


def _looks_english(text: str) -> bool:
    lowered = text.lower()
    markers = ["do not", "provide", "verify", "bank", "financial information", "pause the", "official channel"]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _looks_portuguese(text: str) -> bool:
    lowered = text.lower()
    markers = ["nao", "não", "confirme", "canal oficial", "dados", "conversa"]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _format_model_versions(model_versions: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(model_versions.items()))


if __name__ == "__main__":
    raise SystemExit(main())
