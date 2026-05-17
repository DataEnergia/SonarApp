# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Deterministic decision engine for Track A."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import structlog

from shared.schemas.classification import (
    SIGNAL_SEVERITY,
    AlertLevel,
    CallSegmentClassification,
    CallState,
    RiskLevel,
    ScamSignal,
    SignalSeverity,
)

LOGGER = structlog.get_logger(__name__)
WINDOW_SIZE = 3
RISK_ORDER = {
    RiskLevel.SAFE: 0,
    RiskLevel.SUSPICIOUS: 1,
    RiskLevel.DANGER: 2,
}


class DecisionEngineError(RuntimeError):
    """Raised when decision engine lifecycle is invalid."""


class DecisionEngine:
    """Pure-Python deterministic engine. Same inputs produce same outputs."""

    def __init__(self, taxonomy_path: Path | None = None) -> None:
        self.taxonomy_path = taxonomy_path
        self._window: list[CallSegmentClassification] = []
        self._call_id: str | None = None
        self._highest_risk_seen: RiskLevel = RiskLevel.SAFE
        self._red_alert_fired = False
        self._last_state: CallState | None = None

    def begin_call(self, call_id: str) -> None:
        self._window = []
        self._call_id = call_id
        self._highest_risk_seen = RiskLevel.SAFE
        self._red_alert_fired = False
        self._last_state = None
        LOGGER.info("decision_call_started", call_id=call_id)

    def update(self, classification: CallSegmentClassification) -> CallState:
        """Add a classification, recompute rolling risk, and apply monotonic escalation."""

        if self._call_id is None:
            raise DecisionEngineError("begin_call must be called before update")
        self._window.append(classification)
        self._window = self._window[-WINDOW_SIZE:]

        aggregate_signals = self._aggregate_signals()
        computed_risk = self._compute_risk_level(aggregate_signals)
        current_risk = self._max_risk(self._highest_risk_seen, computed_risk)
        self._highest_risk_seen = current_risk

        should_notify_family = self._should_notify_family(current_risk)
        should_play_audio_alert = current_risk == RiskLevel.DANGER
        alert_level = _alert_for_risk(current_risk)
        state = CallState(
            call_id=self._call_id,
            overall_risk=current_risk,
            top_signals=self._top_signals(),
            alert_level=alert_level,
            should_notify_family=should_notify_family,
            should_play_audio_alert=should_play_audio_alert,
            rationale_for_user=_user_rationale(current_risk, aggregate_signals),
            rationale_for_audit_log=_audit_rationale(current_risk, aggregate_signals, self._window),
        )
        self._last_state = state
        LOGGER.info(
            "decision_state_updated",
            call_id=self._call_id,
            risk_level=current_risk.value,
            alert_level=alert_level.value,
            signals=[signal.value for signal in aggregate_signals],
        )
        return state

    def end_call(self) -> CallState:
        if self._last_state is None:
            if self._call_id is None:
                raise DecisionEngineError("No active call")
            return CallState(
                call_id=self._call_id,
                overall_risk=RiskLevel.SAFE,
                top_signals=[],
                alert_level=AlertLevel.NONE,
                should_notify_family=False,
                should_play_audio_alert=False,
                rationale_for_user="Nenhum sinal de golpe foi detectado nesta chamada.",
                rationale_for_audit_log="Call ended without classified segments.",
            )
        return self._last_state

    def _compute_risk_level(self, signals: set[ScamSignal]) -> RiskLevel:
        critical_count = _count_by_severity(signals, SignalSeverity.CRITICAL)
        high_count = _count_by_severity(signals, SignalSeverity.HIGH)
        medium_count = _count_by_severity(signals, SignalSeverity.MEDIUM)
        if critical_count > 0:
            return RiskLevel.DANGER
        if high_count >= 2:
            return RiskLevel.DANGER
        if high_count == 1:
            return RiskLevel.SUSPICIOUS
        if medium_count >= 2:
            return RiskLevel.SUSPICIOUS
        return RiskLevel.SAFE

    def _aggregate_signals(self) -> set[ScamSignal]:
        signals: set[ScamSignal] = set()
        for classification in self._window:
            signals.update(classification.signals_detected)
        return signals

    def _should_notify_family(self, current_risk: RiskLevel) -> bool:
        if current_risk != RiskLevel.DANGER or self._red_alert_fired:
            return False
        self._red_alert_fired = True
        return True

    def _top_signals(self) -> list[ScamSignal]:
        counts: Counter[ScamSignal] = Counter()
        for classification in self._window:
            counts.update(classification.signals_detected)
        return [signal for signal, _ in counts.most_common(5)]

    def _max_risk(self, *risk_levels: RiskLevel) -> RiskLevel:
        return max(risk_levels, key=lambda risk: RISK_ORDER[risk])


def _count_by_severity(signals: set[ScamSignal], severity: SignalSeverity) -> int:
    return sum(1 for signal in signals if SIGNAL_SEVERITY[signal] == severity)


def _alert_for_risk(risk_level: RiskLevel) -> AlertLevel:
    if risk_level == RiskLevel.DANGER:
        return AlertLevel.RED
    if risk_level == RiskLevel.SUSPICIOUS:
        return AlertLevel.YELLOW
    return AlertLevel.NONE


def _user_rationale(risk_level: RiskLevel, signals: set[ScamSignal]) -> str:
    if risk_level == RiskLevel.DANGER:
        return "Sinais fortes de possível golpe foram detectados. Pause a conversa e confirme por outro canal."
    if risk_level == RiskLevel.SUSPICIOUS:
        return "Há sinais suspeitos. Não compartilhe dados e confirme a identidade do chamador."
    return "Nenhum sinal forte de golpe foi detectado até agora."


def _audit_rationale(
    risk_level: RiskLevel,
    signals: set[ScamSignal],
    window: list[CallSegmentClassification],
) -> str:
    signal_values = sorted(signal.value for signal in signals)
    segment_ids = [classification.segment_id for classification in window]
    return f"risk={risk_level.value}; signals={signal_values}; rolling_window_segments={segment_ids}"
