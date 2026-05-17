# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from shared.schemas.classification import CallSegmentClassification, RiskLevel, ScamSignal

from decision_engine import DecisionEngine


def test_authority_and_urgency_is_suspicious() -> None:
    engine = DecisionEngine()
    engine.begin_call("call_test")
    state = engine.update(_classification([ScamSignal.AUTHORITY_CLAIM, ScamSignal.URGENCY_PRESSURE]))
    assert state.overall_risk == RiskLevel.SUSPICIOUS


def test_critical_signal_is_danger_and_notifies_once() -> None:
    engine = DecisionEngine()
    engine.begin_call("call_test")
    first = engine.update(_classification([ScamSignal.PERSONAL_DATA_REQUEST], segment_id="seg_1"))
    second = engine.update(_classification([ScamSignal.PERSONAL_DATA_REQUEST], segment_id="seg_2"))
    assert first.overall_risk == RiskLevel.DANGER
    assert first.should_notify_family is True
    assert second.should_notify_family is False


def test_monotonic_escalation_keeps_danger() -> None:
    engine = DecisionEngine()
    engine.begin_call("call_test")
    engine.update(_classification([ScamSignal.FINANCIAL_REQUEST], segment_id="seg_1"))
    state = engine.update(_classification([], risk_level=RiskLevel.SAFE, segment_id="seg_2"))
    assert state.overall_risk == RiskLevel.DANGER


def test_single_medium_signal_stays_safe_even_if_classifier_says_suspicious() -> None:
    engine = DecisionEngine()
    engine.begin_call("call_test")
    state = engine.update(
        _classification(
            [ScamSignal.EMOTIONAL_MANIPULATION],
            risk_level=RiskLevel.SUSPICIOUS,
            segment_id="seg_1",
        )
    )
    assert state.overall_risk == RiskLevel.SAFE


def _classification(
    signals: list[ScamSignal],
    risk_level: RiskLevel | None = None,
    segment_id: str = "seg_1",
) -> CallSegmentClassification:
    inferred_risk = risk_level or (RiskLevel.DANGER if ScamSignal.PERSONAL_DATA_REQUEST in signals else RiskLevel.SUSPICIOUS)
    return CallSegmentClassification(
        segment_id=segment_id,
        transcript_excerpt="teste",
        signals_detected=signals,
        risk_level=inferred_risk,
        confidence=0.9,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por outro canal.",
        needs_deeper_analysis=False,
    )
