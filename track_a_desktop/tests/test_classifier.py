# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from shared.schemas.classification import CallSegmentClassification, CallSegmentInput, ScamSignal

from classifier import (
    GemmaClassifier,
    _enforce_language_fields,
    _normalize_arguments,
    _suppress_unsupported_signals,
    _has_personal_data_request_pattern,
    _has_personal_data_terms,
)


def test_normalize_signal_objects_from_ollama() -> None:
    arguments = {
        "segment_id": "seg_1",
        "transcript_excerpt": "Banco pediu codigo SMS.",
        "signals_detected": [{"name": "authority_claim"}, {"type": "personal_data_request"}],
        "risk_level": "danger",
        "confidence": 0.95,
        "reasoning": "Pedido de codigo indica possivel golpe.",
        "suggested_action_for_user": "Pause e confirme pelo canal oficial.",
        "needs_deeper_analysis": False,
    }

    normalized = _normalize_arguments(arguments)
    classification = CallSegmentClassification.model_validate(normalized)

    assert classification.signals_detected == [
        ScamSignal.AUTHORITY_CLAIM,
        ScamSignal.PERSONAL_DATA_REQUEST,
    ]


def test_normalize_truncates_bounded_text_fields() -> None:
    arguments = {
        "segment_id": "seg_1",
        "transcript_excerpt": "x" * 260,
        "signals_detected": [],
        "risk_level": "safe",
        "confidence": 0.9,
        "reasoning": "r" * 330,
        "suggested_action_for_user": "a" * 180,
        "needs_deeper_analysis": False,
    }

    normalized = _normalize_arguments(arguments)
    classification = CallSegmentClassification.model_validate(normalized)

    assert len(classification.transcript_excerpt) == 200
    assert len(classification.reasoning) == 300
    assert len(classification.suggested_action_for_user) == 150


def test_normalize_maps_transcript_to_excerpt() -> None:
    arguments = {
        "segment_id": "seg_1",
        "transcript": "fallback transcript",
        "signals_detected": [],
        "risk_level": "safe",
        "confidence": 0.8,
        "reasoning": "Sem sinais claros.",
        "suggested_action_for_user": "Continue com cautela.",
        "needs_deeper_analysis": False,
    }

    normalized = _normalize_arguments(arguments)
    classification = CallSegmentClassification.model_validate(normalized)

    assert classification.transcript_excerpt == "fallback transcript"


def test_parse_json_fallback_response() -> None:
    classifier = GemmaClassifier(model_name="test", prompt_path=None)
    data = {
        "message": {
            "content": '{"segment_id":"seg_1","transcript_excerpt":"ok","signals_detected":[],"risk_level":"safe","confidence":0.8,"reasoning":"Sem sinais claros.","suggested_action_for_user":"Continue com cautela.","needs_deeper_analysis":false}'
        }
    }

    classification = classifier._parse_response(data, expect_tool_call=False)

    assert classification.segment_id == "seg_1"
    assert classification.risk_level == "safe"


def test_enforce_language_repairs_pt_br_reasoning() -> None:
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="Banco pediu codigo.",
        signals_detected=[ScamSignal.AUTHORITY_CLAIM, ScamSignal.PERSONAL_DATA_REQUEST],
        risk_level="danger",
        confidence=0.9,
        reasoning="The caller claims to be from a bank and the request suggests risk.",
        suggested_action_for_user="Continue the conversation cautiously.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(segment_id="seg_1", transcript="Banco pediu codigo.", language="pt-BR")

    repaired = _enforce_language_fields(classification, segment_input)

    assert "O trecho sugere" in repaired.reasoning
    assert "conversation" not in repaired.suggested_action_for_user
    assert "canal oficial" in repaired.suggested_action_for_user


def test_enforce_language_repairs_pt_br_action_only() -> None:
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="Banco pediu codigo.",
        signals_detected=[ScamSignal.AUTHORITY_CLAIM],
        risk_level="suspicious",
        confidence=0.9,
        reasoning="O trecho sugere possível risco.",
        suggested_action_for_user="Do not provide any personal or financial information. Verify the claim directly with the bank.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(segment_id="seg_1", transcript="Banco pediu codigo.", language="pt-BR")

    repaired = _enforce_language_fields(classification, segment_input)

    assert "Do not" not in repaired.suggested_action_for_user
    assert "canal oficial" in repaired.suggested_action_for_user


def test_suppress_financial_request_without_financial_terms() -> None:
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="Combinado. Te ligo antes de sair de casa.",
        signals_detected=[ScamSignal.FINANCIAL_REQUEST, ScamSignal.EMOTIONAL_MANIPULATION],
        risk_level="danger",
        confidence=0.9,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por canal oficial.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(segment_id="seg_1", transcript="Combinado. Te ligo antes de sair de casa.", language="pt-BR")

    repaired = _suppress_unsupported_signals(classification, segment_input)

    assert ScamSignal.FINANCIAL_REQUEST not in repaired.signals_detected
    assert ScamSignal.EMOTIONAL_MANIPULATION in repaired.signals_detected


def test_adds_personal_data_request_for_confirmar_dados() -> None:
    """'confirmar dados da conta' missed by LLM should be recovered by lexical check."""
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="preciso confirmar rapidamente alguns dados de sua conta e o collo que vai",
        signals_detected=[ScamSignal.URGENCY_PRESSURE],
        risk_level="suspicious",
        confidence=0.9,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por canal oficial.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(
        segment_id="seg_1",
        transcript="debitado, preciso confirmar rapidamente alguns dados de sua conta e o collo que vai",
        language="pt-BR",
    )

    repaired = _suppress_unsupported_signals(classification, segment_input)

    assert ScamSignal.PERSONAL_DATA_REQUEST in repaired.signals_detected
    assert ScamSignal.URGENCY_PRESSURE in repaired.signals_detected


def test_adds_personal_data_request_for_sms_otp_pattern() -> None:
    """'chegar por SMS' without explicit 'código' should recover personal_data_request via OTP pattern."""
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="chegar por SMS, se não validamos a alguns minutos a operação pode ser aprovada.",
        signals_detected=[ScamSignal.URGENCY_PRESSURE],
        risk_level="suspicious",
        confidence=0.85,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por canal oficial.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(
        segment_id="seg_1",
        transcript="chegar por SMS, se não validamos a alguns minutos a operação pode ser aprovada automaticamente.",
        language="pt-BR",
    )

    repaired = _suppress_unsupported_signals(classification, segment_input)

    assert ScamSignal.PERSONAL_DATA_REQUEST in repaired.signals_detected


def test_suppresses_personal_data_request_without_data_terms() -> None:
    """personal_data_request added by LLM with no supporting data vocabulary should be removed."""
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="Boa tarde, tudo bem com a senhora?",
        signals_detected=[ScamSignal.PERSONAL_DATA_REQUEST],
        risk_level="danger",
        confidence=0.7,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por canal oficial.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(
        segment_id="seg_1",
        transcript="Boa tarde, tudo bem com a senhora?",
        language="pt-BR",
    )

    repaired = _suppress_unsupported_signals(classification, segment_input)

    assert ScamSignal.PERSONAL_DATA_REQUEST not in repaired.signals_detected


def test_adds_financial_request_for_stt_pix_variant() -> None:
    classification = CallSegmentClassification(
        segment_id="seg_1",
        transcript_excerpt="Faca agora o Mpix de teste para achar que vou informar.",
        signals_detected=[ScamSignal.AUTHORITY_CLAIM, ScamSignal.URGENCY_PRESSURE],
        risk_level="suspicious",
        confidence=0.85,
        reasoning="Trecho de teste.",
        suggested_action_for_user="Confirme por canal oficial.",
        needs_deeper_analysis=False,
    )
    segment_input = CallSegmentInput(
        segment_id="seg_1",
        transcript="Faca agora o Mpix de teste para achar que vou informar.",
        language="pt-BR",
    )

    repaired = _suppress_unsupported_signals(classification, segment_input)

    assert ScamSignal.FINANCIAL_REQUEST in repaired.signals_detected
