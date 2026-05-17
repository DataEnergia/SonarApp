# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""
Canonical schema for the Sonar scam-call classifier.

This module is the source of truth for data structures used across
Track A (desktop Python pipeline), Track D (evaluation), and the
Kotlin equivalents in Track B.

DO NOT MODIFY these classes without updating CONTRACTS.md and
synchronizing all tracks.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScamSignal(str, Enum):
    """Ten known social-engineering patterns in elder-targeted scams."""

    URGENCY_PRESSURE = "urgency_pressure"
    AUTHORITY_CLAIM = "authority_claim"
    ISOLATION_REQUEST = "isolation_request"
    FINANCIAL_REQUEST = "financial_request"
    PERSONAL_DATA_REQUEST = "personal_data_request"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    FAMILY_EMERGENCY_CLAIM = "family_emergency_claim"
    UNUSUAL_PAYMENT_METHOD = "unusual_payment_method"
    REMOTE_ACCESS_REQUEST = "remote_access_request"
    SECRET_KEEPING_REQUEST = "secret_keeping_request"


class SignalSeverity(str, Enum):
    """Severity tier of each signal — used by decision engine."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SIGNAL_SEVERITY: dict[ScamSignal, SignalSeverity] = {
    ScamSignal.URGENCY_PRESSURE: SignalSeverity.MEDIUM,
    ScamSignal.AUTHORITY_CLAIM: SignalSeverity.HIGH,
    ScamSignal.ISOLATION_REQUEST: SignalSeverity.HIGH,
    ScamSignal.FINANCIAL_REQUEST: SignalSeverity.CRITICAL,
    ScamSignal.PERSONAL_DATA_REQUEST: SignalSeverity.CRITICAL,
    ScamSignal.EMOTIONAL_MANIPULATION: SignalSeverity.MEDIUM,
    ScamSignal.FAMILY_EMERGENCY_CLAIM: SignalSeverity.CRITICAL,
    ScamSignal.UNUSUAL_PAYMENT_METHOD: SignalSeverity.CRITICAL,
    ScamSignal.REMOTE_ACCESS_REQUEST: SignalSeverity.CRITICAL,
    ScamSignal.SECRET_KEEPING_REQUEST: SignalSeverity.HIGH,
}


class RiskLevel(str, Enum):
    """Aggregate risk state for a call segment or session."""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGER = "danger"


class AlertLevel(str, Enum):
    """UI alert level shown to the user."""

    NONE = "none"
    YELLOW = "yellow"
    RED = "red"


class Language(str, Enum):
    """Supported call languages in the MVP."""

    PT_BR = "pt-BR"
    EN_US = "en-US"
    ES_419 = "es-419"


class CallSegmentInput(BaseModel):
    """Input to the classifier for a single ~5-second segment."""

    segment_id: str = Field(..., description="Unique within a call session")
    transcript: str = Field(..., description="Text transcribed by STT for this window")
    history_summary: Optional[str] = Field(
        None, description="Short summary of prior segments in the same call"
    )
    language: Language = Field(default=Language.PT_BR)


class CallSegmentClassification(BaseModel):
    """Output of the classifier for a single segment.

    This is the schema the LLM emits via function calling.
    """

    segment_id: str
    transcript_excerpt: str = Field(..., max_length=200)
    signals_detected: list[ScamSignal]
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=300, description="In pt-BR simple language")
    suggested_action_for_user: str = Field(..., max_length=150)
    needs_deeper_analysis: bool = Field(
        default=False, description="If True, triggers E2B → E4B fallback routing"
    )


class CallState(BaseModel):
    """Aggregate state across a call, output of the decision engine."""

    call_id: str
    overall_risk: RiskLevel
    top_signals: list[ScamSignal] = Field(default_factory=list, max_length=5)
    alert_level: AlertLevel
    should_notify_family: bool = False
    should_play_audio_alert: bool = False
    rationale_for_user: str = Field(..., max_length=200)
    rationale_for_audit_log: str = Field(..., max_length=1000)


class CallReport(BaseModel):
    """Full report of a single call session — Track A output."""

    call_id: str
    language: Language
    duration_seconds: float
    segments: list[CallSegmentClassification]
    final_state: CallState
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'classifier': 'gemma4-e2b-q4', 'stt': 'whisper-tiny'}",
    )
