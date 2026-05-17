# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Ollama-backed Gemma classifier for Track A."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import structlog
from pydantic import ValidationError

from shared.schemas.classification import CallSegmentClassification, CallSegmentInput, ScamSignal

LOGGER = structlog.get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = REPO_ROOT / "shared" / "prompts" / "classifier_v1.txt"


class ClassifierError(RuntimeError):
    """Raised after classifier retries are exhausted."""


class ClassifierTimeoutError(ClassifierError):
    """Raised when Ollama is unreachable or times out."""


class GemmaClassifier:
    """LM Studio (OpenAI-compatible) classifier using Gemma 4 E2B by default."""

    def __init__(
        self,
        model_name: str = "google/gemma-4-e2b",
        ollama_url: str = "http://localhost:1234",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        prompt_path: Path | None = None,
        allow_json_fallback: bool = True,
        num_predict: int = 2000,
        few_shot_context: str = "",
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.prompt_path = prompt_path or SYSTEM_PROMPT_PATH
        self.system_prompt = self.prompt_path.read_text(encoding="utf-8")
        self.allow_json_fallback = allow_json_fallback
        self.num_predict = num_predict
        self.few_shot_context = few_shot_context

    def classify_segment(self, segment_input: CallSegmentInput) -> CallSegmentClassification:
        """Classify one segment via Ollama function calling and validate output."""

        feedback: str | None = None
        use_tools = True
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            started_at = time.perf_counter()
            try:
                payload = (
                    self._build_payload(segment_input, feedback, use_tools=True)
                    if use_tools
                    else self._build_generate_payload(segment_input, feedback)
                )
                response = requests.post(
                    f"{self.ollama_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 400 and use_tools and self.allow_json_fallback:
                    use_tools = False
                    feedback = "Tool calling failed. Return only a raw JSON object matching the schema."
                    raise ValueError("LM Studio rejected tool calling; retrying with JSON fallback")
                if not response.ok:
                    last_error = ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
                    if use_tools and self.allow_json_fallback:
                        use_tools = False
                        feedback = "Return only a raw JSON object matching the schema. No markdown, no tool calls."
                    else:
                        feedback = f"HTTP {response.status_code} error. Return a valid JSON response."
                    LOGGER.warning("classifier_retry", segment_id=segment_input.segment_id, attempt=attempt, error=str(last_error))
                    continue
                classification = self._parse_response(response.json(), expect_tool_call=use_tools)
                if classification.segment_id != segment_input.segment_id:
                    raise ValueError(
                        f"segment_id mismatch: expected {segment_input.segment_id}, got {classification.segment_id}"
                    )
                classification = _suppress_unsupported_signals(classification, segment_input)
                classification = _enforce_language_fields(classification, segment_input)
                LOGGER.info(
                    "classifier_segment_ok",
                    segment_id=segment_input.segment_id,
                    risk_level=classification.risk_level.value,
                    elapsed_seconds=round(time.perf_counter() - started_at, 3),
                    attempt=attempt,
                )
                return classification
            except requests.Timeout as exc:
                last_error = exc
                feedback = "The previous Ollama request timed out. Return only the function call."
            except requests.RequestException as exc:
                raise ClassifierTimeoutError(f"Ollama request failed: {exc}") from exc
            except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                if use_tools and self.allow_json_fallback and "did not include tool_calls" in str(exc):
                    use_tools = False
                    feedback = "Tool calling was not emitted. Return only a raw JSON object matching the schema (no markdown)."
                else:
                    feedback = f"Previous output was invalid: {exc}. Return a valid classify_call_segment tool call."

            LOGGER.warning(
                "classifier_retry",
                segment_id=segment_input.segment_id,
                attempt=attempt,
                error=str(last_error),
            )

        raise ClassifierError(f"Classifier failed after retries: {last_error}")

    def _build_payload(
        self,
        segment_input: CallSegmentInput,
        feedback: str | None,
        use_tools: bool = True,
    ) -> dict[str, Any]:
        user_content = segment_input.model_dump_json()
        if feedback:
            user_content = f"{user_content}\n\nValidation feedback: {feedback}"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt(use_tools)},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
            "stream": False,
        }
        if use_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "classify_call_segment",
                        "description": "Classify a phone call segment for scam signals.",
                        "parameters": CallSegmentClassification.model_json_schema(),
                    },
                }
            ]
        else:
            payload["format"] = "json"
        return payload

    def _build_generate_payload(self, segment_input: CallSegmentInput, feedback: str | None) -> dict[str, Any]:
        user_content = (
            f"{_json_fallback_prompt()}\n\n"
            f"Input segment JSON:\n{segment_input.model_dump_json()}"
        )
        if feedback:
            user_content = f"{user_content}\n\nValidation feedback: {feedback}"
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
        }

    def _parse_response(self, data: dict[str, Any], expect_tool_call: bool = True) -> CallSegmentClassification:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else data.get("message", {})
        content = message.get("content", "")

        if not expect_tool_call:
            extracted = _extract_json_from_content(content) if isinstance(content, str) else content
            if extracted is None:
                raise ValueError(f"No JSON found in response content: {str(content)[:200]}")
            arguments = _normalize_arguments(extracted)
            return CallSegmentClassification.model_validate(arguments)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # Gemma thinking mode: JSON may be embedded in content after reasoning block
            extracted = _extract_json_from_content(content) if isinstance(content, str) else None
            if extracted:
                arguments = _normalize_arguments(extracted)
                return CallSegmentClassification.model_validate(arguments)
            raise ValueError(f"LM Studio response did not include tool_calls: {str(content)[:300]}")
        arguments = tool_calls[0]["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        arguments = _normalize_arguments(arguments)
        return CallSegmentClassification.model_validate(arguments)

    def _system_prompt(self, use_tools: bool) -> str:
        # Prepend instruction to suppress Gemma 4 thinking mode — keeps responses compact and fast
        no_thinking = "Do not use thinking or reasoning blocks. Do not output <|channel>thought tags. Respond immediately with the function call only.\n\n"
        base = self.system_prompt if use_tools else _json_fallback_prompt()
        prompt = no_thinking + base
        if self.few_shot_context:
            prompt = prompt + "\n\n" + self.few_shot_context
        return prompt


def _json_fallback_prompt() -> str:
    return """You are Sonar, a local scam-call classifier.

Return only one valid JSON object. Do not use markdown. Do not call tools.

Required keys:
- segment_id: exactly the input segment_id
- transcript_excerpt: max 200 chars
- signals_detected: array of strings only
- risk_level: safe, suspicious, or danger
- confidence: number between 0 and 1
- reasoning: max 300 chars; pt-BR if language=pt-BR, English if language=en-US, Spanish if language=es-419
- suggested_action_for_user: max 150 chars; same language as input
- needs_deeper_analysis: boolean

Allowed signals:
urgency_pressure, authority_claim, isolation_request, financial_request, personal_data_request, emotional_manipulation, family_emergency_claim, unusual_payment_method, remote_access_request, secret_keeping_request.

Risk rules:
danger = any critical signal or two high signals.
suspicious = one high signal or two medium signals.
safe = no clear signals.

Critical: financial_request, personal_data_request, family_emergency_claim, unusual_payment_method, remote_access_request.
High: authority_claim, isolation_request, secret_keeping_request.
Medium: urgency_pressure, emotional_manipulation.

Be conservative and never claim certainty.
"""


_STT_CORRECTIONS_ES: list[tuple[str, str]] = [
    # Whisper substitutes DNI (Argentine national ID) with DNA
    (r"\bDNA\b", "DNI"),
    (r"\bD\.N\.A\.?\b", "D.N.I."),
    # Common phone/acronym confusions
    (r"\bCUILL\b", "CUIL"),
    (r"\bKUIL\b", "CUIL"),
    (r"\bC\.U\.I\.L\.?\b", "CUIL"),
    # "variedación" → "verificación" (phonetic garble)
    (r"\bvariedaci[oó]n\b", "verificación"),
    # "supervivencia" sometimes garbled
    (r"\bsupervivencia\b", "supervivencia"),  # keep as-is; listed for documentation
]


def correct_stt_transcript(text: str, language_value: str) -> str:
    """Apply deterministic STT error corrections before classification.

    Only called for es-419 for now; other languages can be added later.
    """
    if language_value != "es-419":
        return text
    for pattern, replacement in _STT_CORRECTIONS_ES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _extract_json_from_content(content: str) -> dict | None:
    """Extract first valid JSON object from content that may contain thinking tags or prose."""
    # Strip Gemma thinking blocks: <|channel>thought ... and <think> ...
    clean = re.sub(r"<\|channel\|?>thought.*?</?\|channel\|?>thought>?", "", content, flags=re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", clean)
    # Find the outermost JSON object
    start = clean.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(clean[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(clean[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_arguments(arguments: Any) -> Any:
    """Normalize minor Ollama tool-call shape drift before Pydantic validation."""

    if not isinstance(arguments, dict):
        return arguments
    signals = arguments.get("signals_detected")
    if "transcript_excerpt" not in arguments and isinstance(arguments.get("transcript"), str):
        arguments["transcript_excerpt"] = arguments["transcript"]
    if isinstance(signals, list):
        arguments["signals_detected"] = [_normalize_signal(signal) for signal in signals]
    _truncate_string_field(arguments, "transcript_excerpt", 200)
    _truncate_string_field(arguments, "reasoning", 300)
    _truncate_string_field(arguments, "suggested_action_for_user", 150)
    return arguments


def _normalize_signal(signal: Any) -> Any:
    if not isinstance(signal, dict):
        return signal
    for key in ("name", "type", "value"):
        if key in signal:
            return signal[key]
    return signal


def _truncate_string_field(arguments: dict[str, Any], field_name: str, max_length: int) -> None:
    value = arguments.get(field_name)
    if isinstance(value, str) and len(value) > max_length:
        arguments[field_name] = value[: max_length - 1].rstrip() + "…"


def _enforce_language_fields(
    classification: CallSegmentClassification,
    segment_input: CallSegmentInput,
) -> CallSegmentClassification:
    """Repair obvious language drift in bounded user-facing fields."""

    lang = segment_input.language.value
    if lang == "pt-BR" and _looks_english(classification.reasoning):
        updates = {"reasoning": _portuguese_reasoning(classification.signals_detected)}
        if _looks_english(classification.suggested_action_for_user):
            updates["suggested_action_for_user"] = _portuguese_action(classification.signals_detected)
        return classification.model_copy(update=updates)
    if lang == "pt-BR" and _looks_english(classification.suggested_action_for_user):
        return classification.model_copy(update={"suggested_action_for_user": _portuguese_action(classification.signals_detected)})
    if lang == "en-US" and _looks_portuguese(classification.reasoning):
        updates = {"reasoning": _english_reasoning(classification.signals_detected)}
        if _looks_portuguese(classification.suggested_action_for_user):
            updates["suggested_action_for_user"] = _english_action(classification.signals_detected)
        return classification.model_copy(update=updates)
    if lang == "en-US" and _looks_portuguese(classification.suggested_action_for_user):
        return classification.model_copy(update={"suggested_action_for_user": _english_action(classification.signals_detected)})
    if lang == "es-419" and not _looks_spanish(classification.reasoning):
        updates = {"reasoning": _spanish_reasoning(classification.signals_detected)}
        if not _looks_spanish(classification.suggested_action_for_user):
            updates["suggested_action_for_user"] = _spanish_action(classification.signals_detected)
        return classification.model_copy(update=updates)
    return classification


def _suppress_unsupported_signals(
    classification: CallSegmentClassification,
    segment_input: CallSegmentInput,
) -> CallSegmentClassification:
    """Remove high-cost critical signals when transcript lacks minimum lexical support."""

    transcript = f"{segment_input.transcript} {classification.transcript_excerpt}".lower()
    signals = list(classification.signals_detected)
    if not any(_signal_value(signal) == "financial_request" for signal in signals) and _has_financial_request_pattern(transcript):
        signals.append(ScamSignal.FINANCIAL_REQUEST)
    if any(_signal_value(signal) == "financial_request" for signal in signals) and not _has_financial_terms(transcript):
        signals = [signal for signal in signals if _signal_value(signal) != "financial_request"]
    # personal_data_request: recover when OTP/SMS or data-confirmation pattern is present but LLM missed it
    if not any(_signal_value(signal) == "personal_data_request" for signal in signals) and _has_personal_data_request_pattern(transcript):
        signals.append(ScamSignal.PERSONAL_DATA_REQUEST)
    # personal_data_request: suppress when no personal-data vocabulary is present at all
    if any(_signal_value(signal) == "personal_data_request" for signal in signals) and not _has_personal_data_terms(transcript):
        signals = [signal for signal in signals if _signal_value(signal) != "personal_data_request"]
    if signals == list(classification.signals_detected):
        return classification
    return classification.model_copy(update={"signals_detected": signals})


def _has_personal_data_terms(text: str) -> bool:
    """Return True when the transcript contains personal/auth-data vocabulary."""
    terms = [
        # Portuguese
        "sms", "código", "codigo", "senha", "cpf", "rg", "otp",
        "dados bancários", "dados bancarios",
        "informações pessoais", "informacoes pessoais",
        "número do cartão", "numero do cartao",
        "código de verificação", "codigo de verificacao",
        "dados", "informações", "informacoes",
        # English
        "password", "pin", "verification code", "security code", "social security",
        # Spanish — standard
        "contraseña", "clave", "cuil", "curp", "datos personales", "datos bancarios",
        # Spanish — Argentine/Latin American identity documents
        "dni", "cuit", "rut", "documento nacional", "número de documento",
        "número de identidad", "numero de identidad",
        # Generic personal-data phrases in Spanish
        "datos", "tarjeta vinculada", "cuenta vinculada",
        "número de cuenta", "numero de cuenta",
        "número de tarjeta", "numero de tarjeta",
    ]
    return any(term in text for term in terms)


def _has_personal_data_request_pattern(text: str) -> bool:
    """Return True when text shows a request for personal data or OTP code.

    Catches two common STT failure modes: (a) 'código' garbled so only 'sms'
    survives, (b) 'confirmar dados da conta' missed by the LLM under urgency framing.
    """
    sms_otp = "sms" in text and any(
        m in text
        for m in [
            "chegar", "chegou", "validar", "validamos", "código", "codigo",
            "confirmar", "confirme", "enviamos", "enviado", "receber", "recebeu",
            "code", "verify",
        ]
    )
    pt_data_confirm = any(
        v in text
        for v in [
            "confirmar", "confirme", "informe", "fornece", "forneça",
            "me passa", "me diz", "validar", "valide",
        ]
    ) and any(
        n in text
        for n in ["dados", "informações", "informacoes", "conta", "senha", "código", "codigo", "cpf"]
    )
    en_data = (
        "verification code" in text
        or ("sms" in text and "code" in text)
        or (
            any(v in text for v in ["confirm your", "provide your", "give me your", "what is your"])
            and any(n in text for n in ["password", "pin", "account", "social security"])
        )
    )
    es_data = (
        "sms" in text and any(m in text for m in ["código", "codigo", "clave", "validar", "confirmar"])
    ) or (
        any(v in text for v in [
            "confirme", "proporcione", "dígame", "digame",
            "necesito", "necesitamos", "voy a pedir", "le voy a pedir",
            "pedir", "pedimos", "me da", "me diga", "me pase", "me puede pasar",
            "validar", "validamos", "confirmar", "verifique", "verifiquemos",
        ])
        and any(n in text for n in [
            "contraseña", "clave", "cuil", "cuit", "curp", "datos", "cuenta",
            "dni", "documento", "tarjeta", "identidad", "número",
        ])
    )
    return sms_otp or pt_data_confirm or en_data or es_data


def _has_financial_terms(text: str) -> bool:
    terms = [
        "pix", "mpix",
        "dinheiro", "reais", "real",
        "pagar", "pagamento", "transferencia", "transferência",
        "deposito", "depósito", "cartao", "cartão",
        "gift card", "gift cards",
        "wire", "payment", "pay", "money", "card", "crypto",
        # Spanish — standard
        "dinero", "transferencia", "depositar", "depósito",
        "tarjeta", "efectivo", "pesos", "dolares", "dólares",
        "bitcoin", "criptomoneda",
        # Spanish — colloquial Latin America
        "plata", "lana", "guita", "billete", "pasta", "fondos",
    ]
    return any(term in text for term in terms)


def _has_financial_request_pattern(text: str) -> bool:
    request_markers = [
        # Portuguese
        "faca", "faça", "faz", "fazer",
        "envie", "manda", "mande", "pague", "pagar", "transfira",
        # English
        "send", "transfer", "buy", "pay",
        # Spanish — imperative / request forms (formal and informal)
        "haga", "envíe", "envia", "envía", "mande", "manda",
        "deposite", "transfiera", "compre", "dame", "deme",
        "préstame", "prestame", "preste", "mándame", "mandame",
        "necesito", "necesitas", "necesitamos",
    ]
    return _has_financial_terms(text) and any(marker in text for marker in request_markers)


def _signal_value(signal: Any) -> str:
    return signal.value if hasattr(signal, "value") else str(signal)


def _looks_english(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "caller",
        "call",
        "bank",
        "request",
        "urgency",
        "suggests",
        "may indicate",
        "risk",
        "transcript",
        "scam",
        "signals",
        "conversation",
        "cautious",
        "do not",
        "provide",
        "verify",
        "financial information",
        "immediate",
        "demands",
    ]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _looks_portuguese(text: str) -> bool:
    lowered = text.lower()
    markers = ["chamada", "banco", "pode", "sugere", "possível", "risco", "dados"]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _portuguese_reasoning(signals: list[Any]) -> str:
    if not signals:
        return "O trecho não mostra sinais claros de golpe. Ainda assim, confirme informações sensíveis por canais oficiais."
    signal_text = ", ".join(_signal_value(signal) for signal in signals[:4])
    return f"O trecho sugere possível risco por sinais como {signal_text}. Confirme por um canal oficial antes de agir."


def _english_reasoning(signals: list[Any]) -> str:
    if not signals:
        return "The segment shows no clear scam signals. Still verify sensitive requests through official channels."
    signal_text = ", ".join(_signal_value(signal) for signal in signals[:4])
    return f"The segment may indicate risk due to signals such as {signal_text}. Verify through an official channel first."


def _portuguese_action(signals: list[Any]) -> str:
    if not signals:
        return "Continue com cautela e confirme pedidos sensíveis por canais oficiais."
    return "Pause a conversa e confirme por um canal oficial antes de agir."


def _english_action(signals: list[Any]) -> str:
    if not signals:
        return "Continue carefully and verify sensitive requests through official channels."
    return "Pause the call and verify through an official channel before acting."


def _looks_spanish(text: str) -> bool:
    lowered = text.lower()
    markers = ["llamada", "banco", "puede", "sugiere", "posible", "riesgo", "datos", "señales", "canal"]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _spanish_reasoning(signals: list[Any]) -> str:
    if not signals:
        return "El fragmento no muestra señales claras de estafa. De todos modos, confirme datos sensibles por canales oficiales."
    signal_text = ", ".join(_signal_value(signal) for signal in signals[:4])
    return f"El fragmento puede indicar riesgo por señales como {signal_text}. Verifique por un canal oficial antes de actuar."


def _spanish_action(signals: list[Any]) -> str:
    if not signals:
        return "Continúe con cautela y confirme solicitudes sensibles por canales oficiales."
    return "Pause la llamada y verifique por un canal oficial antes de actuar."
