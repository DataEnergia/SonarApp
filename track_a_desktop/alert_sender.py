# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Emergency alert sender for Sonar — emails trusted contacts when a scam is detected."""

from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import structlog

LOGGER = structlog.get_logger(__name__)

_SIGNAL_LABELS: dict[str, str] = {
    "financial_request": "Pedido de transferência / PIX",
    "personal_data_request": "Solicitação de dados pessoais (CPF, senha, código)",
    "authority_claim": "Falsa autoridade (banco, governo, INSS, polícia)",
    "urgency_pressure": "Pressão de urgência",
    "isolation_request": "Pedido para não desligar / não contar",
    "secret_keeping_request": "Pedido de sigilo",
    "family_emergency_claim": "Falsa emergência familiar",
    "unusual_payment_method": "Método de pagamento incomum (gift card, cripto)",
    "remote_access_request": "Pedido de acesso remoto ao celular/computador",
    "emotional_manipulation": "Manipulação emocional",
}

_RISK_HEADER: dict[str, str] = {
    "suspicious": "⚠️ CHAMADA SUSPEITA DETECTADA",
    "danger":     "🚨 GOLPE EM ANDAMENTO — AÇÃO URGENTE",
}


def load_smtp_config(config_path: Path) -> dict[str, Any]:
    """Load SMTP settings from sonar_config.json. Returns empty dict if not configured."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("smtp", {})
    except Exception:
        return {}


def save_smtp_config(config_path: Path, smtp: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["smtp"] = smtp
    config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def send_alert(
    smtp_config: dict[str, Any],
    contacts: list[dict[str, Any]],
    alert: dict[str, Any],
) -> list[str]:
    """Send email alert to all contacts with a valid email address.

    Returns list of addresses the alert was successfully sent to.
    Logs warnings for any failures without raising.
    """
    host = smtp_config.get("host", "").strip()
    port = int(smtp_config.get("port", 587))
    username = smtp_config.get("username", "").strip()
    password = smtp_config.get("password", "").strip()
    from_name = smtp_config.get("from_name", "Sonar — Proteção Contra Golpes")

    if not (host and username and password):
        LOGGER.warning("alert_smtp_not_configured")
        return []

    risk = alert.get("risk", "suspicious")
    signals: list[str] = alert.get("signals", [])
    excerpt: str = alert.get("excerpt", "")
    duration_s: int = int(alert.get("duration_seconds", 0))

    header = _RISK_HEADER.get(risk, "⚠️ CHAMADA SUSPEITA")
    signals_text = "\n".join(
        f"  • {_SIGNAL_LABELS.get(s, s)}" for s in signals
    ) or "  • (sem detalhes)"

    excerpt_block = f'\nTrecho da conversa:\n  "{excerpt}"\n' if excerpt.strip() else ""
    duration_block = f"⏱ Duração da chamada até o alerta: {duration_s}s\n" if duration_s else ""

    subject = f"[Sonar] {header}"
    body = f"""{header}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O Sonar detectou sinais de golpe em uma chamada ativa.
Alguém próximo pode estar sendo vítima agora.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sinais detectados:
{signals_text}
{excerpt_block}
{duration_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 O QUE FAZER AGORA:
  1. Ligue imediatamente para confirmar se a pessoa está bem
  2. Oriente-a a desligar a chamada suspeita
  3. Lembre: nenhum banco, governo ou INSS pede
     transferências, senhas ou códigos por telefone

– Sonar, proteção ativa contra golpes telefônicos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""".strip()

    sent_to: list[str] = []
    for contact in contacts:
        email_addr = contact.get("email", "").strip()
        if not email_addr:
            continue
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{from_name} <{username}>"
            msg["To"] = f"{contact.get('name', '')} <{email_addr}>"
            msg.attach(MIMEText(body, "plain", "utf-8"))
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(username, password)
                server.sendmail(username, email_addr, msg.as_string())
            sent_to.append(email_addr)
            LOGGER.info("alert_sent", to=email_addr, risk=risk)
        except Exception as exc:
            LOGGER.warning("alert_send_failed", to=email_addr, error=str(exc))

    return sent_to
