# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

"""Persistent trusted-contact storage for Sonar emergency alerts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def load_contacts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_contacts(path: Path, contacts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contacts, indent=2, ensure_ascii=False), encoding="utf-8")


def add_contact(path: Path, name: str, phone: str, email: str) -> list[dict[str, Any]]:
    contacts = load_contacts(path)
    contacts.append({"id": uuid.uuid4().hex[:12], "name": name.strip(), "phone": phone.strip(), "email": email.strip()})
    save_contacts(path, contacts)
    return contacts


def remove_contact(path: Path, contact_id: str) -> list[dict[str, Any]]:
    contacts = [c for c in load_contacts(path) if c.get("id") != contact_id]
    save_contacts(path, contacts)
    return contacts
