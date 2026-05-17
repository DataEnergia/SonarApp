# CONTRACTS — Frozen interfaces between tracks

> **This file is the source of truth for data structures and inter-track interfaces. Changes require explicit approval from the human orchestrator (Carlos). Agents MUST NOT modify the contracts unilaterally.**

---

## 1. Canonical schema (Python — Pydantic)

The Python schema in `shared/schemas/classification.py` is the canonical reference. Track B (Android Kotlin) MUST replicate the equivalent data classes with identical field names and types.

### 1.1 ScamSignal enum (10 values)

```
URGENCY_PRESSURE        = "urgency_pressure"
AUTHORITY_CLAIM         = "authority_claim"
ISOLATION_REQUEST       = "isolation_request"
FINANCIAL_REQUEST       = "financial_request"
PERSONAL_DATA_REQUEST   = "personal_data_request"
EMOTIONAL_MANIPULATION  = "emotional_manipulation"
FAMILY_EMERGENCY_CLAIM  = "family_emergency_claim"
UNUSUAL_PAYMENT_METHOD  = "unusual_payment_method"
REMOTE_ACCESS_REQUEST   = "remote_access_request"
SECRET_KEEPING_REQUEST  = "secret_keeping_request"
```

Definitions, severities, and examples in `shared/signals_taxonomy.yaml`.

### 1.2 RiskLevel enum (3 values)

```
SAFE        = "safe"        # no signals, or 1 low-severity signal
SUSPICIOUS  = "suspicious"  # 1 high-severity OR 2+ medium signals
DANGER      = "danger"      # any critical signal OR 2+ high signals
```

### 1.3 CallSegmentInput (input to classifier)

```
{
  "segment_id": "string, unique within a call session",
  "transcript": "string, transcribed text of the current ~5s window",
  "history_summary": "string|null, short summary of prior segments in this call",
  "language": "pt-BR | en-US"
}
```

### 1.4 CallSegmentClassification (output from classifier)

```
{
  "segment_id": "string, must match input",
  "transcript_excerpt": "string, max 200 chars, the 1-2 most relevant sentences",
  "signals_detected": ["array of ScamSignal values"],
  "risk_level": "RiskLevel value",
  "confidence": "float, 0.0 to 1.0",
  "reasoning": "string, max 300 chars, in pt-BR simple language",
  "suggested_action_for_user": "string, max 150 chars",
  "needs_deeper_analysis": "bool, triggers E2B → E4B routing if true"
}
```

### 1.5 CallState (output of decision engine)

```
{
  "call_id": "string",
  "overall_risk": "RiskLevel",
  "top_signals": ["array of ScamSignal, max 5, most prominent first"],
  "alert_level": "none | yellow | red",
  "should_notify_family": "bool",
  "should_play_audio_alert": "bool",
  "rationale_for_user": "string, max 200 chars",
  "rationale_for_audit_log": "string, max 1000 chars"
}
```

---

## 2. Decision Engine rules (deterministic)

These rules are shared between Track A (Python) and Track B (Kotlin). Same inputs MUST produce same outputs.

### 2.1 Signal severity (from `shared/signals_taxonomy.yaml`)

```
CRITICAL: financial_request, personal_data_request,
          family_emergency_claim, unusual_payment_method,
          remote_access_request
HIGH:     authority_claim, isolation_request,
          secret_keeping_request
MEDIUM:   urgency_pressure, emotional_manipulation
```

### 2.2 Risk level computation

Given a list of unique signals detected within the rolling 3-segment window:

```
if any signal has severity CRITICAL:
    risk_level = DANGER
elif count(signals with severity HIGH) >= 2:
    risk_level = DANGER
elif count(signals with severity HIGH) == 1:
    risk_level = SUSPICIOUS
elif count(signals with severity MEDIUM) >= 2:
    risk_level = SUSPICIOUS
else:
    risk_level = SAFE
```

### 2.3 Monotonic escalation

Within a single call session, `risk_level` is **monotonically non-decreasing**. Once a call hits `DANGER`, it stays at `DANGER` until the call ends, even if subsequent segments classify lower. Rationale: false negatives mid-call are common; the user should not be lulled back into safety.

### 2.4 Alert level mapping

```
risk_level = SAFE       → alert_level = none
risk_level = SUSPICIOUS → alert_level = yellow
risk_level = DANGER     → alert_level = red
```

### 2.5 Notification triggers

```
should_play_audio_alert = (alert_level == red)
should_notify_family    = (alert_level == red) AND (user has configured contact)
                          AND (this is the first red event in this call)
```

---

## 3. Routing protocol (E2B → E4B fallback)

### 3.1 Trigger conditions

E4B is invoked instead of E2B's result when **any** of:

- `confidence < 0.7` AND `risk_level != SAFE`
- `needs_deeper_analysis == true`
- Cumulative E2B classifications in current call show inconsistency (oscillation between risk levels)

### 3.2 Memory swap protocol (Android, RAM constraint)

When E4B is invoked:

1. Save current call buffer state to RAM/disk
2. Unload E2B from memory (`ModelManager.unload(E2B)`)
3. Wait ≤ 1s for GC
4. Load E4B (`ModelManager.load(E4B)`)
5. Reclassify the buffer with E4B
6. Unload E4B (`ModelManager.unload(E4B)`)
7. Reload E2B (`ModelManager.load(E2B)`)
8. Resume normal flow

Expected pause: 3-5 seconds. Acceptable for rare triggers.

### 3.3 E4B classification overrides E2B

When E4B classifies a segment, its result REPLACES the E2B result in the audit log. The CallState recomputes based on E4B's verdict for that segment.

---

## 4. Audit log schema (SQLite via Room on Android, sqlite3 on desktop)

```sql
CREATE TABLE call_session (
    call_id TEXT PRIMARY KEY,
    started_at INTEGER NOT NULL,    -- unix epoch ms
    ended_at INTEGER,                -- nullable while in progress
    final_risk TEXT,                 -- RiskLevel string
    language TEXT NOT NULL,
    model_version TEXT NOT NULL,     -- e.g. "gemma4-e2b-q4_k_m"
    user_consent_given BOOLEAN NOT NULL
);

CREATE TABLE call_segment (
    segment_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES call_session(call_id),
    t_start_ms INTEGER NOT NULL,
    t_end_ms INTEGER NOT NULL,
    transcript TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    signals_detected_json TEXT NOT NULL,   -- JSON array of ScamSignal
    reasoning TEXT NOT NULL,
    model_used TEXT NOT NULL                -- "E2B" or "E4B"
);

CREATE TABLE alert_event (
    event_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES call_session(call_id),
    triggered_at INTEGER NOT NULL,
    alert_level TEXT NOT NULL,
    family_notified BOOLEAN NOT NULL,
    user_response TEXT          -- "dismissed" | "kept_call" | "hung_up" | "unknown"
);
```

---

## 5. Inter-process communication (Track A → Track D)

Track D's evaluation harness consumes Track A's output. The contract:

- Track A's `pipeline.py` produces a `CallReport` JSON per audio file
- Schema: `{ call_id, language, segments: [CallSegmentClassification...], final_state: CallState }`
- Track D's `compute_metrics.py` reads these and compares against `track_c_dataset/ground_truth.json`

---

## 6. Versioning

- CONTRACTS.md version: `1.0.0` (set on hackathon submission)
- Breaking changes during the 6-day hackathon require:
  1. Documented rationale in `docs/DECISIONS.md`
  2. All tracks updated in same commit
  3. Human orchestrator approval

---

## 7. What MUST NOT change without explicit approval

- ScamSignal enum values (10 signals)
- RiskLevel enum values (3 levels)
- Severity assignments
- Decision engine rules (sections 2.2, 2.3)
- Routing trigger conditions (section 3.1)
- Audit log table structure

## 8. What CAN evolve freely within tracks

- Implementation details (libraries, internal class structure)
- Prompt wording (Track D's job)
- UI design
- TTS voices, audio alerts
- Internal logging beyond audit schema
