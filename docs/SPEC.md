# SPEC — Senti Technical Specification

> Local on-device AI scam-call detection for elderly users.
> Submission to The Gemma 4 Good Hackathon, May 2026.

---

## 1. Problem statement

Phone scams against elderly users cause approximately R$ 2.5 billion in losses per year in Brazil and US$ 11 billion in the United States. Over 75% of victims are over 60 years old. Existing defenses — caller-ID blocking, number blacklists — fail because:

1. Scammers use VoIP services that rotate numbers in minutes.
2. The actual fraud signal is in the **content** of the conversation, not metadata.
3. Elderly users frequently answer before reading the caller display.
4. Many elderly users have low digital literacy and cannot operate complex filtering apps.

The state of the art in industry (Truecaller, Hiya) is cloud-based, raising privacy concerns, requires constant connectivity, and does not analyze conversation content.

## 2. Solution

Senti is an Android app that:

- Listens (via the device microphone, with the call on **speakerphone** — never via the restricted `VOICE_CALL` audio source) to the conversation while it is happening.
- Transcribes the audio locally using `whisper.cpp tiny` (or `small` on capable devices).
- Classifies short segments using **Gemma 4 E2B** running entirely on-device via **MediaPipe LLM Inference** (Google AI Edge / LiteRT).
- Detects ten known social-engineering patterns (see `shared/signals_taxonomy.yaml`).
- Surfaces risk to the user through a high-contrast, large-font UI with visual color states and TTS audio alerts.
- Optionally notifies a pre-configured trusted contact (family member) via FCM — metadata only, never audio.

The user activates Senti by pressing one large green button when a suspicious call begins.

## 3. Why on-device, why Gemma 4

| Constraint | Reason |
|---|---|
| Audio cannot leave the device | Calls of elderly users frequently contain financial information; legal status of recording varies by jurisdiction; vulnerable populations cannot be exposed to cloud breach risk |
| Must work offline | Rural and low-income users may have unreliable connectivity |
| Must be explainable | Safety-critical domain (avoiding wrong "this is a scam" verdicts on a real bank call) demands transparent reasoning |
| Must be multilingual | Brazil has multilingual populations; global deployment requires PT/EN/ES at minimum |
| Must run on consumer Android | Target demographic uses 4-6GB RAM phones, not flagship devices |

Gemma 4 E2B fits these constraints:
- ~2GB RAM quantized (Q4_K_M)
- Native multilingual (PT, EN, ES, FR, IT — all covered)
- Function calling for structured output
- Apache 2.0 licensed (compatible with hackathon CC-BY 4.0 requirement)
- MediaPipe LLM Inference integration (Google AI Edge — official deployment path)

## 4. Architecture

### 4.1 High-level

```
┌─────────────────────────────────────────────────────────────┐
│  ANDROID PHONE (4GB+ RAM)                                    │
│                                                              │
│  [Microphone] (speakerphone) → AudioCaptureService           │
│       │ (foreground service, RECORD_AUDIO permission)        │
│       ▼                                                       │
│  [AudioBuffer] rolling 30s, 16kHz mono PCM                   │
│       │                                                       │
│       ▼ chunked 5s with 1s overlap                            │
│  [whisper.cpp tiny] (or small, depending on RAM headroom)    │
│       │                                                       │
│       ▼ transcript text                                       │
│  [Transcript buffer] last 3 segments                          │
│       │                                                       │
│       ▼ every 5s                                              │
│  [Gemma 4 E2B] via MediaPipe LLM Inference                   │
│       │   function call: classify_call_segment               │
│       ▼ CallSegmentClassification                             │
│  [Decision Engine] applies risk_rules                         │
│       │ if confidence < 0.7 and risk != safe:                │
│       ▼                                                       │
│  [E4B fallback router] swap E2B↔E4B in memory                │
│       │ reclassify, then swap back                            │
│       ▼ updated CallSegmentClassification                     │
│  [CallState] aggregated                                       │
│       │                                                       │
│       ▼                                                       │
│  [UI: Jetpack Compose]                                        │
│       - State color (green/yellow/red)                       │
│       - Detected signals shown in plain language             │
│       - Suggested action                                      │
│       │                                                       │
│       ▼ if alert_level = red                                  │
│  [TTS audio alert] "Possível golpe, desligue"                │
│       │                                                       │
│       ▼ (first time red in this call only)                    │
│  [FCM notification] to family contact — metadata only        │
│       │                                                       │
│       ▼                                                       │
│  [Room database] audit log: session, segments, alerts        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Components

| Component | Tech | Role |
|---|---|---|
| AudioCaptureService | Android `MediaRecorder`, foreground service | Microphone capture, 16kHz mono |
| AudioBuffer | Kotlin in-memory ring buffer | 30s rolling window |
| WhisperBridge | `whisper.cpp` via NDK, JNI | Speech-to-text |
| ScamClassifier | MediaPipe LLM Inference API | Gemma 4 E2B inference + function calling |
| ModelManager | Custom Kotlin | E2B/E4B loading, unloading, memory swap |
| DecisionEngine | Pure Kotlin | Applies risk rules from `signals_taxonomy.yaml` |
| UI | Jetpack Compose, accessibility-first | Listening / Alert / History screens |
| AuditDb | Room (SQLite) | Persistent log of sessions and decisions |
| NotificationGateway | FCM | Push to family contact (metadata only) |

### 4.3 Parallel reference (Track A)

The desktop track (`track_a_desktop/`) reimplements the same logic in Python with Ollama, primarily as:
- Algorithmic reference (canonical decision engine)
- Evaluation harness (run on dataset, compute F1)
- Fallback demo if Android implementation has issues at video time

## 5. Data flow per segment

```
t=0:  user presses ESCUTAR button
t=0+: AudioCaptureService starts, foreground notification shown
t=5s: first audio chunk ready
t=5s+δ: whisper.cpp returns transcript ("Senhora Maria? Aqui é da Caixa...")
t=5s+δ+ε: Gemma 4 E2B classifies segment
           → signals_detected=[authority_claim]
           → risk_level=suspicious
           → confidence=0.78
t=5s+δ+ε+ζ: DecisionEngine updates CallState
              → alert_level=yellow
              → UI turns yellow, shows "Pode ser falso atendente do banco"
...
t=20s: cumulative classifications show [authority_claim, urgency_pressure, personal_data_request]
        → risk_level=danger (critical signal present)
        → alert_level=red
        → TTS plays "Possível golpe. Desligue."
        → UI turns red, large STOP button
        → FCM push sent to family contact: "Mãe está em chamada de risco alto (87%)"
```

Total expected latency for first red alert in a typical scam call: ~20-30 seconds from start of the call. Acceptable for typical scams that take 60-180 seconds.

## 6. Performance budget (per segment)

Target on a Snapdragon 6 Gen 1 / Helio G99 / equivalent (mid-range Android, 4-6GB RAM):

| Stage | Budget | Stretch |
|---|---|---|
| 5s audio capture | 5.0s (real-time) | — |
| whisper.cpp tiny PT | 1.5s | 0.8s with INT8 |
| Gemma 4 E2B (Q4_K_M) inference + function call | 2.0s | 1.2s |
| Decision engine + UI update | 0.05s | — |
| **Total wall-clock from user press to first verdict** | **~7s** | **~5s** |

E4B fallback cost (rare): 3-5s pause for model swap, plus 4-6s inference. Total ~10s. Acceptable for the ~10% of segments where E2B is uncertain.

## 7. Out of scope for MVP

- Cloud-based contact management
- User accounts, login, multi-device sync
- Custom voice training per user
- iOS support
- Tablet UI
- More than 2 languages (PT-BR, EN-US)
- Real call recording (we use mic + speakerphone only)
- Real-time deepfake voice detection (separate research problem)
- Carrier-level integration

## 8. Future work (not in submission)

- Federated learning of new scam patterns across users (opt-in, differentially private)
- Integration with `Disque 100` (Brazil) / `1-800-FRAUD` (US) for direct reporting
- Local fine-tuning per region for dialect adaptation
- Hardware-accelerated inference on Qualcomm Hexagon / Apple Neural Engine
- Support for non-mic input (e.g. recording uploaded after call)

## 9. Success criteria (for the hackathon)

1. **Android app compiles and installs** on a 4GB+ RAM Android 14 phone.
2. **End-to-end demo** in a video (90 seconds) showing:
   - User presses button on phone
   - Synthetic scam audio plays via another speaker (simulating call)
   - Phone transcribes and classifies in real-time
   - UI escalates from green to yellow to red
   - TTS plays alert
   - Family notification simulated
3. **Pipeline evaluation** (desktop MVP) shows:
   - F1 ≥ 0.85 on `risk_level=danger`
   - F1 ≥ 0.70 on `risk_level=suspicious`
   - False positive rate < 10% on legitimate calls
4. **Write-up** clearly explains:
   - Use of Gemma 4's multimodal/function-calling capabilities
   - On-device deployment via MediaPipe LLM Inference (LiteRT)
   - E2B → E4B routing protocol with memory swap (Cactus alignment)
   - Ethics, governance, declared limitations

## 10. Submission deliverables (Kaggle requirements)

1. Public GitHub repository (Apache 2.0 + CC-BY 4.0)
2. Working demo (video + downloadable APK + desktop pipeline)
3. Technical write-up (Kaggle notebook or markdown)
4. Short demo video (3-5 minutes, narrated)
5. Reproducibility instructions (this SPEC + per-track READMEs)
