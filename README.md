# Sonar — Real-Time Scam Call Detection

**Submission for The Gemma 4 Good Hackathon (Kaggle × Google DeepMind, May 2026)**

On-device AI pipeline that detects phone scams in real time using Gemma 4. Fully local. No audio or transcription leaves the device. Multilingual: Portuguese (pt-BR), English (en-US), Latin American Spanish (es-419).

---

## Evaluation Results

Blind test on 55 independent calls — Gemma had no access to ground truth labels at any point.

| Dataset | n | Precision | Recall | F1 | Accuracy |
|---------|---|-----------|--------|----|----------|
| Real human recordings | 30 | 100% | 100% | 100% | 100% |
| TTS synthetic | 25 | 63% | 100% | 77% | 72% |
| **Combined** | **55** | **79%** | **100%** | **88.5%** | **87.3%** |

- **TP=27 · FN=0 · TN=21 · FP=7**
- Zero scam calls missed across all 55 tests
- False positives concentrated in TTS synthetic audio (acoustic artifacts from machine-generated speech, not real calls)

---

## Architecture

```
Incoming call audio (microphone / speakerphone)
        │
        ▼
  faster-whisper (STT)          ← local, CPU/GPU, small model
        │  transcript chunk (every ~30s)
        ▼
  GemmaClassifier               ← Gemma 4 E2B via LM Studio (OpenAI-compatible API)
        │  classify_call_segment() — function calling, structured output
        ▼
  DecisionEngine                ← deterministic risk aggregation
        │  CallState { overall_risk, alert_level, signals }
        ▼
  UI Alert + Email notification  ← browser UI (localhost) + SMTP alert to family contacts
```

All components run locally. LM Studio serves Gemma 4 E2B (Q4_K_M) via `http://localhost:1234/v1`.

---

## Scam Signal Taxonomy

10 signals detected, defined in `shared/signals_taxonomy.yaml`:

| Signal | Severity |
|--------|----------|
| `financial_request` | Critical |
| `personal_data_request` | Critical |
| `family_emergency_claim` | Critical |
| `unusual_payment_method` | Critical |
| `remote_access_request` | Critical |
| `authority_claim` | High |
| `isolation_request` | High |
| `secret_keeping_request` | High |
| `urgency_pressure` | Medium |
| `emotional_manipulation` | Medium |

Risk levels: `safe` → `suspicious` → `danger`

---

## Repository Structure

```
sonar/
├── README.md
├── LICENSE                           Apache 2.0
├── CONTRACTS.md                      Frozen interface contracts
├── LIMITATIONS.md                    Known limitations
├── start_network.bat                 Launch UI accessible on local WiFi
│
├── shared/
│   ├── schemas/classification.py     Pydantic models (CallSegmentInput, CallReport, ...)
│   ├── signals_taxonomy.yaml         Signal definitions and severity levels
│   └── prompts/classifier_v3.txt     Gemma system prompt (function calling)
│
├── track_a_desktop/                  Core pipeline (Python)
│   ├── classifier.py                 GemmaClassifier + STT correction + lexical recovery
│   ├── decision_engine.py            Risk aggregation state machine
│   ├── pipeline.py                   CLI: audio → CallReport JSON
│   ├── stt_module.py                 faster-whisper wrapper
│   ├── preflight.py                  Dependency and model health checks
│   ├── alert_sender.py               SMTP email alert
│   ├── demo_run.py                   Demo runner (batch)
│   ├── eval_harness.py               Evaluation harness
│   ├── render_report.py              Markdown report renderer
│   ├── pyproject.toml                Dependencies
│   ├── ui_local/
│   │   ├── app.py                    Browser UI server (ThreadingHTTPServer)
│   │   ├── store.py                  Call library + feedback store
│   │   └── contacts_store.py         Trusted contacts store
│   └── tests/                        Unit + integration tests (pytest)
│
├── docs/
│   ├── SPEC.md                       Technical specification
│   ├── EVAL.md                       Evaluation methodology
│   └── ETHICS.md                     Ethics checklist
│
└── track_c_dataset/                  Dataset + evaluation
    ├── ground_truth_tts.json         Ground truth for TTS synthetic dataset
    ├── audio_real/
    │   ├── ground_truth_real.json    Ground truth for real human recordings
    │   ├── audios_SONAr2026/         30 real human recordings (15 scam + 15 legit)
    │   └── eval_outputs/
    │       └── eval_summary.json     Blind eval results (real)
    ├── audio_tts/                    25 TTS synthetic recordings (WAV + transcripts)
    ├── eval_outputs_tts/
    │   └── eval_tts_summary.json     Blind eval results (TTS)
    ├── scenarios/                    YAML scenario definitions
    ├── eval_real.py                  Eval script — real recordings
    ├── eval_tts.py                   Eval script — TTS synthetic
    ├── tts_generator.py              TTS audio generation
    ├── build_ground_truth.py         Ground truth builder
    └── quality_check.py              Dataset quality checker
```

---

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with `google/gemma-4-e2b` loaded (Q4_K_M recommended)
- GPU recommended (RTX 4070+ or equivalent); CPU fallback supported but slow

```bash
pip install faster-whisper structlog pydantic python-docx requests
```

Or install from `pyproject.toml`:

```bash
cd track_a_desktop
pip install -e .
```

---

## Quick Start

### 1. Start LM Studio

Load `google/gemma-4-e2b` and start the local server on port 1234.

### 2. Run the browser UI

```bash
cd track_a_desktop
python ui_local/app.py
```

Open `http://localhost:8765` in your browser.

**To share on local WiFi (phone / tablet):**

```bash
start_network.bat   # Windows — shows your local IP
python ui_local/app.py --host 0.0.0.0 --port 8765
```

> Note: microphone recording requires localhost. Phone can view results but cannot record audio (browser HTTPS restriction).

### 3. Run the CLI pipeline on an audio file

```bash
cd track_a_desktop
python pipeline.py \
  --audio path/to/call.wav \
  --output report.json \
  --language pt-BR \
  --whisper-model small
```

Output is a validated `CallReport` JSON with per-segment classifications and final risk state.

---

## Reproduce Evaluation

### Real human recordings (30 calls)

```bash
cd track_c_dataset
python eval_real.py
# Results: audio_real/eval_outputs/eval_summary.json
```

### TTS synthetic dataset (25 calls)

```bash
cd track_c_dataset
python eval_tts.py
# Results: eval_outputs_tts/eval_tts_summary.json
```

Both scripts run fully blind — ground truth is loaded after prediction, never passed to Gemma.

---

## Tests

```bash
cd track_a_desktop
pytest tests/ -v
```

---

## Key Design Decisions

**STT correction before classification:** Whisper makes predictable errors on domain-specific terms (e.g., Argentine "DNI" transcribed as "DNA"). A deterministic regex correction layer runs before the Gemma call to reduce signal miss rate.

**Lexical recovery:** If Gemma misses a critical signal that is lexically evident in the transcript, a post-processing step adds it back. This prevents false negatives caused by LLM uncertainty.

**Background classification thread:** Whisper runs in the HTTP request thread (~2-4s). Gemma runs in a per-session background thread. This decouples STT latency from LLM latency, enabling 4-6 Gemma classifications per 2-minute call instead of 1.

**Language auto-detection:** When set to "auto", Whisper detects the language on the first chunk and locks it for the session. Supported: pt-BR, en-US, es-419.

**Conservative risk model:** The system uses probabilistic language and never claims a call is "definitely a scam." Legitimate calls with authority claims are classified as `suspicious`, not `danger`, unless critical signals are present.

---

## Limitations

See `LIMITATIONS.md` for the full list. Key constraints:

- No speaker diarization — both sides of the call are mixed in the transcript
- Tested on short scripted calls; real-world long calls may have different dynamics
- TTS synthetic audio produces more false positives than real human recordings
- LM Studio + Gemma 4 requires local GPU for acceptable latency (~10-15s per chunk)

---

## License

- Code: Apache 2.0
- Dataset and prompts: CC-BY 4.0
- Gemma 4: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
