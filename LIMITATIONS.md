# Limitations

This document records the current limitations of the Senti prototype so the demo and submission remain accurate.

## Current Scope

Senti is currently a local desktop prototype for scam-call risk detection. The demonstrated path is:

```text
local audio -> faster-whisper -> Ollama gemma4:e2b -> deterministic Decision Engine -> CallReport -> local report/UI
```

The desktop path is useful for demo, auditability, prompt iteration, and local evidence collection. It is not yet the final Android implementation.

## Evidence Limitations

1. Current audio evidence is based on synthetic TTS samples, not real scam recordings or calls from older adults.
2. The strongest post-mitigation metric result is a two-sample smoke test, not a statistically meaningful benchmark.
3. The current TTS dataset has 25 samples, below the original 75-sample evaluation target.
4. Many TTS samples are shorter than the original 60-180 second target call duration.
5. `suspicious` is not yet evaluated deeply enough; current strongest smoke evidence focuses on `safe` versus `danger`.
6. Per-signal F1 is not yet proven across all taxonomy signals.
7. Fairness and robustness across PT-BR and EN-US are not yet established with enough real-STT runs.

## Runtime Limitations

1. `gemma4:e2b` currently runs CPU-only through Ollama in this environment.
2. Observed runtime can be several minutes per short sample.
3. The current latency does not meet the original desktop latency target.
4. The current desktop runtime does not represent final mobile or edge-device performance.

## Platform Limitations

1. Android implementation remains pending.
2. MediaPipe LLM Inference, LiteRT, `whisper.cpp` Android NDK, `AudioRecord`, and foreground-service tests are not completed.
3. No physical Android device validation has been completed yet.
4. E2B-to-E4B escalation/routing is not implemented in the current desktop prototype.

## Product Limitations

1. Senti is a safety assistant, not a definitive fraud detector.
2. Results should be treated as risk warnings, not proof of fraud.
3. The prototype should not collect or process real sensitive personal data during demos.
4. The local UI supports upload of WAV/MP3 files, not direct browser microphone recording yet.
5. Human feedback is saved locally for evaluation and iteration; it does not automatically train or update the model.

## Known Failure Modes

1. Speech-to-text errors can change risk classification, especially for short financial terms such as Pix.
2. LLM outputs may omit tool calls; the desktop classifier uses a JSON fallback path to recover.
3. LLM outputs may include unsupported signals; post-processing suppresses some unsupported cases, but this is not exhaustive.
4. User-facing language can drift; the report path includes repair logic, but this should continue to be tested.
5. Keyword-like mitigations can improve known cases while still missing paraphrased, adversarial, or noisy scams.

## Reporting Guidance

Claims that are currently supportable:

- The local desktop prototype runs end-to-end with local audio, local STT, local Gemma inference, deterministic risk aggregation, and local report generation.
- The explicit Pix scam demo currently reaches `danger/red`.
- A known legitimate-family false-danger case was mitigated in the current two-sample smoke pair.
- All current evidence is local and auditable through generated JSON/Markdown artifacts.

Claims that should not be made yet:

- Production-ready scam detection.
- Real-time mobile protection.
- Final benchmark performance against the original evaluation targets.
- Broad robustness to accents, noisy calls, adversarial wording, or real-world call-center audio.
- Medical, legal, financial, or law-enforcement certainty.

## Next Evidence Needed

1. Run a larger stratified subset from the 25-sample TTS dataset with real STT and `gemma4:e2b`.
2. Include at least one `suspicious` sample in the next short evaluation batch.
3. Add real user-recorded non-sensitive test audio through the local UI and save feedback.
4. Revisit GPU acceleration or a mobile runtime path to address latency.
5. Complete Android spikes only after the desktop deliverable is packaged cleanly.
