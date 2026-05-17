# EVAL — Evaluation Methodology

> How we measure whether Senti works.

---

## 1. What we measure

| Metric | Target | Track |
|---|---|---|
| F1 on `risk_level=danger` | ≥ 0.85 | A → D |
| F1 on `risk_level=suspicious` | ≥ 0.70 | A → D |
| Precision on `risk_level=danger` | ≥ 0.90 (FP is costly) | A → D |
| False positive rate on legitimate calls | < 10% | A → D |
| Per-signal F1 (10 signals) | ≥ 0.75 each | A → D |
| Inference latency (desktop, RTX 4070) | p50 < 1.5s, p95 < 3s | A |
| Inference latency (Android 4GB) | p50 < 3s, p95 < 6s | B |
| End-to-end latency: audio chunk → UI alert | < 10s | B |
| Cross-language consistency (PT vs EN) | F1 difference < 0.10 | D |
| Audit log completeness | 100% of classifications logged | A, B |

## 2. Dataset

Built by Track C. Composition:

- 50 scam audios (5 per signal × 10 signals)
- 25 legitimate audios (banks, family, telemarketing, hospitals, INSS, govt agencies)
- Total: 75 audios
- Language split: ~70% PT-BR, ~30% EN-US
- Voice variation: 3 female + 3 male TTS voices
- Duration: 60-180s each

Stored in `track_c_dataset/audio_dataset/` with ground truth in `track_c_dataset/ground_truth.json`.

## 3. Ground truth structure

Per-audio, we annotate:

- Primary signal (the headline pattern)
- All signals present
- Final risk level after full audio is processed
- Risk evolution (which time-ranges should be at which level)

The risk-evolution annotation is critical: it lets us measure not just final accuracy, but **how quickly** the system escalates correctly. A scam detected at second 60 is more valuable than the same scam detected at second 120.

## 4. Evaluation procedure

### 4.1 Single-audio evaluation

```bash
cd track_a_desktop
python pipeline.py --audio ../track_c_dataset/audio_dataset/scam_authority_001.wav \
                   --language pt-BR \
                   --output report.json
```

Produces `CallReport`. Compare to ground truth entry.

### 4.2 Batch evaluation

```bash
cd track_a_desktop
python eval_harness.py --dataset ../track_c_dataset/audio_dataset/ \
                       --ground-truth ../track_c_dataset/ground_truth.json \
                       --output ../track_d_eval/reports/eval_run_$(date +%Y%m%d_%H%M%S).json
```

Produces per-audio results plus aggregate statistics.

### 4.3 Confusion matrix

3x3 confusion matrix on `risk_level`:

```
              predicted
              safe  suspicious  danger
actual safe   [TN]  [FP_sus]    [FP_dan]
       susp.  [FN]  [TP]        [over]
       danger [miss][under]     [TP]
```

Critical cells:
- `FP_dan` (legitimate call → danger): must be very low (< 5%)
- `miss` (real scam → safe): must be very low (< 5%)
- `under` (danger scam → suspicious): acceptable if signals were still shown

### 4.4 Per-signal evaluation

For each of the 10 signals:

- Precision: of times the system detected signal X, how often was X actually present?
- Recall: of times signal X was actually present, how often did the system detect it?
- F1: harmonic mean

Aggregate over all 75 audios.

### 4.5 Latency measurement

Wall-clock from `pipeline.py` start to JSON output. Broken down per stage:

- Audio load
- STT
- Per-segment LLM inference
- Decision engine
- JSON output

Recorded in `latency_breakdown` field of each `CallReport`.

## 5. Iteration protocol (Track D)

Each iteration of prompt or model takes ~1 hour:

1. Apply candidate prompt (`shared/prompts/classifier_v<N+1>.txt`)
2. Run batch evaluation
3. Compare to previous iteration:
   - Did F1 improve?
   - Did FP rate improve?
   - Per-signal trade-offs?
4. If improvement: keep, document delta
5. If regression: revert, try different change
6. Document iteration in `track_d_eval/reports/iteration_<N>.md`

Stop when:
- F1 targets met, OR
- 5 iterations without improvement, OR
- Time budget exhausted (Day 4 end)

## 6. Holdout protocol

To prevent overfitting:

- 20% of dataset (15 audios) is **holdout** — never used in prompt iteration
- Track D only sees the 80% (60 audios)
- Final write-up reports results on holdout separately
- Holdout split is fixed at dataset creation time (Track C)

## 7. Fairness analysis

Required for the write-up:

| Slice | Metric | Tolerance |
|---|---|---|
| PT-BR vs EN-US | F1 difference | < 0.10 |
| Female-voice vs male-voice scam audios | F1 difference | < 0.10 |
| Short audio (< 90s) vs long audio (> 90s) | F1 difference | < 0.10 |

If any slice exceeds tolerance, declare in write-up and discuss mitigation.

## 8. Reproducibility checklist

- [ ] All seeds set explicitly in code
- [ ] Model versions pinned (e.g. `gemma4:e2b` with hash)
- [ ] Dataset versioned (`ground_truth.json` has `version` field)
- [ ] Eval results timestamped in filename
- [ ] All hyperparameters logged in report
- [ ] Hardware specs noted (CPU, GPU, RAM)
- [ ] Repro command provided in `track_d_eval/REPRO.md`

## 9. Limitations to declare in write-up

1. Dataset is synthetic (TTS, not real scam recordings). Real scams have voice artifacts, background noise, network compression that our dataset approximates with bandpass filtering but does not fully capture.
2. 75 audios is small; results are point estimates with wide confidence intervals.
3. Holdout is sampled from same distribution as training; out-of-distribution scams (new patterns, regional dialects) are not tested.
4. Real elderly users are not in the loop for this MVP. Future work requires co-design with this population.
5. Adversarial robustness (a scammer who knows Senti exists) is not formally tested.
