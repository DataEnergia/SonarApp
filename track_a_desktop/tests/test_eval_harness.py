# Copyright 2026 Carlos Alejandro Urzagasti
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from eval_harness import _filter_samples, compute_metrics


def test_compute_metrics_for_risk_levels_and_false_danger_rate() -> None:
    results = [
        {
            "type": "legitimate",
            "expected_risk": "safe",
            "predicted_risk": "safe",
            "correct_risk": True,
            "expected_signals": [],
            "predicted_signals": [],
            "latency_seconds": 1.0,
        },
        {
            "type": "legitimate",
            "expected_risk": "suspicious",
            "predicted_risk": "danger",
            "correct_risk": False,
            "expected_signals": ["authority_claim"],
            "predicted_signals": ["authority_claim", "financial_request"],
            "latency_seconds": 3.0,
        },
        {
            "type": "scam",
            "expected_risk": "danger",
            "predicted_risk": "danger",
            "correct_risk": True,
            "expected_signals": ["financial_request"],
            "predicted_signals": ["financial_request"],
            "latency_seconds": 2.0,
        },
    ]

    metrics = compute_metrics(results)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["false_danger_rate_on_legitimate"] == 0.5
    assert metrics["confusion_matrix"]["danger"]["danger"] == 1
    assert metrics["per_risk"]["danger"]["precision"] == 0.5
    assert metrics["per_signal"]["financial_request"]["precision"] == 0.5
    assert metrics["latency_seconds"]["mean"] == 2.0


def test_filter_samples_by_name_and_split() -> None:
    samples = [
        {"audio_file": "a.wav", "split": "train"},
        {"audio_file": "b.wav", "split": "holdout"},
        {"audio_file": "c.wav", "split": "train"},
    ]

    filtered = _filter_samples(samples, limit=None, split="train", sample_names={"c.wav", "b.wav"})

    assert filtered == [{"audio_file": "c.wav", "split": "train"}]
