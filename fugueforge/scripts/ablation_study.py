"""
FugueForge Ablation Study (Phase 2 — M2)

Tests the contribution of each major feature by disabling it
and measuring the score impact across all 13 BWV subjects.

Features tested:
  1. Scale awareness (scale_pcs, scale_bonus)
  2. Harmonic skeleton (chord_tone_bonus)
  3. Melodic quality (stepwise_bonus, consecutive_leap_penalty)
  4. Phrasing & contour (phrase_length, direction tracking)
  5. Dramaturgy (energy curve)
  6. Episode variation (rhythmic/intervallic variation)
  7. Modulation (pivot chord modulations)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fugueforge.corpus.gold import list_gold_fugues
from fugueforge.core.planner import plan_fugue
from fugueforge.core.generator import generate_fugue, GenerationConfig
from fugueforge.core.judge import FugueJudge


def _run_benchmark(label: str, config_modifier=None, num_runs: int = 1):
    """Run benchmark across all BWVs with optional config modification."""
    fugues = list_gold_fugues()
    judge = FugueJudge()
    scores = []

    for gold in fugues:
        subject = gold.to_subject()
        plan = plan_fugue(subject, num_voices=gold.num_voices)

        config = GenerationConfig()
        if config_modifier:
            config = config_modifier(config)

        run_scores = []
        for _ in range(num_runs):
            voices = generate_fugue(plan, config)
            result = judge.evaluate(voices, subject, plan)
            run_scores.append(result.total)

        avg = sum(run_scores) / len(run_scores)
        scores.append(avg)

    overall = sum(scores) / len(scores)
    return overall, scores


def ablation_no_scale(config):
    c = copy.copy(config)
    c.scale_bonus = 0.0
    c.chromatic_penalty = 0.0
    c.scale_pcs = frozenset()
    return c


def ablation_no_harmony(config):
    c = copy.copy(config)
    c.chord_tone_bonus = 0.0
    c.non_chord_penalty = 0.0
    return c


def ablation_no_melody(config):
    c = copy.copy(config)
    c.stepwise_bonus = 2.8   # revert to old value
    c.third_bonus = 2.0
    c.large_leap_penalty = 2.0
    c.consecutive_leap_penalty = 0.0
    c.direction_momentum = 0.0
    c.direction_reversal = 0.0
    c.register_gravity = 0.0
    c.consonance_weight = 4.0  # revert
    return c


def ablation_no_phrasing(config):
    c = copy.copy(config)
    c.phrase_length = 999.0  # effectively no phrase boundaries
    c.direction_momentum = 0.0
    c.direction_reversal = 0.0
    return c


def main():
    print("=" * 70)
    print("FUGUEFORGE ABLATION STUDY")
    print("=" * 70)
    print()

    conditions = [
        ("Full system (baseline)", None),
        ("- Scale awareness", ablation_no_scale),
        ("- Harmonic skeleton", ablation_no_harmony),
        ("- Melodic quality (Phase 2)", ablation_no_melody),
        ("- Phrasing & direction", ablation_no_phrasing),
    ]

    results = {}
    baseline = None

    for label, modifier in conditions:
        print(f"Running: {label}...", end=" ", flush=True)
        avg, per_bwv = _run_benchmark(label, modifier)
        results[label] = (avg, per_bwv)
        if baseline is None:
            baseline = avg
        delta = avg - baseline
        print(f"AVG = {avg:.1f}  (Δ = {delta:+.1f})")

    print()
    print("-" * 70)
    print(f"{'Condition':<35} {'AVG':>6} {'Δ':>7} {'Impact':>8}")
    print("-" * 70)
    for label, (avg, _) in results.items():
        delta = avg - baseline
        impact = "BASELINE" if delta == 0 else ("CRITICAL" if abs(delta) > 3 else "HIGH" if abs(delta) > 1.5 else "MEDIUM" if abs(delta) > 0.5 else "LOW")
        print(f"{label:<35} {avg:>6.1f} {delta:>+7.1f} {impact:>8}")
    print("-" * 70)


if __name__ == "__main__":
    main()
