"""
FugueForge Statistical Benchmarking (Phase 2 — M5)

Run 5× per BWV to measure stochastic variance (mean ± std).
"""
from __future__ import annotations

import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fugueforge.corpus.gold import list_gold_fugues
from fugueforge.core.planner import plan_fugue
from fugueforge.core.generator import generate_fugue, GenerationConfig
from fugueforge.core.judge import FugueJudge

NUM_RUNS = 5


def main():
    fugues = list_gold_fugues()
    judge = FugueJudge()

    print(f"{'BWV':>5} {'V':>2} {'Mean':>6} {'Std':>5} {'Min':>6} {'Max':>6}  Scores")
    print("-" * 72)

    all_means = []

    for gold in fugues:
        subject = gold.to_subject()
        plan = plan_fugue(subject, num_voices=gold.num_voices)
        config = GenerationConfig()

        scores = []
        for _ in range(NUM_RUNS):
            voices = generate_fugue(plan, config)
            result = judge.evaluate(voices, subject, plan)
            scores.append(result.total)

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        all_means.append(mean)

        scores_str = " ".join(f"{s:.1f}" for s in scores)
        print(f"{gold.bwv:>5} {gold.num_voices:>2} {mean:>6.1f} {std:>5.1f} {min(scores):>6.1f} {max(scores):>6.1f}  [{scores_str}]")

    print("-" * 72)
    overall_mean = statistics.mean(all_means)
    overall_std = statistics.stdev(all_means) if len(all_means) > 1 else 0.0
    print(f"{'AVG':>5}    {overall_mean:>6.1f} {overall_std:>5.1f}")
    print(f"\nResult: {overall_mean:.1f} ± {overall_std:.1f} (single-pass, {NUM_RUNS} runs × {len(fugues)} BWVs)")


if __name__ == "__main__":
    main()
