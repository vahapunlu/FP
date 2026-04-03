"""
Corpus evaluation pipeline.

Runs the FugueForge analyzer against gold annotations to measure
detection accuracy, and generates statistics across the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fugueforge.core.analyzer import (
    detect_answer_type,
    extract_interval_profile,
    find_subject_occurrences,
    analyze_fugue,
)
from fugueforge.core.judge import FugueJudge
from fugueforge.core.planner import plan_fugue
from fugueforge.core.generator import generate_fugue, GenerationConfig, _adapt_voice_ranges
from fugueforge.core.search import FugueSearch, SearchConfig
from fugueforge.core.representation import FugueNote, Subject

from .gold import GoldFugue, GOLD_SUBJECTS, list_gold_fugues


@dataclass
class FugueEvalResult:
    """Evaluation result for a single fugue."""
    bwv: int
    title: str
    num_voices: int
    subject_notes: int
    subject_duration: float
    subject_range: int
    # Generation results
    gen_theory: float = 0.0
    gen_structure: float = 0.0
    gen_style: float = 0.0
    gen_aesthetic: float = 0.0
    gen_total: float = 0.0
    # Search results (beam search)
    search_theory: float = 0.0
    search_structure: float = 0.0
    search_style: float = 0.0
    search_aesthetic: float = 0.0
    search_total: float = 0.0


@dataclass
class CorpusReport:
    """Aggregate report across the entire corpus."""
    results: list[FugueEvalResult] = field(default_factory=list)

    @property
    def num_fugues(self) -> int:
        return len(self.results)

    @property
    def avg_gen_total(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.gen_total for r in self.results) / len(self.results)

    @property
    def avg_search_total(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.search_total for r in self.results) / len(self.results)

    @property
    def avg_theory(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.search_theory for r in self.results) / len(self.results)

    @property
    def avg_structure(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.search_structure for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"FugueForge Corpus Evaluation — {self.num_fugues} fugues",
            f"{'='*60}",
            "",
            f"{'BWV':<6} {'Voices':<7} {'Gen':>6} {'Search':>8} {'Theory':>8} {'Struct':>8} {'Style':>7} {'Aesth':>7}",
            f"{'-'*60}",
        ]
        for r in sorted(self.results, key=lambda x: x.bwv):
            lines.append(
                f"{r.bwv:<6} {r.num_voices:<7} "
                f"{r.gen_total:>6.1f} {r.search_total:>8.1f} "
                f"{r.search_theory:>8.1f} {r.search_structure:>8.1f} "
                f"{r.search_style:>7.1f} {r.search_aesthetic:>7.1f}"
            )
        lines.extend([
            f"{'-'*60}",
            f"{'AVG':<6} {'':>6} {self.avg_gen_total:>6.1f} {self.avg_search_total:>8.1f} "
            f"{self.avg_theory:>8.1f} {self.avg_structure:>8.1f}",
            "",
            f"Search improvement over single-pass: "
            f"{self.avg_search_total - self.avg_gen_total:+.1f}",
        ])
        return "\n".join(lines)


def evaluate_single(
    gold: GoldFugue,
    num_gen_candidates: int = 8,
    search_config: Optional[SearchConfig] = None,
) -> FugueEvalResult:
    """
    Evaluate our system's ability to generate a fugue from a gold subject.
    Generates using both single-pass and beam search, returns scores.
    """
    subject = gold.to_subject()
    judge = FugueJudge()

    result = FugueEvalResult(
        bwv=gold.bwv,
        title=gold.title,
        num_voices=gold.num_voices,
        subject_notes=len(gold.subject_pitches),
        subject_duration=subject.duration,
        subject_range=subject.pitch_range[1] - subject.pitch_range[0],
    )

    plan = plan_fugue(
        subject,
        num_voices=gold.num_voices,
        target_measures=max(16, gold.num_voices * 8),
    )

    # Single-pass: best of N
    best_gen = -1.0
    for _ in range(num_gen_candidates):
        voices = generate_fugue(plan)
        js = judge.evaluate(voices, subject, plan)
        if js.total > best_gen:
            best_gen = js.total
            result.gen_theory = js.theory
            result.gen_structure = js.structure
            result.gen_style = js.style
            result.gen_aesthetic = js.aesthetic
            result.gen_total = js.total

    # Beam search
    if search_config is None:
        search_config = SearchConfig(
            candidates_per_section=6,
            survivors_per_section=3,
        )

    _, sj = FugueSearch(plan, search_config, judge).search()
    result.search_theory = sj.theory
    result.search_structure = sj.structure
    result.search_style = sj.style
    result.search_aesthetic = sj.aesthetic
    result.search_total = sj.total

    return result


def evaluate_corpus(
    bwv_list: Optional[list[int]] = None,
    num_gen_candidates: int = 8,
    search_config: Optional[SearchConfig] = None,
    verbose: bool = True,
) -> CorpusReport:
    """
    Evaluate the system across multiple gold fugues.
    If bwv_list is None, evaluates all gold fugues.
    """
    if bwv_list is None:
        fugues = list_gold_fugues()
    else:
        fugues = [GOLD_SUBJECTS[b] for b in bwv_list if b in GOLD_SUBJECTS]

    report = CorpusReport()

    for gold in fugues:
        if verbose:
            print(f"Evaluating BWV {gold.bwv} ({gold.title})...", end=" ", flush=True)

        result = evaluate_single(gold, num_gen_candidates, search_config)
        report.results.append(result)

        if verbose:
            print(
                f"Gen={result.gen_total:.1f} Search={result.search_total:.1f} "
                f"(T={result.search_theory:.1f} S={result.search_structure:.1f})"
            )

    return report
