"""
FugueForge — Phase 1: Listen & Validate

Generates MIDI files for all 13 gold-standard BWV subjects.
Also establishes baselines:
  - Random baseline: shuffle pitches within range
  - Single-pass: best of N without beam search
  - Beam search: section-by-section search (our main pipeline)

Usage:
    python -m fugueforge.scripts.generate_midi          # all 13
    python -m fugueforge.scripts.generate_midi 847 851  # specific BWVs
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from fugueforge.corpus.gold import GOLD_SUBJECTS, list_gold_fugues
from fugueforge.core.planner import plan_fugue
from fugueforge.core.generator import (
    GenerationConfig,
    generate_fugue,
    voices_to_score,
    _adapt_voice_ranges,
)
from fugueforge.core.judge import FugueJudge, print_judge_score
from fugueforge.core.search import FugueSearch, SearchConfig
from fugueforge.core.representation import FugueNote, EntryRole


# ---------------------------------------------------------------------------
# Random baseline: generates notes randomly within voice ranges
# ---------------------------------------------------------------------------

def generate_random_baseline(
    plan,
    config: GenerationConfig | None = None,
) -> dict[int, list[FugueNote]]:
    """Generate a 'fugue' with random pitches — our floor baseline."""
    config = config or GenerationConfig()
    config = _adapt_voice_ranges(config, plan.subject, plan.num_voices)

    total_dur = sum(s.estimated_duration for s in plan.sections)
    voices: dict[int, list[FugueNote]] = {}

    for v in range(plan.num_voices):
        lo, hi = config.voice_ranges.get(v, (48, 72))
        notes: list[FugueNote] = []
        offset = 0.0
        while offset < total_dur:
            pitch = random.randint(lo, hi)
            dur = random.choice([0.5, 0.5, 1.0, 0.5])
            notes.append(FugueNote(
                pitch=pitch, duration=dur, voice=v,
                offset=offset, role=EntryRole.FREE_COUNTERPOINT,
            ))
            offset += dur
        voices[v] = notes

    return voices


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_all_midi(
    bwv_list: list[int] | None = None,
    output_dir: str = "output/midi",
    num_single_pass: int = 5,
) -> None:
    """Generate MIDI for all (or selected) BWV subjects with full reporting."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if bwv_list is None:
        fugues = list_gold_fugues()
    else:
        fugues = [GOLD_SUBJECTS[b] for b in bwv_list if b in GOLD_SUBJECTS]

    judge = FugueJudge()

    # Header
    print("=" * 72)
    print("FugueForge — Phase 1: MIDI Generation & Baseline Comparison")
    print("=" * 72)
    print()

    # Results table
    results: list[dict] = []

    for gold in fugues:
        print(f"━━━ BWV {gold.bwv}: {gold.title} ({gold.num_voices}v) ━━━")

        subject = gold.to_subject()
        plan = plan_fugue(
            subject,
            num_voices=gold.num_voices,
            target_measures=max(16, gold.num_voices * 8),
        )

        # --- 1. Random baseline ---
        t0 = time.time()
        random_voices = generate_random_baseline(plan)
        random_js = judge.evaluate(random_voices, subject, plan)
        t_random = time.time() - t0
        print(f"  Random baseline:  {random_js.total:5.1f}  ({t_random:.1f}s)")

        # --- 2. Single-pass (best of N, no search) ---
        t0 = time.time()
        best_single_score = -1.0
        best_single_voices = None
        best_single_js = None
        for _ in range(num_single_pass):
            voices = generate_fugue(plan)
            js = judge.evaluate(voices, subject, plan)
            if js.total > best_single_score:
                best_single_score = js.total
                best_single_voices = voices
                best_single_js = js
        t_single = time.time() - t0
        print(f"  Single-pass (×{num_single_pass}): {best_single_js.total:5.1f}  ({t_single:.1f}s)")

        # --- 3. Beam search ---
        t0 = time.time()
        search_cfg = SearchConfig(
            candidates_per_section=6,
            survivors_per_section=3,
        )
        search_voices, search_js = FugueSearch(plan, search_cfg, judge).search()
        t_search = time.time() - t0
        print(f"  Beam search:      {search_js.total:5.1f}  ({t_search:.1f}s)")

        # --- Improvement ---
        delta_vs_random = search_js.total - random_js.total
        delta_vs_single = search_js.total - best_single_js.total
        print(f"  Δ vs random: +{delta_vs_random:.1f}  |  Δ vs single: {delta_vs_single:+.1f}")

        # --- Save MIDI (beam search result) ---
        score = voices_to_score(search_voices, title=f"BWV {gold.bwv} — {gold.title}")
        midi_path = out / f"bwv{gold.bwv}_{gold.num_voices}v.mid"
        score.write("midi", fp=str(midi_path))
        print(f"  → MIDI saved: {midi_path}")

        # --- Save single-pass MIDI for comparison ---
        single_score = voices_to_score(
            best_single_voices, title=f"BWV {gold.bwv} — Single Pass",
        )
        single_path = out / f"bwv{gold.bwv}_{gold.num_voices}v_single.mid"
        single_score.write("midi", fp=str(single_path))

        # --- Detailed score breakdown ---
        print(f"  Score breakdown:  T={search_js.theory:.1f}  S={search_js.structure:.1f}  "
              f"St={search_js.style:.1f}  A={search_js.aesthetic:.1f}")
        if search_js.notes:
            for note in search_js.notes[:5]:
                print(f"    • {note}")

        results.append({
            "bwv": gold.bwv,
            "voices": gold.num_voices,
            "random": random_js.total,
            "single": best_single_js.total,
            "search": search_js.total,
            "theory": search_js.theory,
            "structure": search_js.structure,
            "style": search_js.style,
            "aesthetic": search_js.aesthetic,
        })
        print()

    # --- Summary table ---
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print(f"{'BWV':<6} {'V':>2} {'Random':>7} {'Single':>7} {'Search':>7} "
          f"{'Theory':>7} {'Struct':>7} {'Style':>7} {'Aesth':>7}")
    print("─" * 72)

    for r in results:
        print(f"{r['bwv']:<6} {r['voices']:>2} {r['random']:>7.1f} {r['single']:>7.1f} "
              f"{r['search']:>7.1f} {r['theory']:>7.1f} {r['structure']:>7.1f} "
              f"{r['style']:>7.1f} {r['aesthetic']:>7.1f}")

    print("─" * 72)

    n = len(results)
    if n > 0:
        avg = lambda key: sum(r[key] for r in results) / n
        print(f"{'AVG':<6} {'':>2} {avg('random'):>7.1f} {avg('single'):>7.1f} "
              f"{avg('search'):>7.1f} {avg('theory'):>7.1f} {avg('structure'):>7.1f} "
              f"{avg('style'):>7.1f} {avg('aesthetic'):>7.1f}")
        print()
        print(f"Search vs Random: +{avg('search') - avg('random'):.1f}")
        print(f"Search vs Single: {avg('search') - avg('single'):+.1f}")

    print()
    print(f"MIDI files saved to: {out.resolve()}")
    print(f"Total fugues generated: {n}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bwv_args = None
    if len(sys.argv) > 1:
        bwv_args = [int(b) for b in sys.argv[1:]]

    generate_all_midi(bwv_list=bwv_args)
