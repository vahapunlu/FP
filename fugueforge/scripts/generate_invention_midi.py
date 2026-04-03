"""
FugueForge — Invention MIDI Generator

Generates MIDI files for all 15 Bach Two-Part Inventions (BWV 772-786).

Usage:
    python -m fugueforge.scripts.generate_invention_midi          # all 15
    python -m fugueforge.scripts.generate_invention_midi 772 773  # specific BWVs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fugueforge.corpus.inventions import GOLD_INVENTIONS, list_gold_inventions
from fugueforge.core.invention_planner import plan_invention
from fugueforge.core.invention_generator import generate_invention, _invention_config
from fugueforge.core.voice_utils import voices_to_score
from fugueforge.core.judge import FugueJudge, print_judge_score
from fugueforge.core.rules import is_consonance, is_dissonance


def analyze_quality(voices: dict[int, list], label: str = "") -> dict:
    """Analyze dissonance rate, stepwise %, and overlap for a 2-voice result."""
    total_vert = 0
    dissonant = 0
    total_mel = 0
    stepwise = 0
    large_leaps = 0

    for v_idx in sorted(voices.keys()):
        notes = sorted(voices[v_idx], key=lambda n: n.offset)
        # Melodic intervals
        for j in range(1, len(notes)):
            interval = abs(notes[j].pitch - notes[j - 1].pitch)
            total_mel += 1
            if interval <= 2:
                stepwise += 1
            if interval > 7:
                large_leaps += 1

    # Vertical dissonance (only 2 voices)
    v_keys = sorted(voices.keys())
    if len(v_keys) >= 2:
        v0 = sorted(voices[v_keys[0]], key=lambda n: n.offset)
        v1 = sorted(voices[v_keys[1]], key=lambda n: n.offset)
        for n0 in v0:
            for n1 in v1:
                ov_start = max(n0.offset, n1.offset)
                ov_end = min(n0.offset + n0.duration, n1.offset + n1.duration)
                if ov_end > ov_start + 0.01:
                    total_vert += 1
                    if is_dissonance(n0.pitch - n1.pitch):
                        dissonant += 1

    diss_pct = (dissonant / total_vert * 100) if total_vert > 0 else 0
    step_pct = (stepwise / total_mel * 100) if total_mel > 0 else 0
    leap_pct = (large_leaps / total_mel * 100) if total_mel > 0 else 0

    return {
        "diss_pct": diss_pct,
        "step_pct": step_pct,
        "leap_pct": leap_pct,
        "total_notes": sum(len(voices[v]) for v in voices),
    }


def generate_all_invention_midi(
    bwv_list: list[int] | None = None,
    output_dir: str = "output/inventions",
) -> None:
    """Generate MIDI for all (or selected) inventions with quality reporting."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if bwv_list is None:
        inventions = list_gold_inventions()
    else:
        inventions = [GOLD_INVENTIONS[b] for b in bwv_list if b in GOLD_INVENTIONS]

    judge = FugueJudge()

    print("=" * 72)
    print("FugueForge — Two-Part Invention Generation")
    print("=" * 72)
    print()

    results: list[dict] = []

    for inv in inventions:
        print(f"━━━ BWV {inv.bwv}: {inv.title} ━━━")

        subject = inv.to_subject()
        plan = plan_invention(
            subject,
            target_measures=inv.approx_measures,
            time_signature=inv.time_signature,
            answer_at_fifth=inv.answer_at_fifth,
        )

        # Generate invention
        t0 = time.time()
        config = _invention_config()
        voices = generate_invention(plan, config, answer_at_fifth=inv.answer_at_fifth)
        elapsed = time.time() - t0

        # Judge score (reuse fugue judge — structure axis less relevant)
        js = judge.evaluate(voices, subject, plan)

        # Quality analysis
        qa = analyze_quality(voices)

        # Save MIDI
        score = voices_to_score(voices, title=f"BWV {inv.bwv} — {inv.title}")
        midi_path = out / f"bwv{inv.bwv}_invention.mid"
        score.write("midi", fp=str(midi_path))

        print(f"  Score: {js.total:5.1f}  (T={js.theory:.1f} S={js.structure:.1f} "
              f"St={js.style:.1f} A={js.aesthetic:.1f})")
        print(f"  Quality: diss={qa['diss_pct']:.1f}%  stepwise={qa['step_pct']:.0f}%  "
              f"leaps={qa['leap_pct']:.1f}%  notes={qa['total_notes']}")
        print(f"  → MIDI: {midi_path}  ({elapsed:.1f}s)")
        print()

        results.append({
            "bwv": inv.bwv,
            "score": js.total,
            "theory": js.theory,
            "structure": js.structure,
            "style": js.style,
            "aesthetic": js.aesthetic,
            **qa,
        })

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'BWV':>4}  {'Score':>6}  {'Theory':>6}  {'Struct':>6}  "
          f"{'Style':>6}  {'Aesth':>6}  {'Diss%':>5}  {'Step%':>5}  {'Leaps%':>6}")
    print("─" * 72)

    for r in results:
        print(f"{r['bwv']:>4}  {r['score']:>6.1f}  {r['theory']:>6.1f}  {r['structure']:>6.1f}  "
              f"{r['style']:>6.1f}  {r['aesthetic']:>6.1f}  {r['diss_pct']:>5.1f}  "
              f"{r['step_pct']:>5.0f}  {r['leap_pct']:>6.1f}")

    print("─" * 72)
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_diss = sum(r["diss_pct"] for r in results) / len(results)
    avg_step = sum(r["step_pct"] for r in results) / len(results)
    avg_leap = sum(r["leap_pct"] for r in results) / len(results)
    print(f" AVG  {avg_score:>6.1f}  {'':>6}  {'':>6}  {'':>6}  {'':>6}  "
          f"{avg_diss:>5.1f}  {avg_step:>5.0f}  {avg_leap:>6.1f}")

    print(f"\nMIDI files saved to: {out.resolve()}")
    print(f"Total inventions generated: {len(results)}")


if __name__ == "__main__":
    bwv_args = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    generate_all_invention_midi(bwv_args)
