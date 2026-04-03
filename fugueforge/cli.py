"""
FugueForge CLI

Interactive composer loop and batch generation interface.
Supports both step-by-step fugue copilot mode and
automated full-generation mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.representation import (
    FugueNote,
    FuguePlan,
    Subject,
    VoiceType,
)
from .core.analyzer import analyze_fugue_file
from .core.planner import plan_fugue, print_plan
from .core.generator import generate_fugue, GenerationConfig, voices_to_score
from .core.judge import FugueJudge, print_judge_score
from .core.search import FugueSearch, SearchConfig


def cmd_generate(args: argparse.Namespace) -> None:
    """Full automated fugue generation pipeline."""
    print("═══ FugueForge: Automated Generation ═══\n")

    # Step 1: Create or load subject
    if args.subject_file:
        from .core.representation import load_subject_from_file
        subject = load_subject_from_file(args.subject_file)
        print(f"Loaded subject from {args.subject_file}")
    else:
        subject = _create_demo_subject(args.key, args.time_sig)
        print(f"Using demo subject in {args.key}")

    print(f"Subject: {len(subject.notes)} notes, {subject.duration:.1f} beats")
    print(f"Range: {subject.pitch_range}")
    print()

    # Step 2: Plan
    plan = plan_fugue(
        subject,
        num_voices=args.voices,
        target_measures=args.measures,
        time_signature=args.time_sig,
    )
    print(print_plan(plan))
    print()

    # Step 3: Generate candidates
    config = GenerationConfig()
    judge = FugueJudge()

    if args.search:
        # Beam search mode: section-by-section exploration
        print("Using beam search (section-by-section)...")
        search_cfg = SearchConfig(
            candidates_per_section=args.candidates,
            survivors_per_section=max(2, args.candidates // 2),
            gen_config=config,
        )
        best_voices, best_js = FugueSearch(plan, search_cfg, judge).search()
        best_score = best_js.total if best_js else 0.0
        print(f"  Best: {best_score:.1f}/100")
    else:
        num_candidates = args.candidates
        print(f"Generating {num_candidates} candidates...")

        best_score = -1.0
        best_voices = None
        best_js = None

        for i in range(num_candidates):
            voices = generate_fugue(plan, config)
            js = judge.evaluate(voices, subject, plan)

            marker = ""
            if js.total > best_score:
                best_score = js.total
                best_voices = voices
                best_js = js
                marker = " ★"

            print(f"  Candidate {i + 1}: {js.total:.1f}/100{marker}")

    print()

    # Step 4: Show best result
    if best_js:
        print(print_judge_score(best_js))

    # Step 5: Export
    if best_voices and args.output:
        score = voices_to_score(best_voices, title="FugueForge Fugue")

        output = Path(args.output)
        if output.suffix == ".mid":
            midi_file = score.write("midi", fp=str(output))
            print(f"\nExported MIDI: {midi_file}")
        elif output.suffix == ".xml":
            xml_file = score.write("musicxml", fp=str(output))
            print(f"\nExported MusicXML: {xml_file}")
        elif output.suffix == ".pdf":
            pdf_file = score.write("lily.pdf", fp=str(output))
            print(f"\nExported PDF: {pdf_file}")
        else:
            # Default to MIDI
            midi_file = score.write("midi", fp=str(output.with_suffix(".mid")))
            print(f"\nExported MIDI: {midi_file}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze an existing fugue."""
    print(f"═══ FugueForge: Analysis ═══\n")
    print(f"Analyzing: {args.input_file}\n")

    analysis = analyze_fugue_file(args.input_file)

    print(f"Voices: {analysis.num_voices}")
    if analysis.subject:
        print(f"Subject: {len(analysis.subject.notes)} notes, {analysis.subject.duration:.1f} beats")
        print(f"Subject intervals: {analysis.subject.interval_sequence}")

    print(f"\nSubject occurrences: {len(analysis.subject_occurrences)}")
    for occ in analysis.subject_occurrences:
        print(f"  Voice {occ.voice}: {occ.role.name} at beat {occ.start_offset:.1f} (T+{occ.transposition})")

    print(f"\nSections: {len(analysis.sections)}")
    for sec in analysis.sections:
        print(f"  [{sec.section_type.name}] {sec.start_offset:.1f}–{sec.end_offset:.1f} ({sec.key_area})")

    print(f"\nCadences: {len(analysis.cadences)}")
    for cad in analysis.cadences:
        print(f"  {cad.type} at beat {cad.offset:.1f} ({cad.key_area})")


def cmd_plan(args: argparse.Namespace) -> None:
    """Plan a fugue structure."""
    if args.subject_file:
        from .core.representation import load_subject_from_file
        subject = load_subject_from_file(args.subject_file)
    else:
        subject = _create_demo_subject(args.key, args.time_sig)

    plan = plan_fugue(
        subject,
        num_voices=args.voices,
        target_measures=args.measures,
        time_signature=args.time_sig,
    )
    print(print_plan(plan))


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate across the gold corpus."""
    from .corpus.pipeline import evaluate_corpus
    from .core.search import SearchConfig

    print("FugueForge Corpus Evaluation\n")

    search_cfg = SearchConfig(
        candidates_per_section=args.beam,
        survivors_per_section=max(2, args.beam // 2),
    )

    report = evaluate_corpus(
        bwv_list=args.bwv,
        num_gen_candidates=args.candidates,
        search_config=search_cfg,
        verbose=True,
    )
    print()
    print(report.summary())


# ---------------------------------------------------------------------------
# Demo subject
# ---------------------------------------------------------------------------

def _create_demo_subject(key_sig: str = "C", time_sig: str = "4/4") -> Subject:
    """Create a demo subject inspired by Bach WTC I fugue in C minor."""
    # C minor subject-like pattern: C-Eb-D-C-B-C-G (simplified)
    base_notes = [
        (60, 1.0),   # C4
        (63, 0.5),   # Eb4
        (62, 0.5),   # D4
        (60, 0.5),   # C4
        (59, 0.5),   # B3
        (60, 0.5),   # C4
        (55, 1.5),   # G3
    ]

    notes: list[FugueNote] = []
    offset = 0.0
    for pitch, dur in base_notes:
        notes.append(FugueNote(
            pitch=pitch,
            duration=dur,
            voice=0,
            offset=offset,
        ))
        offset += dur

    return Subject(
        notes=notes,
        key_signature=key_sig,
        time_signature=time_sig,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fugueforge",
        description="FugueForge — Neurosymbolic Fugue Composition Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a fugue")
    gen_parser.add_argument("--subject-file", type=str, help="MIDI/MusicXML file with subject")
    gen_parser.add_argument("--key", type=str, default="c", help="Key signature (default: c minor)")
    gen_parser.add_argument("--time-sig", type=str, default="4/4", help="Time signature")
    gen_parser.add_argument("--voices", type=int, default=3, help="Number of voices (2-4)")
    gen_parser.add_argument("--measures", type=int, default=24, help="Target length in measures")
    gen_parser.add_argument("--candidates", type=int, default=5, help="Number of candidates to generate")
    gen_parser.add_argument("--search", action="store_true", help="Use beam search (section-by-section)")
    gen_parser.add_argument("--output", "-o", type=str, help="Output file path")
    gen_parser.set_defaults(func=cmd_generate)

    # Analyze command
    ana_parser = subparsers.add_parser("analyze", help="Analyze an existing fugue")
    ana_parser.add_argument("input_file", type=str, help="MIDI/MusicXML file to analyze")
    ana_parser.set_defaults(func=cmd_analyze)

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Plan a fugue structure")
    plan_parser.add_argument("--subject-file", type=str, help="MIDI/MusicXML file with subject")
    plan_parser.add_argument("--key", type=str, default="c", help="Key signature")
    plan_parser.add_argument("--time-sig", type=str, default="4/4", help="Time signature")
    plan_parser.add_argument("--voices", type=int, default=3, help="Number of voices")
    plan_parser.add_argument("--measures", type=int, default=24, help="Target measures")
    plan_parser.set_defaults(func=cmd_plan)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate across gold corpus")
    eval_parser.add_argument("--bwv", type=int, nargs="*", help="BWV numbers to evaluate (default: all)")
    eval_parser.add_argument("--candidates", type=int, default=5, help="Candidates per single-pass")
    eval_parser.add_argument("--beam", type=int, default=4, help="Beam search candidates per section")
    eval_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
