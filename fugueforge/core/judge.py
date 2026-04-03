"""
FugueForge Judge

Multi-criteria evaluation system for generated fugues.
Scores along four axes:
  1. Theory score  – counterpoint rule compliance
  2. Structure score – subject recurrence, section balance, cadences
  3. Style score – Baroque fugue idiom adherence
  4. Aesthetic score – melodic interest, voice independence, climax quality

Used for:
  - generate → score → re-rank pipeline
  - self-critic loop
  - candidate selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    FuguePlan,
    SectionType,
    Subject,
)
from .rules import CounterpointRuleChecker, Severity, Violation, VOICE_RANGES, check_counterpoint
from .analyzer import (
    detect_cadences_simple,
    detect_sections,
    find_subject_occurrences,
    extract_interval_profile,
)


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------

@dataclass
class JudgeScore:
    """Complete evaluation of a generated fugue."""
    theory: float = 0.0       # 0..100 counterpoint rule compliance
    structure: float = 0.0    # 0..100 fugue structure quality
    style: float = 0.0        # 0..100 Baroque idiom adherence
    aesthetic: float = 0.0    # 0..100 musical interest
    total: float = 0.0        # weighted combination

    violations: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def compute_total(
        self,
        w_theory: float = 0.30,
        w_structure: float = 0.30,
        w_style: float = 0.20,
        w_aesthetic: float = 0.20,
    ) -> float:
        self.total = (
            self.theory * w_theory
            + self.structure * w_structure
            + self.style * w_style
            + self.aesthetic * w_aesthetic
        )
        return self.total


# ---------------------------------------------------------------------------
# Theory scorer
# ---------------------------------------------------------------------------

def score_theory(
    voices: dict[int, list[FugueNote]],
    voice_ranges: Optional[dict[int, tuple[int, int]]] = None,
) -> tuple[float, list[Violation]]:
    """Evaluate counterpoint rule compliance."""
    if voice_ranges is None:
        # Infer ranges from actual note data with padding
        voice_ranges = _infer_voice_ranges(voices)
    checker = CounterpointRuleChecker(voice_ranges=voice_ranges, strict=True)
    violations = checker.check(voices)
    score = checker.score(voices)
    return score, violations


def _infer_voice_ranges(voices: dict[int, list[FugueNote]]) -> dict[int, tuple[int, int]]:
    """Infer voice ranges from actual note data, with comfortable padding."""
    ranges: dict[int, tuple[int, int]] = {}
    for v, notes in voices.items():
        pitches = [n.pitch for n in notes if not n.is_rest]
        if pitches:
            lo = min(pitches) - 3
            hi = max(pitches) + 3
            ranges[v] = (max(0, lo), min(127, hi))
        else:
            ranges[v] = VOICE_RANGES.get(v, (36, 84))
    return ranges


# ---------------------------------------------------------------------------
# Structure scorer
# ---------------------------------------------------------------------------

def score_structure(
    voices: dict[int, list[FugueNote]],
    subject: Subject,
    plan: Optional[FuguePlan] = None,
) -> tuple[float, list[str]]:
    """
    Evaluate structural quality:
    - Subject recurrence frequency and placement
    - Section balance (exposition, episodes, stretto)
    - Cadence quality and placement
    - Key area variety
    """
    notes_list: list[str] = []
    score = 100.0

    # 1. Subject occurrences
    occurrences = find_subject_occurrences(subject, voices)
    num_voices = len(voices)

    if len(occurrences) < num_voices:
        penalty = (num_voices - len(occurrences)) * 10
        score -= penalty
        notes_list.append(f"Too few subject entries: {len(occurrences)} (expected >= {num_voices})")

    # Check that subject appears in all voices
    voices_with_subject = {o.voice for o in occurrences}
    missing = set(voices.keys()) - voices_with_subject
    if missing:
        score -= len(missing) * 8
        notes_list.append(f"Subject missing in voices: {missing}")

    # 2. Section detection and balance
    total_dur = max(
        (n.offset + n.duration for v_notes in voices.values() for n in v_notes),
        default=0.0,
    )
    sections = detect_sections(subject, voices, total_dur)

    has_exposition = any(s.section_type == SectionType.EXPOSITION for s in sections)
    has_episode = any(s.section_type == SectionType.EPISODE for s in sections)
    has_stretto = any(s.section_type == SectionType.STRETTO for s in sections)

    if not has_exposition:
        score -= 20
        notes_list.append("No clear exposition detected")
    if not has_episode and total_dur > subject.duration * 6:
        score -= 10
        notes_list.append("No episodes detected in a fugue that should have them")
    if has_stretto:
        score += 5  # bonus for stretto
        notes_list.append("Stretto detected (bonus)")

    # 3. Cadences
    cadences = detect_cadences_simple(voices, subject.key_signature)
    if not cadences and total_dur > subject.duration * 4:
        score -= 10
        notes_list.append("No cadences detected")
    elif cadences:
        # Check final cadence strength
        last_cad = cadences[-1]
        if last_cad.type != "PAC":
            score -= 5
            notes_list.append(f"Final cadence is {last_cad.type}, not PAC")

    # 4. Countersubject detection: bonus for recurring CS material
    cs_voices = 0
    for v, v_notes in voices.items():
        cs_count = sum(1 for n in v_notes if n.role == EntryRole.COUNTERSUBJECT)
        if cs_count > 0:
            cs_voices += 1
    if cs_voices >= 2:
        score += 5
        notes_list.append(f"Countersubject present in {cs_voices} voices")

    return max(0.0, min(100.0, score)), notes_list


# ---------------------------------------------------------------------------
# Style scorer
# ---------------------------------------------------------------------------

def score_style(
    voices: dict[int, list[FugueNote]],
    subject: Subject,
) -> tuple[float, list[str]]:
    """
    Evaluate Baroque fugue style adherence:
    - Voice independence (not moving in parallel too much)
    - Rhythmic variety
    - Appropriate voice ranges
    - Melodic contour quality
    """
    notes_list: list[str] = []
    score = 100.0

    # 1. Voice independence: measure parallel motion ratio
    parallel_ratio = _measure_parallel_motion(voices)
    if parallel_ratio > 0.5:
        penalty = (parallel_ratio - 0.5) * 40
        score -= penalty
        notes_list.append(f"Too much parallel motion: {parallel_ratio:.0%}")

    # 2. Rhythmic variety per voice
    for v, v_notes in voices.items():
        durations = [n.duration for n in v_notes if not n.is_rest]
        if durations:
            unique_durs = len(set(durations))
            if unique_durs == 1 and len(durations) > 4:
                score -= 5
                notes_list.append(f"Voice {v}: monotonous rhythm")

    # 3. Melodic contour: check for too many repeated notes
    for v, v_notes in voices.items():
        pitches = [n.pitch for n in v_notes if not n.is_rest]
        if len(pitches) > 4:
            repeats = sum(1 for a, b in zip(pitches, pitches[1:]) if a == b)
            repeat_ratio = repeats / len(pitches)
            if repeat_ratio > 0.4:
                score -= 5
                notes_list.append(f"Voice {v}: too many repeated notes ({repeat_ratio:.0%})")

    # 4. Leap resolution: large leaps should resolve by step in opposite direction
    #    Rate-based: penalize by the RATIO of unresolved leaps, not absolute count
    total_leaps = 0
    unresolved_leaps = 0
    for v, v_notes in voices.items():
        pitches = [n.pitch for n in v_notes if not n.is_rest]
        for i in range(len(pitches) - 2):
            leap = pitches[i + 1] - pitches[i]
            if abs(leap) >= 5:  # leap of a 4th or more
                total_leaps += 1
                resolution = pitches[i + 2] - pitches[i + 1]
                # Should resolve by step in opposite direction
                if abs(resolution) > 2 or (leap > 0 and resolution > 0) or (leap < 0 and resolution < 0):
                    unresolved_leaps += 1

    if total_leaps > 0:
        unresolved_ratio = unresolved_leaps / total_leaps
        # Penalize proportionally: 50% unresolved = -15, 100% unresolved = -30
        score -= unresolved_ratio * 30
        if unresolved_ratio > 0.5:
            notes_list.append(f"Poor leap resolution: {unresolved_ratio:.0%} unresolved")

    return max(0.0, min(100.0, score)), notes_list


def _measure_parallel_motion(voices: dict[int, list[FugueNote]]) -> float:
    """Measure what fraction of beat-pairs have all voices in parallel motion."""
    voice_ids = sorted(voices.keys())
    if len(voice_ids) < 2:
        return 0.0

    # Build pitch sequences per voice (aligned by offset)
    all_offsets = sorted(set(
        n.offset for v_notes in voices.values() for n in v_notes
    ))

    parallel_count = 0
    total_pairs = 0

    for idx in range(len(all_offsets) - 1):
        t1, t2 = all_offsets[idx], all_offsets[idx + 1]

        motions = []
        for v in voice_ids:
            p1 = _pitch_at(voices[v], t1)
            p2 = _pitch_at(voices[v], t2)
            if p1 is not None and p2 is not None:
                motions.append(p2 - p1)

        if len(motions) >= 2:
            total_pairs += 1
            # All voices same direction and similar magnitude
            if all(m > 0 for m in motions) or all(m < 0 for m in motions):
                if max(motions) - min(motions) <= 2:
                    parallel_count += 1

    return parallel_count / total_pairs if total_pairs > 0 else 0.0


def _pitch_at(notes: list[FugueNote], offset: float) -> Optional[int]:
    """Find the pitch sounding at a given offset."""
    for n in notes:
        if n.offset <= offset < n.offset + n.duration and not n.is_rest:
            return n.pitch
    return None


# ---------------------------------------------------------------------------
# Aesthetic scorer
# ---------------------------------------------------------------------------

def score_aesthetic(
    voices: dict[int, list[FugueNote]],
    subject: Subject,
) -> tuple[float, list[str]]:
    """
    Evaluate overall musical interest:
    - Melodic diversity
    - Climax placement (high point in the later section)
    - Dynamic contour (tension builds and releases)
    - Voice interaction quality
    - Thematic density (use of subject/CS material vs free CP)
    - Heightening activity toward the end (Burrows/Lester principle)
    """
    notes_list: list[str] = []
    score = 70.0  # Base score

    # 1. Pitch range usage
    all_pitches = [n.pitch for v_notes in voices.values() for n in v_notes if not n.is_rest]
    if all_pitches:
        range_used = max(all_pitches) - min(all_pitches)
        if range_used >= 24:  # 2 octaves
            score += 10
            notes_list.append("Good pitch range coverage")
        elif range_used < 12:
            score -= 5
            notes_list.append("Very narrow pitch range")

    # 2. Climax placement: highest note should be in the latter half
    total_dur = max(
        (n.offset + n.duration for v_notes in voices.values() for n in v_notes),
        default=0.0,
    )
    if all_pitches and total_dur > 0:
        highest_notes = [
            n for v_notes in voices.values() for n in v_notes
            if not n.is_rest and n.pitch == max(all_pitches)
        ]
        if highest_notes:
            climax_offset = highest_notes[0].offset
            climax_position = climax_offset / total_dur
            if 0.5 <= climax_position <= 0.85:
                score += 10
                notes_list.append("Well-placed climax")
            elif climax_position < 0.3:
                score -= 5
                notes_list.append("Climax too early")

    # 3. Interval variety in subject
    subject_intervals = extract_interval_profile(subject.notes)
    if subject_intervals:
        unique_intervals = len(set(abs(i) for i in subject_intervals))
        if unique_intervals >= 4:
            score += 5
            notes_list.append("Subject has good interval variety")
        elif unique_intervals <= 2:
            score -= 5
            notes_list.append("Subject intervals too repetitive")

    # 4. Thematic density: ratio of thematic notes (S/CS/episode) vs free CP
    all_notes = [n for v_notes in voices.values() for n in v_notes if not n.is_rest]
    if all_notes:
        from .representation import EntryRole
        thematic_roles = {
            EntryRole.SUBJECT, EntryRole.ANSWER,
            EntryRole.COUNTERSUBJECT, EntryRole.EPISODE_MATERIAL,
        }
        thematic_count = sum(1 for n in all_notes if n.role in thematic_roles)
        thematic_ratio = thematic_count / len(all_notes)
        if thematic_ratio >= 0.6:
            score += 5
            notes_list.append(f"High thematic density ({thematic_ratio:.0%})")
        elif thematic_ratio < 0.3:
            score -= 3
            notes_list.append(f"Low thematic density ({thematic_ratio:.0%})")

    # 5. Heightening activity: latter half should have more notes per beat
    if total_dur > 0 and all_notes:
        mid = total_dur / 2
        first_half = [n for n in all_notes if n.offset < mid]
        second_half = [n for n in all_notes if n.offset >= mid]
        if first_half and second_half and mid > 0:
            density_first = len(first_half) / mid
            density_second = len(second_half) / max(0.1, total_dur - mid)
            if density_second >= density_first * 1.1:
                score += 5
                notes_list.append("Good heightening of activity")

    # 6. Use of subject inversion / augmentation (contrapuntal sophistication)
    from .representation import TransformType
    transforms_used = {n.transform for n in all_notes if hasattr(n, 'transform')}
    advanced_transforms = transforms_used & {
        TransformType.INVERSION, TransformType.AUGMENTATION,
        TransformType.DIMINUTION, TransformType.RETROGRADE,
    }
    if advanced_transforms:
        bonus = min(5, len(advanced_transforms) * 3)
        score += bonus
        names = [t.name.lower() for t in advanced_transforms]
        notes_list.append(f"Uses {', '.join(names)}")

    return max(0.0, min(100.0, score)), notes_list


# ---------------------------------------------------------------------------
# Full judge
# ---------------------------------------------------------------------------

class FugueJudge:
    """Complete fugue evaluation ensemble."""

    def __init__(
        self,
        w_theory: float = 0.30,
        w_structure: float = 0.30,
        w_style: float = 0.20,
        w_aesthetic: float = 0.20,
    ):
        self.w_theory = w_theory
        self.w_structure = w_structure
        self.w_style = w_style
        self.w_aesthetic = w_aesthetic

    def evaluate(
        self,
        voices: dict[int, list[FugueNote]],
        subject: Subject,
        plan: Optional[FuguePlan] = None,
    ) -> JudgeScore:
        """Run all evaluations and return composite score."""
        # Theory
        theory_score, violations = score_theory(voices)

        # Structure
        structure_score, struct_notes = score_structure(voices, subject, plan)

        # Style
        style_score, style_notes = score_style(voices, subject)

        # Aesthetic
        aesthetic_score, aesthetic_notes = score_aesthetic(voices, subject)

        result = JudgeScore(
            theory=theory_score,
            structure=structure_score,
            style=style_score,
            aesthetic=aesthetic_score,
            violations=violations,
            notes=struct_notes + style_notes + aesthetic_notes,
        )
        result.compute_total(
            self.w_theory, self.w_structure, self.w_style, self.w_aesthetic,
        )
        return result

    def rank_candidates(
        self,
        candidates: list[dict[int, list[FugueNote]]],
        subject: Subject,
        plan: Optional[FuguePlan] = None,
    ) -> list[tuple[float, int, JudgeScore]]:
        """
        Evaluate multiple candidates and return them ranked by total score.
        Returns [(score, index, JudgeScore), ...] sorted descending.
        """
        results: list[tuple[float, int, JudgeScore]] = []
        for idx, candidate in enumerate(candidates):
            js = self.evaluate(candidate, subject, plan)
            results.append((js.total, idx, js))

        results.sort(reverse=True, key=lambda x: x[0])
        return results


def print_judge_score(js: JudgeScore) -> str:
    """Pretty-print a judge score."""
    lines = [
        f"═══ FugueForge Judge ═══",
        f"Theory:    {js.theory:5.1f}/100",
        f"Structure: {js.structure:5.1f}/100",
        f"Style:     {js.style:5.1f}/100",
        f"Aesthetic:  {js.aesthetic:5.1f}/100",
        f"───────────────────────",
        f"TOTAL:     {js.total:5.1f}/100",
        f"",
    ]

    if js.violations:
        err_count = sum(1 for v in js.violations if v.severity == Severity.ERROR)
        warn_count = sum(1 for v in js.violations if v.severity == Severity.WARNING)
        lines.append(f"Violations: {err_count} errors, {warn_count} warnings")

    if js.notes:
        lines.append(f"Notes:")
        for n in js.notes:
            lines.append(f"  • {n}")

    return "\n".join(lines)
