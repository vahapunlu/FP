"""
FugueForge Planner

Plans the macrostructure of a fugue given a subject:
- Exposition layout (voice entries, answer type, countersubject strategy)
- Episode plan (derived from subject material, sequence patterns)
- Middle entries and key areas
- Stretto possibilities
- Coda / final cadence
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

from .representation import (
    AnswerType,
    EntryRole,
    ExpositionPlan,
    FugueNote,
    FuguePlan,
    SectionPlan,
    SectionType,
    Subject,
    TransformType,
    VoiceEntry,
)
from .analyzer import compute_tonal_answer, extract_interval_profile


# ---------------------------------------------------------------------------
# Exposition planning
# ---------------------------------------------------------------------------

# Standard voice entry orders for common voice counts
ENTRY_ORDERS: dict[int, list[list[int]]] = {
    2: [[0, 1], [1, 0]],
    3: [
        [0, 1, 2],  # S→A→T (or top→mid→bottom)
        [1, 0, 2],  # A→S→T
        [2, 1, 0],  # T→A→S
        [1, 2, 0],  # A→T→S
    ],
    4: [
        [1, 0, 2, 3],  # A→S→T→B
        [2, 1, 3, 0],  # T→A→B→S
        [0, 1, 2, 3],  # S→A→T→B
        [2, 3, 1, 0],  # T→B→A→S
    ],
}


def plan_exposition(
    subject: Subject,
    num_voices: int = 3,
    entry_order: Optional[list[int]] = None,
    answer_type: Optional[AnswerType] = None,
    codetta_duration: float = 0.0,
) -> ExpositionPlan:
    """
    Create an exposition plan for the given subject.

    The plan determines:
    - Voice entry order and timing
    - Whether the answer is tonal or real
    - Where codettas go (if needed)
    """
    if entry_order is None:
        orders = ENTRY_ORDERS.get(num_voices, [[i for i in range(num_voices)]])
        entry_order = orders[0]

    # Decide answer type based on subject intervals
    if answer_type is None:
        intervals = extract_interval_profile(subject.notes)
        # If subject starts/ends on 5th degree, likely needs tonal answer
        answer_type = _suggest_answer_type(subject)

    subject_dur = subject.duration

    entries: list[VoiceEntry] = []
    codetta_offsets: list[float] = []
    current_offset = 0.0

    for idx, voice in enumerate(entry_order):
        # Alternate subject and answer
        if idx % 2 == 0:
            role = EntryRole.SUBJECT
        else:
            role = EntryRole.ANSWER

        entries.append(VoiceEntry(
            voice=voice,
            role=role,
            start_offset=current_offset,
            subject_ref=subject,
            transform=TransformType.NONE,
        ))

        current_offset += subject_dur

        # Add codetta between entries if specified
        if codetta_duration > 0 and idx < len(entry_order) - 1:
            codetta_offsets.append(current_offset)
            current_offset += codetta_duration

    return ExpositionPlan(
        num_voices=num_voices,
        entries=entries,
        answer_type=answer_type,
        has_countersubject=True,
        codetta_offsets=codetta_offsets,
    )


def _suggest_answer_type(subject: Subject) -> AnswerType:
    """
    Heuristic: if the subject prominently features the 5th scale degree
    early on (large leap to dominant), suggest tonal answer.
    """
    if not subject.notes:
        return AnswerType.REAL

    intervals = extract_interval_profile(subject.notes)
    if not intervals:
        return AnswerType.REAL

    # If first interval is a 5th up or 4th down, tonal answer is typical
    first = intervals[0]
    if first in (7, -5):  # P5 up or P4 down
        return AnswerType.TONAL
    if first in (-7, 5):  # P5 down or P4 up
        return AnswerType.TONAL

    return AnswerType.REAL


# ---------------------------------------------------------------------------
# Episode planning
# ---------------------------------------------------------------------------

@dataclass
class EpisodePlan:
    """Plan for a single episode."""
    start_offset: float
    duration: float
    source_material: str = "subject_head"  # subject_head, subject_tail, countersubject, sequence
    sequence_interval: int = -2            # descending step by default
    sequence_count: int = 3
    target_key: str = ""
    voices_active: list[int] = field(default_factory=list)


def plan_episodes(
    subject: Subject,
    exposition_end: float,
    total_target_duration: float,
    num_voices: int = 3,
    num_episodes: int = 2,
) -> list[EpisodePlan]:
    """
    Plan episodes between middle entries.
    Episodes are derived from subject material using sequences.
    """
    subject_dur = subject.duration
    # Divide available time
    available = total_target_duration - exposition_end
    if available <= 0 or num_episodes == 0:
        return []

    middle_entry_dur = subject_dur * num_voices * 0.8
    episode_dur = (available - middle_entry_dur * num_episodes) / max(num_episodes, 1)
    episode_dur = max(episode_dur, subject_dur)

    episodes: list[EpisodePlan] = []
    offset = exposition_end

    # Derive episode materials from subject
    intervals = extract_interval_profile(subject.notes)
    head_len = max(2, len(intervals) // 3)
    tail_start = max(0, len(intervals) - head_len)

    materials = ["subject_head", "subject_tail", "countersubject", "sequence"]

    for i in range(num_episodes):
        material = materials[i % len(materials)]
        # Alternate sequence direction
        seq_interval = -2 if i % 2 == 0 else 2

        ep = EpisodePlan(
            start_offset=offset,
            duration=episode_dur,
            source_material=material,
            sequence_interval=seq_interval,
            sequence_count=3,
            voices_active=list(range(num_voices)),
        )
        episodes.append(ep)
        offset += episode_dur + middle_entry_dur

    return episodes


# ---------------------------------------------------------------------------
# Stretto analysis
# ---------------------------------------------------------------------------

@dataclass
class StrettoPossibility:
    """A possible stretto entry configuration."""
    overlap_beats: float     # how many beats of overlap
    voices: list[int]
    starting_offsets: list[float]
    works_harmonically: bool = False  # to be validated by rules engine
    tension_score: float = 0.0


def find_stretto_possibilities(
    subject: Subject,
    num_voices: int = 3,
    min_overlap: float = 1.0,
) -> list[StrettoPossibility]:
    """
    Find possible stretto entry points where the subject overlaps
    with itself at different time offsets.

    Tests each possible entry delay and checks if intervals
    create acceptable vertical sonorities.
    """
    subject_dur = subject.duration
    if subject_dur == 0:
        return []

    possibilities: list[StrettoPossibility] = []
    intervals = extract_interval_profile(subject.notes)

    # Try different entry delays (in quarter note steps)
    for delay_q in range(1, int(subject_dur * 2)):
        delay = delay_q * 0.5
        overlap = subject_dur - delay
        if overlap < min_overlap:
            continue

        # For each pair of voices, compute the interval at each beat
        # between the leader (at offset 0) and the follower (at offset delay)
        vertical_ok = _check_stretto_vertical(subject, delay)

        poss = StrettoPossibility(
            overlap_beats=overlap,
            voices=list(range(min(num_voices, 2))),
            starting_offsets=[0.0, delay],
            works_harmonically=vertical_ok,
            tension_score=overlap / subject_dur,  # More overlap = more tension
        )

        possibilities.append(poss)

    # Sort by tension score (more overlap = more dramatic)
    possibilities.sort(key=lambda p: p.tension_score, reverse=True)
    return possibilities


def _check_stretto_vertical(subject: Subject, delay: float) -> bool:
    """
    Quick check: at each overlap point, are the vertical intervals
    between leader and follower mostly consonant?
    """
    leader_pitches = {n.offset: n.pitch for n in subject.notes if not n.is_rest}
    # Follower: same pitches, transposed up a 5th, offset by delay
    follower_pitches = {
        n.offset + delay: n.pitch + 7
        for n in subject.notes if not n.is_rest
    }

    consonant_count = 0
    total = 0

    for offset, l_pitch in leader_pitches.items():
        # Find closest follower note
        for f_offset, f_pitch in follower_pitches.items():
            if abs(f_offset - offset) < 0.25:
                ic = abs(l_pitch - f_pitch) % 12
                if ic in {0, 3, 4, 7, 8, 9}:  # consonances
                    consonant_count += 1
                total += 1
                break

    if total == 0:
        return True
    return consonant_count / total >= 0.6


# ---------------------------------------------------------------------------
# Key area planning
# ---------------------------------------------------------------------------

# Standard key relationships for fugue middle entries
MAJOR_KEY_PLAN = {
    "C":  ["G", "a", "e", "F", "d"],
    "G":  ["D", "e", "b", "C", "a"],
    "D":  ["A", "b", "f#", "G", "e"],
    "F":  ["C", "d", "a", "Bb", "g"],
    "Bb": ["F", "g", "d", "Eb", "c"],
    "Eb": ["Bb", "c", "g", "Ab", "f"],
}

MINOR_KEY_PLAN = {
    "a":  ["C", "e", "d", "G", "F"],
    "d":  ["F", "a", "g", "C", "Bb"],
    "g":  ["Bb", "d", "c", "F", "Eb"],
    "c":  ["Eb", "g", "f", "Bb", "Ab"],
    "e":  ["G", "b", "a", "D", "C"],
    "b":  ["D", "f#", "e", "A", "G"],
}


def plan_key_areas(
    home_key: str,
    num_middle_entries: int = 2,
) -> list[str]:
    """Return a sequence of key areas for middle entries."""
    if home_key[0].isupper():
        plan = MAJOR_KEY_PLAN.get(home_key, ["G", "a"])
    else:
        plan = MINOR_KEY_PLAN.get(home_key, ["C", "e"])

    return plan[:num_middle_entries]


# ---------------------------------------------------------------------------
# Full fugue plan
# ---------------------------------------------------------------------------

def plan_fugue(
    subject: Subject,
    num_voices: int = 3,
    target_measures: int = 30,
    time_signature: str = "4/4",
) -> FuguePlan:
    """
    Generate a complete structural plan for a fugue.

    This is the "architect" that designs the blueprint before
    any notes are actually generated.
    """
    # Parse time signature for beats per measure
    parts = time_signature.split("/")
    beats_per_measure = int(parts[0]) * (4.0 / int(parts[1]))
    total_duration = target_measures * beats_per_measure

    ks = subject.key_signature

    # 1. Plan exposition
    exposition = plan_exposition(subject, num_voices)
    exposition_end = max(e.start_offset + subject.duration for e in exposition.entries)

    # 2. Plan key areas
    key_areas = plan_key_areas(ks, num_middle_entries=2)

    # 3. Plan stretto
    stretto_options = find_stretto_possibilities(subject, num_voices)
    best_stretto = stretto_options[0] if stretto_options else None

    # 4. Build section plan
    sections: list[SectionPlan] = []

    # Exposition
    sections.append(SectionPlan(
        section_type=SectionType.EXPOSITION,
        start_offset=0.0,
        estimated_duration=exposition_end,
        key_area=ks,
        entries=exposition.entries,
    ))

    # Episodes and middle entries
    offset = exposition_end
    for i, key_area in enumerate(key_areas):
        # Episode
        ep_dur = subject.duration * 1.5
        sections.append(SectionPlan(
            section_type=SectionType.EPISODE,
            start_offset=offset,
            estimated_duration=ep_dur,
            key_area=f"{ks}→{key_area}",
            notes=f"Sequence derived from subject, modulating to {key_area}",
        ))
        offset += ep_dur

        # Middle entry
        me_dur = subject.duration * num_voices * 0.7
        me_entries = [
            VoiceEntry(
                voice=v,
                role=EntryRole.SUBJECT,
                start_offset=offset + v * subject.duration * 0.8,
            )
            for v in range(num_voices)
        ]
        sections.append(SectionPlan(
            section_type=SectionType.MIDDLE_ENTRY,
            start_offset=offset,
            estimated_duration=me_dur,
            key_area=key_area,
            entries=me_entries,
        ))
        offset += me_dur

    # Stretto section
    if best_stretto and best_stretto.works_harmonically:
        stretto_dur = subject.duration * 2
        sections.append(SectionPlan(
            section_type=SectionType.STRETTO,
            start_offset=offset,
            estimated_duration=stretto_dur,
            key_area=ks,
            notes=f"Stretto with {best_stretto.overlap_beats:.1f} beats overlap",
        ))
        offset += stretto_dur

    # Coda
    coda_dur = beats_per_measure * 3
    sections.append(SectionPlan(
        section_type=SectionType.CODA,
        start_offset=offset,
        estimated_duration=coda_dur,
        key_area=ks,
        notes="Final cadence, possibly with pedal point",
    ))

    return FuguePlan(
        subject=subject,
        num_voices=num_voices,
        key_signature=ks,
        time_signature=time_signature,
        exposition=exposition,
        sections=sections,
    )


def print_plan(plan: FuguePlan) -> str:
    """Pretty-print a fugue plan."""
    lines: list[str] = []
    lines.append(f"═══ FugueForge Plan ═══")
    lines.append(f"Key: {plan.key_signature} | Voices: {plan.num_voices} | Time: {plan.time_signature}")
    lines.append(f"Subject duration: {plan.subject.duration:.1f} beats")
    lines.append(f"Subject range: {plan.subject.pitch_range}")
    lines.append(f"")

    if plan.exposition:
        lines.append(f"── Exposition ──")
        lines.append(f"Answer type: {plan.exposition.answer_type.name}")
        for e in plan.exposition.entries:
            lines.append(f"  Voice {e.voice}: {e.role.name} at beat {e.start_offset:.1f}")
        lines.append(f"")

    for i, section in enumerate(plan.sections):
        marker = "►" if section.section_type == SectionType.STRETTO else "│"
        lines.append(
            f"{marker} [{section.section_type.name}] "
            f"beat {section.start_offset:.1f} – {section.start_offset + section.estimated_duration:.1f} "
            f"({section.key_area})"
        )
        if section.notes:
            lines.append(f"  {section.notes}")

    return "\n".join(lines)
