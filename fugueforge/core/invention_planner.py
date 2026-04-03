"""
FugueForge Invention Planner

Plans the macrostructure of a Two-Part Invention:
  Exposition → Episode₁ → Middle Entry (dominant) → Episode₂ →
  Middle Entry (relative) → Episode₃ → Recapitulation → Cadence

Inventions are simpler than fugues:
  - Always 2 voices
  - Theme answered at octave (usually) or 5th
  - Episodes derived from theme fragments (sequences)
  - Clear ternary structure (A-B-A')
  - Final cadence, no stretto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .representation import (
    AnswerType,
    EntryRole,
    ExpositionPlan,
    FuguePlan,
    SectionPlan,
    SectionType,
    Subject,
    TransformType,
    VoiceEntry,
)
from .analyzer import compute_tonal_answer


# ---------------------------------------------------------------------------
# Key relationships for inventions
# ---------------------------------------------------------------------------

MAJOR_RELATED = {
    "C": ["G", "a"],   "G": ["D", "e"],   "D": ["A", "b"],
    "F": ["C", "d"],   "Bb": ["F", "g"],  "Eb": ["Bb", "c"],
    "A": ["E", "c#"],  "E": ["B", "c#"],
}
MINOR_RELATED = {
    "a": ["C", "e"],   "d": ["F", "a"],   "g": ["Bb", "d"],
    "c": ["Eb", "g"],  "e": ["G", "b"],   "b": ["D", "f#"],
    "f": ["Ab", "c"],  "c#": ["E", "g#"],
}


def _get_related_keys(home_key: str) -> list[str]:
    """Return dominant + relative key for the home key."""
    if home_key[0].isupper():
        return MAJOR_RELATED.get(home_key, ["G", "a"])
    else:
        return MINOR_RELATED.get(home_key, ["C", "e"])


# ---------------------------------------------------------------------------
# Invention plan
# ---------------------------------------------------------------------------

def plan_invention(
    subject: Subject,
    target_measures: int = 22,
    time_signature: str = "4/4",
    answer_at_fifth: bool = False,
) -> FuguePlan:
    """
    Create a structural plan for a Two-Part Invention.

    Returns a FuguePlan (reusing the same data structure) with 2 voices
    and a simpler section layout:

      1. Exposition: V0 states theme, V1 answers at octave (or 5th)
      2. Episode 1: Sequential development → dominant key
      3. Middle Entry 1: Theme in dominant key
      4. Episode 2: Sequential → relative key
      5. Middle Entry 2: Theme in relative key (optional, only if long enough)
      6. Episode 3: Return sequence → home key
      7. Recapitulation: Theme in home key (inverted voice roles)
      8. Cadence: V→I authentic cadence
    """
    # Parse time signature
    ts_parts = time_signature.split("/")
    beats_per_measure = int(ts_parts[0]) * (4.0 / int(ts_parts[1]))
    total_duration = target_measures * beats_per_measure

    ks = subject.key_signature
    related_keys = _get_related_keys(ks)
    dom_key = related_keys[0]  # dominant or relative major
    rel_key = related_keys[1] if len(related_keys) > 1 else dom_key

    theme_dur = subject.duration
    num_voices = 2

    # --- Build exposition ---
    exposition_entries = [
        VoiceEntry(
            voice=0, role=EntryRole.SUBJECT,
            start_offset=0.0, transform=TransformType.NONE,
        ),
        VoiceEntry(
            voice=1, role=EntryRole.ANSWER,
            start_offset=theme_dur, transform=TransformType.NONE,
        ),
    ]
    exposition_dur = theme_dur * 2
    exposition = ExpositionPlan(
        num_voices=num_voices,
        entries=exposition_entries,
        answer_type=AnswerType.REAL,
        has_countersubject=True,
    )

    # --- Build sections ---
    sections: list[SectionPlan] = []
    offset = 0.0

    # 1. Exposition
    sections.append(SectionPlan(
        section_type=SectionType.EXPOSITION,
        start_offset=0.0,
        estimated_duration=exposition_dur,
        key_area=ks,
        entries=exposition_entries,
    ))
    offset = exposition_dur

    # Episode and entry durations scaled to target length
    ep_dur = theme_dur * 1.5
    me_dur = theme_dur * 1.5  # theme + brief CP overlap

    # Decide if we include a second middle entry (only for longer pieces)
    include_second_me = total_duration >= theme_dur * 10

    # 2. Episode 1: modulate to dominant
    sections.append(SectionPlan(
        section_type=SectionType.EPISODE,
        start_offset=offset,
        estimated_duration=ep_dur,
        key_area=f"{ks}→{dom_key}",
        notes="Sequential development toward dominant",
    ))
    offset += ep_dur

    # 3. Middle Entry 1: theme in dominant
    me1_entries = [
        VoiceEntry(
            voice=1, role=EntryRole.SUBJECT,
            start_offset=offset, transform=TransformType.NONE,
        ),
    ]
    sections.append(SectionPlan(
        section_type=SectionType.MIDDLE_ENTRY,
        start_offset=offset,
        estimated_duration=me_dur,
        key_area=dom_key,
        entries=me1_entries,
        notes="Theme in dominant key",
    ))
    offset += me_dur

    if include_second_me:
        # 4. Episode 2: modulate to relative key
        sections.append(SectionPlan(
            section_type=SectionType.EPISODE,
            start_offset=offset,
            estimated_duration=ep_dur,
            key_area=f"{dom_key}→{rel_key}",
            notes="Sequential toward relative key",
        ))
        offset += ep_dur

        # 5. Middle Entry 2: theme in relative key
        me2_entries = [
            VoiceEntry(
                voice=0, role=EntryRole.SUBJECT,
                start_offset=offset, transform=TransformType.NONE,
            ),
        ]
        sections.append(SectionPlan(
            section_type=SectionType.MIDDLE_ENTRY,
            start_offset=offset,
            estimated_duration=me_dur,
            key_area=rel_key,
            entries=me2_entries,
            notes="Theme in relative key",
        ))
        offset += me_dur

    # 6. Episode 3 (or 2): return to home key
    sections.append(SectionPlan(
        section_type=SectionType.EPISODE,
        start_offset=offset,
        estimated_duration=ep_dur,
        key_area=f"{rel_key if include_second_me else dom_key}→{ks}",
        notes="Return sequence toward home key",
    ))
    offset += ep_dur

    # 7. Recapitulation: theme in home key (voices swapped)
    recap_entries = [
        VoiceEntry(
            voice=1, role=EntryRole.SUBJECT,
            start_offset=offset, transform=TransformType.NONE,
        ),
    ]
    sections.append(SectionPlan(
        section_type=SectionType.MIDDLE_ENTRY,
        start_offset=offset,
        estimated_duration=me_dur,
        key_area=ks,
        entries=recap_entries,
        notes="Recapitulation: theme returns in tonic",
    ))
    offset += me_dur

    # 8. Cadence (short coda)
    cad_dur = beats_per_measure * 2  # 2 measures
    sections.append(SectionPlan(
        section_type=SectionType.CODA,
        start_offset=offset,
        estimated_duration=cad_dur,
        key_area=ks,
        notes="Final V→I cadence",
    ))

    return FuguePlan(
        subject=subject,
        num_voices=num_voices,
        key_signature=ks,
        time_signature=time_signature,
        exposition=exposition,
        sections=sections,
    )
