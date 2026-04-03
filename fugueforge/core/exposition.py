"""
FugueForge Exposition Generation

Generates the fugue exposition: places subject/answer entries
in each voice and fills counterpoint between them, capturing
the countersubject from the first counterpoint against the answer.
"""

from __future__ import annotations

from typing import Optional

from .representation import (
    EntryRole,
    ExpositionPlan,
    FugueNote,
    Subject,
    VoiceEntry,
)
from .candidates import GenerationConfig
from .placement import place_subject, place_answer, _get_transposition
from .counterpoint import generate_free_counterpoint, _place_countersubject
from .harmony import ChordLabel


def generate_exposition(
    plan: ExpositionPlan,
    subject: Subject,
    config: Optional[GenerationConfig] = None,
    harmonic_skeleton: Optional[list[ChordLabel]] = None,
) -> tuple[dict[int, list[FugueNote]], list[FugueNote]]:
    """
    Generate the exposition according to the plan.

    1. Place subject/answer for each voice entry
    2. Generate counterpoint for voices already entered
    3. Capture the first counterpoint against the 2nd entry as countersubject
    4. Reuse the countersubject in subsequent entries

    Returns (voices, countersubject).
    """
    config = config or GenerationConfig()
    voices: dict[int, list[FugueNote]] = {
        e.voice: [] for e in plan.entries
    }

    countersubject: list[FugueNote] = []  # captured after 2nd entry

    for entry_idx, entry in enumerate(plan.entries):
        # Place subject or answer
        if entry.role == EntryRole.SUBJECT:
            notes = place_subject(
                subject, entry.voice, entry.start_offset,
                transposition=0 if entry_idx == 0 else _get_transposition(entry, subject),
            )
        else:
            notes = place_answer(subject, entry.voice, entry.start_offset)

        voices[entry.voice].extend(notes)

        # Generate counterpoint for previously entered voices
        for prev_idx, prev_entry in enumerate(plan.entries[:entry_idx]):
            prev_voice = prev_entry.voice
            if prev_voice == entry.voice:
                continue

            # Find where prev voice's last note ends
            prev_notes = voices[prev_voice]
            if prev_notes:
                prev_end = max(n.offset + n.duration for n in prev_notes)
            else:
                prev_end = entry.start_offset

            # Generate counterpoint from prev_end to current entry end
            entry_end = entry.start_offset + subject.duration
            # Don't overlap: start from whichever is later
            cp_start = max(prev_end, entry.start_offset)
            gap = entry_end - cp_start

            if gap > 0:
                last_pitch = None
                if prev_notes:
                    non_rest = [n for n in prev_notes if not n.is_rest]
                    if non_rest:
                        last_pitch = non_rest[-1].pitch

                # For 3rd+ entry: try to use the countersubject
                if entry_idx >= 2 and countersubject and prev_idx == entry_idx - 1:
                    cp = _place_countersubject(
                        countersubject,
                        voice=prev_voice,
                        start_offset=entry.start_offset,
                        existing_voices=voices,
                        config=config,
                    )
                else:
                    cp = generate_free_counterpoint(
                        voice=prev_voice,
                        start_offset=cp_start,
                        duration=gap,
                        existing_voices=voices,
                        start_pitch=last_pitch,
                        config=config,
                        harmonic_skeleton=harmonic_skeleton,
                    )

                # Capture countersubject from the 1st voice's counterpoint
                # against the 2nd entry (this IS the countersubject)
                if entry_idx == 1 and prev_idx == 0 and not countersubject:
                    # Normalize offsets so CS starts at t=0
                    cs_base = cp[0].offset if cp else entry.start_offset
                    countersubject = [
                        FugueNote(
                            pitch=n.pitch,
                            duration=n.duration,
                            voice=0,  # normalize to voice 0
                            offset=n.offset - cs_base,  # relative offset from 0
                            role=EntryRole.COUNTERSUBJECT,
                        )
                        for n in cp
                    ]

                voices[prev_voice].extend(cp)

    return voices, countersubject
