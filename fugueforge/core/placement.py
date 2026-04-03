"""
FugueForge Placement

Subject, answer, and octave-adjustment placement logic.
Handles transposing and positioning thematic material in voices.
"""

from __future__ import annotations

from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    Subject,
    TransformType,
    VoiceEntry,
)
from .analyzer import compute_tonal_answer
from .candidates import GenerationConfig


# ---------------------------------------------------------------------------
# Subject / Answer placement
# ---------------------------------------------------------------------------

def place_subject(
    subject: Subject,
    voice: int,
    start_offset: float,
    transposition: int = 0,
    transform: TransformType = TransformType.NONE,
) -> list[FugueNote]:
    """Place the subject in a voice at a given offset with optional transposition."""
    result: list[FugueNote] = []
    base_offset = subject.notes[0].offset if subject.notes else 0.0

    for fn in subject.notes:
        p = fn.pitch
        if not fn.is_rest:
            if transform == TransformType.INVERSION:
                first_pitch = subject.notes[0].pitch
                p = first_pitch - (p - first_pitch) + transposition
            elif transform == TransformType.RETROGRADE:
                p += transposition
            else:
                p += transposition

        dur = fn.duration
        rel_offset = fn.offset - base_offset
        if transform == TransformType.AUGMENTATION:
            dur *= 2
            rel_offset *= 2
        elif transform == TransformType.DIMINUTION:
            dur *= 0.5
            rel_offset *= 0.5

        result.append(FugueNote(
            pitch=p if not fn.is_rest else -1,
            duration=dur,
            voice=voice,
            offset=start_offset + rel_offset,
            role=EntryRole.SUBJECT if transposition % 12 == 0 else EntryRole.ANSWER,
            transform=transform,
        ))

    if transform == TransformType.RETROGRADE:
        pitches = [n.pitch for n in result]
        pitches.reverse()
        for i, n in enumerate(result):
            result[i] = FugueNote(
                pitch=pitches[i],
                duration=n.duration,
                voice=n.voice,
                offset=n.offset,
                role=n.role,
                transform=n.transform,
            )

    return result


def place_answer(
    subject: Subject,
    voice: int,
    start_offset: float,
) -> list[FugueNote]:
    """Place the tonal/real answer in a voice."""
    answer_notes = compute_tonal_answer(subject, transposition_semitones=7)
    result: list[FugueNote] = []
    base_offset = answer_notes[0].offset if answer_notes else 0.0

    for fn in answer_notes:
        result.append(FugueNote(
            pitch=fn.pitch,
            duration=fn.duration,
            voice=voice,
            offset=start_offset + (fn.offset - base_offset),
            role=EntryRole.ANSWER,
        ))

    return result


# ---------------------------------------------------------------------------
# Octave adjustment for placement
# ---------------------------------------------------------------------------

def _adjust_placement_octave(
    notes: list[FugueNote],
    voice: int,
    existing_voices: dict[int, list[FugueNote]],
    num_voices: int,
    config: GenerationConfig,
) -> list[FugueNote]:
    """
    Check if a subject/answer placement causes voice crossing.
    If so, shift by octave(s) to minimize crossings while staying in range.
    """
    if not notes:
        return notes

    pitches = [n.pitch for n in notes if not n.is_rest]
    if not pitches:
        return notes

    def count_crossings(shift: int) -> int:
        crossings = 0
        for n in notes:
            if n.is_rest:
                continue
            p = n.pitch + shift
            for ov, ov_notes in existing_voices.items():
                if ov == voice:
                    continue
                for on in ov_notes:
                    if on.is_rest:
                        continue
                    if abs(on.offset - n.offset) < 0.01 or (
                        on.offset < n.offset + n.duration
                        and n.offset < on.offset + on.duration
                    ):
                        if voice < ov and p < on.pitch:
                            crossings += 1
                        elif voice > ov and p > on.pitch:
                            crossings += 1
                        break
        return crossings

    lo, hi = config.voice_ranges.get(voice, (36, 84))
    best_shift = 0
    best_crossings = count_crossings(0)

    for shift in [-12, 12, -24, 24]:
        shifted_lo = min(pitches) + shift
        shifted_hi = max(pitches) + shift
        if shifted_lo < lo - 3 or shifted_hi > hi + 3:
            continue
        c = count_crossings(shift)
        if c < best_crossings:
            best_crossings = c
            best_shift = shift

    if best_shift == 0:
        return notes

    return [
        FugueNote(
            pitch=n.pitch + best_shift if not n.is_rest else n.pitch,
            duration=n.duration,
            voice=n.voice,
            offset=n.offset,
            role=n.role,
            transform=n.transform,
        )
        for n in notes
    ]


def _get_transposition(entry: VoiceEntry, subject: Subject) -> int:
    """Determine appropriate transposition for a middle entry."""
    return 0
