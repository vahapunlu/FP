"""
FugueForge Coda Generation

Generates the cadential coda section at the end of a fugue,
including a final subject statement and V→I authentic cadence.
"""

from __future__ import annotations

from typing import Optional

from .representation import EntryRole, FugueNote, Subject
from .candidates import GenerationConfig
from .placement import place_subject, _adjust_placement_octave
from .counterpoint import generate_free_counterpoint


# ---------------------------------------------------------------------------
# Coda / cadence generator
# ---------------------------------------------------------------------------

def generate_coda(
    subject: Subject,
    num_voices: int,
    start_offset: float,
    duration: float,
    existing_voices: dict[int, list[FugueNote]],
    config: Optional[GenerationConfig] = None,
) -> dict[int, list[FugueNote]]:
    """
    Generate a cadential coda section.

    Strategy:
    - Start with free counterpoint leading to a dominant chord
    - End with V → I authentic cadence
    - Final chord uses tonic triad in appropriate voicing
    """
    config = config or GenerationConfig()
    result: dict[int, list[FugueNote]] = {}

    # Determine tonic pitches based on subject key
    key_sig = subject.key_signature
    tonic_pc = _key_name_to_pc(key_sig)
    dominant_pc = (tonic_pc + 7) % 12
    is_minor = key_sig[0].islower()
    third_pc = (tonic_pc + 3) % 12 if is_minor else (tonic_pc + 4) % 12

    # --- Thematic coda: place subject in bass, counterpoint above, then cadence ---
    free_dur = max(0, duration - 4.0)  # save last 4 beats for cadence
    subj_dur = subject.duration
    bass_voice = num_voices - 1

    working_voices = dict(existing_voices)

    # Place a final subject statement in the lowest voice if room allows
    if free_dur >= subj_dur and subj_dur > 0:
        subj_notes = place_subject(
            subject, bass_voice, start_offset, transposition=0,
        )
        subj_notes = _adjust_placement_octave(
            subj_notes, bass_voice, working_voices, num_voices, config,
        )
        result.setdefault(bass_voice, []).extend(subj_notes)
        working_voices.setdefault(bass_voice, []).extend(subj_notes)

        # Fill upper voices with free counterpoint over the subject
        for v in range(num_voices - 1):
            cp = generate_free_counterpoint(
                voice=v,
                start_offset=start_offset,
                duration=free_dur,
                existing_voices=working_voices,
                config=config,
            )
            result.setdefault(v, []).extend(cp)
            working_voices.setdefault(v, []).extend(cp)
    else:
        for v in range(num_voices):
            if free_dur > 0:
                cp = generate_free_counterpoint(
                    voice=v,
                    start_offset=start_offset,
                    duration=free_dur,
                    existing_voices=working_voices,
                    config=config,
                )
                result.setdefault(v, []).extend(cp)
                working_voices.setdefault(v, []).extend(cp)

    # Final cadence: V chord (2 beats) → I chord (2 beats)
    cadence_offset = start_offset + free_dur

    for v in range(num_voices):
        lo, hi = config.voice_ranges.get(v, (48, 72))
        mid = (lo + hi) // 2

        # V chord pitch for this voice — bass MUST have the dominant root
        if v == num_voices - 1:
            v_pitch = _nearest_pitch_class(mid, dominant_pc, lo, hi)
        elif v == 1:
            leading_tone_pc = (tonic_pc - 1) % 12
            v_pitch = _nearest_pitch_class(mid, leading_tone_pc, lo, hi)
        elif v >= 2:
            v_fifth_pc = (dominant_pc + 7) % 12
            v_pitch = _nearest_pitch_class(mid, v_fifth_pc, lo, hi)
        else:
            v_pitch = _nearest_pitch_class(mid, dominant_pc, lo, hi)

        result.setdefault(v, []).append(FugueNote(
            pitch=v_pitch,
            duration=2.0,
            voice=v,
            offset=cadence_offset,
            role=EntryRole.CODETTA,
        ))

        # I chord pitch for this voice
        if v == 0:
            i_pitch = _nearest_pitch_class(mid, third_pc, lo, hi)
        elif v == num_voices - 1:
            i_pitch = _nearest_pitch_class(mid, tonic_pc, lo, hi)
        else:
            fifth_pc = (tonic_pc + 7) % 12
            i_pitch = _nearest_pitch_class(mid, fifth_pc, lo, hi)

        result.setdefault(v, []).append(FugueNote(
            pitch=i_pitch,
            duration=2.0,
            voice=v,
            offset=cadence_offset + 2.0,
            role=EntryRole.CODETTA,
        ))

    return result


# ---------------------------------------------------------------------------
# Pitch / key helpers
# ---------------------------------------------------------------------------

def _key_name_to_pc(key_name: str) -> int:
    """Convert key name to pitch class (0-11)."""
    pc_map = {
        "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
        "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
        "B": 11, "b": 11,
        "Bb": 10, "bb": 10, "Eb": 3, "eb": 3, "Ab": 8, "ab": 8,
        "F#": 6, "f#": 6, "C#": 1, "c#": 1,
    }
    return pc_map.get(key_name, 0)


def _nearest_pitch_class(center: int, pc: int, lo: int, hi: int) -> int:
    """Find the nearest MIDI pitch to center with the given pitch class."""
    best = center
    best_dist = 999
    for p in range(lo, hi + 1):
        if p % 12 == pc:
            dist = abs(p - center)
            if dist < best_dist:
                best_dist = dist
                best = p
    return best
