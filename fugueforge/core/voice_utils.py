"""
FugueForge Voice Utilities

Voice range adaptation, key transposition, and music21 score conversion.
"""

from __future__ import annotations

from .representation import FugueNote, Subject
from .candidates import GenerationConfig


# ---------------------------------------------------------------------------
# Voice range adaptation
# ---------------------------------------------------------------------------

def _adapt_voice_ranges(
    config: GenerationConfig,
    subject: Subject,
    num_voices: int,
) -> GenerationConfig:
    """
    Adapt voice ranges so the subject fits comfortably in the first voice,
    the answer fits in the second, and all voices have proper separation.
    """
    subj_lo, subj_hi = subject.pitch_range
    subj_range = subj_hi - subj_lo

    # Answer is typically a 5th higher
    answer_lo = subj_lo + 7
    answer_hi = subj_hi + 7

    padding = 6

    new_ranges = dict(config.voice_ranges)

    if num_voices == 3:
        new_ranges[0] = (subj_lo, max(answer_hi + padding, subj_hi + 12))
        new_ranges[1] = (max(36, subj_lo - 7), subj_hi + padding)
        new_ranges[2] = (max(36, subj_lo - 19), subj_hi - 2)
    elif num_voices == 4:
        new_ranges[0] = (answer_lo - padding, answer_hi + padding)
        new_ranges[1] = (subj_lo, subj_hi + padding + 5)
        new_ranges[2] = (max(36, subj_lo - 12), subj_hi)
        new_ranges[3] = (max(36, subj_lo - 24), subj_lo + 5)
    elif num_voices == 2:
        new_ranges[0] = (subj_lo, answer_hi + padding)
        new_ranges[1] = (max(36, subj_lo - 12), subj_hi)
    elif num_voices == 5:
        new_ranges[0] = (answer_lo - padding, answer_hi + padding + 5)
        new_ranges[1] = (subj_lo, subj_hi + padding + 5)
        new_ranges[2] = (max(36, subj_lo - 7), subj_hi + 2)
        new_ranges[3] = (max(36, subj_lo - 19), subj_lo + 5)
        new_ranges[4] = (max(36, subj_lo - 31), max(36, subj_lo - 7))

    new_config = GenerationConfig(
        beam_width=config.beam_width,
        max_candidates_per_step=config.max_candidates_per_step,
        temperature=config.temperature,
        step_resolution=config.step_resolution,
        prefer_stepwise=config.prefer_stepwise,
        max_leap=config.max_leap,
        consonance_weight=config.consonance_weight,
        dissonance_penalty=config.dissonance_penalty,
        parallel_veto=config.parallel_veto,
        crossing_veto=config.crossing_veto,
        leap_resolution_bonus=config.leap_resolution_bonus,
        contrary_motion_bonus=config.contrary_motion_bonus,
        scale_bonus=config.scale_bonus,
        chromatic_penalty=config.chromatic_penalty,
        scale_pcs=config.scale_pcs if config.scale_pcs else key_to_scale_pcs(subject.key_signature),
        voice_ranges=new_ranges,
    )
    return new_config


# ---------------------------------------------------------------------------
# Scale / key helpers
# ---------------------------------------------------------------------------

def key_to_scale_pcs(key_sig: str) -> frozenset[int]:
    """
    Convert a key signature string to a frozenset of diatonic pitch classes.
    For minor keys, includes both natural-minor and harmonic-minor
    (raised 7th / leading tone) so that V→I cadences work naturally.
    """
    key_semitones = {
        "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
        "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
        "B": 11, "b": 11,
        "Bb": 10, "bb": 10, "Eb": 3, "eb": 3, "Ab": 8, "ab": 8,
        "Db": 1, "db": 1, "Gb": 6, "gb": 6,
        "F#": 6, "f#": 6, "C#": 1, "c#": 1,
        "G#": 8, "g#": 8, "D#": 3, "d#": 3,
    }
    root = key_semitones.get(key_sig, 0)
    is_minor = key_sig[0].islower() if key_sig else False

    if is_minor:
        # Natural minor + raised 7th (harmonic minor leading tone)
        #   T S T T S T T  →  0 2 3 5 7 8 10
        # + raised 7th        0 2 3 5 7 8 10 11
        intervals = {0, 2, 3, 5, 7, 8, 10, 11}
    else:
        # Major:  T T S T T T S  →  0 2 4 5 7 9 11
        intervals = {0, 2, 4, 5, 7, 9, 11}

    return frozenset((root + i) % 12 for i in intervals)


# ---------------------------------------------------------------------------
# Key transposition
# ---------------------------------------------------------------------------

def _key_to_transposition(key_area: str, home_key: str) -> int:
    """Convert a key area string to transposition in semitones."""
    key_semitones = {
        "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
        "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
        "B": 11, "b": 11,
        "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "Gb": 6,
        "F#": 6, "f#": 6, "C#": 1, "c#": 1,
    }
    if "→" in key_area:
        key_area = key_area.split("→")[-1].strip()

    home_st = key_semitones.get(home_key, 0)
    target_st = key_semitones.get(key_area, 0)
    return (target_st - home_st) % 12


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def voices_to_score(
    voices: dict[int, list[FugueNote]],
    title: str = "FugueForge Output",
) -> "stream.Score":
    """Convert generated voices to a music21 Score."""
    from music21 import metadata, stream

    score = stream.Score()
    md = metadata.Metadata()
    md.title = title
    md.composer = "FugueForge"
    score.metadata = md

    for v in sorted(voices.keys()):
        part = stream.Part()
        part.id = f"Voice {v}"
        for fn in sorted(voices[v], key=lambda n: n.offset):
            el = fn.to_music21()
            part.insert(fn.offset, el)
        score.append(part)

    return score
