"""
FugueForge Harmonic Skeleton

Provides functional harmony awareness for the generation pipeline.
Generates a chord progression template (harmonic rhythm) that guides
note selection — the missing "backbone" that makes music sound tonal
rather than a random walk through the scale.

Baroque fugue harmony principles (per Burrows/Lester):
  - Strong beats carry clear chord identity (I, IV, V, vi, etc.)
  - Harmonic rhythm is typically 1-2 chords per bar
  - Progressions follow circle-of-fifths logic
  - Episodes modulate through related keys
  - Cadences punctuate phrase endings (V→I, V→vi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Chord / harmony data types
# ---------------------------------------------------------------------------

@dataclass
class ChordLabel:
    """A chord at a specific point in time."""
    offset: float            # absolute offset in quarter notes
    duration: float          # how long this chord lasts
    root_pc: int             # pitch class of the chord root (0-11)
    chord_pcs: frozenset[int]  # all pitch classes in the chord
    function: str            # Roman numeral: "I", "IV", "V", "vi", etc.
    is_cadential: bool = False  # part of a cadence


# ---------------------------------------------------------------------------
# Scale-degree to pitch-class mappings
# ---------------------------------------------------------------------------

# Major scale degrees → semitone offset from tonic
_MAJOR_DEGREES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

# Minor scale degrees (harmonic minor for V chord)
_MINOR_DEGREES = {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 11}  # raised 7th


def _triad(root_pc: int, quality: str) -> frozenset[int]:
    """Build a triad from root pitch class and quality."""
    if quality == "major":
        return frozenset({root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12})
    elif quality == "minor":
        return frozenset({root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12})
    elif quality == "diminished":
        return frozenset({root_pc, (root_pc + 3) % 12, (root_pc + 6) % 12})
    else:  # augmented or fallback
        return frozenset({root_pc, (root_pc + 4) % 12, (root_pc + 8) % 12})


def _build_diatonic_chords(tonic: int, is_minor: bool) -> dict[str, tuple[int, frozenset[int]]]:
    """
    Build all diatonic triads for a key.
    Returns {roman_numeral: (root_pc, chord_pcs)}.
    """
    deg = _MINOR_DEGREES if is_minor else _MAJOR_DEGREES

    if is_minor:
        # Natural minor + harmonic minor V
        return {
            "i":   ((tonic + deg[1]) % 12, _triad((tonic + deg[1]) % 12, "minor")),
            "ii°": ((tonic + deg[2]) % 12, _triad((tonic + deg[2]) % 12, "diminished")),
            "III": ((tonic + deg[3]) % 12, _triad((tonic + deg[3]) % 12, "major")),
            "iv":  ((tonic + deg[4]) % 12, _triad((tonic + deg[4]) % 12, "minor")),
            "V":   ((tonic + deg[5]) % 12, _triad((tonic + deg[5]) % 12, "major")),  # harmonic minor
            "VI":  ((tonic + deg[6]) % 12, _triad((tonic + deg[6]) % 12, "major")),
            "vii°":((tonic + deg[7]) % 12, _triad((tonic + deg[7]) % 12, "diminished")),
        }
    else:
        return {
            "I":   ((tonic + deg[1]) % 12, _triad((tonic + deg[1]) % 12, "major")),
            "ii":  ((tonic + deg[2]) % 12, _triad((tonic + deg[2]) % 12, "minor")),
            "iii": ((tonic + deg[3]) % 12, _triad((tonic + deg[3]) % 12, "minor")),
            "IV":  ((tonic + deg[4]) % 12, _triad((tonic + deg[4]) % 12, "major")),
            "V":   ((tonic + deg[5]) % 12, _triad((tonic + deg[5]) % 12, "major")),
            "vi":  ((tonic + deg[6]) % 12, _triad((tonic + deg[6]) % 12, "minor")),
            "vii°":((tonic + deg[7]) % 12, _triad((tonic + deg[7]) % 12, "diminished")),
        }


# ---------------------------------------------------------------------------
# Progression templates — circle-of-fifths based
# ---------------------------------------------------------------------------

# Common Baroque fugue progressions (each is a list of Roman numerals)
# These cycle to fill any section length.

_MAJOR_PROGRESSIONS = [
    # Strong: circle of fifths
    ["I", "IV", "vii°", "iii", "vi", "ii", "V", "I"],
    # Medium: standard
    ["I", "vi", "IV", "V", "I", "ii", "V", "I"],
    # Cadential
    ["I", "IV", "I", "V", "vi", "IV", "V", "I"],
]

_MINOR_PROGRESSIONS = [
    # Circle of fifths in minor
    ["i", "iv", "vii°", "III", "VI", "ii°", "V", "i"],
    # Standard minor
    ["i", "VI", "iv", "V", "i", "ii°", "V", "i"],
    # With III (relative major)
    ["i", "iv", "i", "V", "III", "iv", "V", "i"],
]

# Episode progressions: sequences that modulate
_EPISODE_MAJOR = ["I", "V", "ii", "vi", "iii", "vii°", "IV", "I"]
_EPISODE_MINOR = ["i", "V", "ii°", "VI", "III", "vii°", "iv", "i"]


# ---------------------------------------------------------------------------
# Harmonic skeleton generator
# ---------------------------------------------------------------------------

def generate_harmonic_skeleton(
    tonic_pc: int,
    is_minor: bool,
    start_offset: float,
    duration: float,
    beats_per_chord: float = 2.0,
    section_type: str = "normal",  # "normal", "episode", "cadence"
    progression_idx: int = 0,
) -> list[ChordLabel]:
    """
    Generate a sequence of ChordLabels for a section.

    Args:
        tonic_pc: pitch class of the tonic (0=C, 2=D, etc.)
        is_minor: True for minor keys
        start_offset: where this section starts
        duration: how long the section is
        beats_per_chord: harmonic rhythm (default: chord changes every 2 beats)
        section_type: affects which progression template to use
        progression_idx: which template variation to pick

    Returns:
        List of ChordLabels covering the section duration.
    """
    chords_dict = _build_diatonic_chords(tonic_pc, is_minor)

    # Pick progression template
    if section_type == "episode":
        prog = _EPISODE_MINOR if is_minor else _EPISODE_MAJOR
    elif section_type == "cadence":
        # Short cadential: IV/iv → V → I/i
        tonic_rn = "i" if is_minor else "I"
        sub_rn = "iv" if is_minor else "IV"
        prog = [sub_rn, "V", tonic_rn]
        beats_per_chord = max(1.0, duration / 3.0)
    else:
        progs = _MINOR_PROGRESSIONS if is_minor else _MAJOR_PROGRESSIONS
        prog = progs[progression_idx % len(progs)]

    # Generate chord labels
    result: list[ChordLabel] = []
    offset = start_offset
    end = start_offset + duration
    idx = 0

    while offset < end - 0.01:
        rn = prog[idx % len(prog)]
        root_pc, chord_pcs = chords_dict.get(rn, chords_dict.get(
            "i" if is_minor else "I", (tonic_pc, _triad(tonic_pc, "major"))
        ))

        remaining = end - offset
        dur = min(beats_per_chord, remaining)

        # Mark last chord as cadential if it's a V or I at the end
        is_cad = (offset + dur >= end - 0.01) and rn in ("V", "I", "i")

        result.append(ChordLabel(
            offset=offset,
            duration=dur,
            root_pc=root_pc,
            chord_pcs=chord_pcs,
            function=rn,
            is_cadential=is_cad,
        ))

        offset += dur
        idx += 1

    return result


def get_chord_at(skeleton: list[ChordLabel], offset: float) -> Optional[ChordLabel]:
    """Find which chord is active at a given offset."""
    for chord in skeleton:
        if chord.offset <= offset < chord.offset + chord.duration:
            return chord
    # If past the end, return the last chord
    return skeleton[-1] if skeleton else None


# ---------------------------------------------------------------------------
# Key name to pitch class (shared utility)
# ---------------------------------------------------------------------------

_KEY_PC_MAP = {
    "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
    "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
    "B": 11, "b": 11,
    "Bb": 10, "bb": 10, "Eb": 3, "eb": 3, "Ab": 8, "ab": 8,
    "Db": 1, "db": 1, "Gb": 6, "gb": 6,
    "F#": 6, "f#": 6, "C#": 1, "c#": 1,
    "G#": 8, "g#": 8, "D#": 3, "d#": 3,
}


def key_name_to_pc(name: str) -> int:
    return _KEY_PC_MAP.get(name, 0)


def key_is_minor(name: str) -> bool:
    return name[0].islower() if name else False
