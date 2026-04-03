"""
Gold standard annotations for Bach Two-Part Inventions (BWV 772-786).

Each entry encodes the invention theme (subject) as MIDI pitches + durations,
along with key and structural annotations.

All 15 Two-Part Inventions from the Aufrichtige Anleitung (1723).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fugueforge.core.representation import (
    AnswerType,
    FugueNote,
    Subject,
)


@dataclass
class GoldInvention:
    """Gold standard annotation for a Bach Two-Part Invention."""
    bwv: int
    number: int           # 1-15
    title: str
    key_signature: str
    time_signature: str
    # Theme as (midi_pitch, duration_in_quarter_notes) pairs
    theme_pitches: list[tuple[int, float]]
    # Does the answer enter at the octave (typical) or at the 5th?
    answer_at_fifth: bool = False
    # Approximate length in measures
    approx_measures: int = 22
    # Notes
    notes: str = ""

    def to_subject(self) -> Subject:
        """Convert to a FugueForge Subject object."""
        fn_notes: list[FugueNote] = []
        offset = 0.0
        for pitch, dur in self.theme_pitches:
            if pitch < 0:
                fn_notes.append(FugueNote(
                    pitch=0, duration=dur, voice=0,
                    offset=offset, is_rest=True,
                ))
            else:
                fn_notes.append(FugueNote(
                    pitch=pitch, duration=dur, voice=0, offset=offset,
                ))
            offset += dur
        return Subject(
            notes=fn_notes,
            key_signature=self.key_signature,
            time_signature=self.time_signature,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bach Two-Part Inventions BWV 772-786
# ═══════════════════════════════════════════════════════════════════════════

GOLD_INVENTIONS: dict[int, GoldInvention] = {}


# -------------------------------------------------------------------
# BWV 772 — Invention No.1 in C major
# Theme: C-D-E-F-D-E-C-G (scalar rising, then leap)
# -------------------------------------------------------------------
GOLD_INVENTIONS[772] = GoldInvention(
    bwv=772,
    number=1,
    title="Invention No. 1 in C major",
    key_signature="C",
    time_signature="4/4",
    theme_pitches=[
        (72, 0.25),  # C5   ─┐
        (74, 0.25),  # D5    │ rising scale
        (76, 0.25),  # E5    │
        (72, 0.25),  # C5   ─┘
        (77, 0.25),  # F5   ─┐
        (74, 0.25),  # D5    │ descending turn
        (76, 0.25),  # E5    │
        (72, 0.25),  # C5   ─┘
    ],
    answer_at_fifth=False,
    approx_measures=22,
    notes="The most famous invention. Simple rising scale motive, perpetual motion character.",
)

# -------------------------------------------------------------------
# BWV 773 — Invention No.2 in C minor
# Theme: C-D-Eb-F-G-Eb-F-D-Eb-C-D-B-C
# -------------------------------------------------------------------
GOLD_INVENTIONS[773] = GoldInvention(
    bwv=773,
    number=2,
    title="Invention No. 2 in C minor",
    key_signature="c",
    time_signature="4/4",
    theme_pitches=[
        (60, 0.25),  # C4
        (62, 0.25),  # D4
        (63, 0.25),  # Eb4
        (65, 0.25),  # F4
        (67, 0.25),  # G4
        (63, 0.25),  # Eb4
        (65, 0.25),  # F4
        (62, 0.25),  # D4
        (63, 0.25),  # Eb4
        (60, 0.25),  # C4
        (62, 0.25),  # D4
        (59, 0.25),  # B3
        (60, 0.5),   # C4 (longer final note)
    ],
    answer_at_fifth=False,
    approx_measures=27,
    notes="Scalar theme in 16ths. Imitation at octave. Flowing continuous motion.",
)

# -------------------------------------------------------------------
# BWV 774 — Invention No.3 in D major
# Theme: D-F#-A ascending triad arpeggio, then stepwise descent
# -------------------------------------------------------------------
GOLD_INVENTIONS[774] = GoldInvention(
    bwv=774,
    number=3,
    title="Invention No. 3 in D major",
    key_signature="D",
    time_signature="3/8",
    theme_pitches=[
        (62, 0.25),  # D4
        (66, 0.25),  # F#4
        (69, 0.25),  # A4
        (74, 0.25),  # D5
        (73, 0.25),  # C#5
        (74, 0.25),  # D5
        (69, 0.25),  # A4
        (71, 0.25),  # B4
        (73, 0.25),  # C#5
        (66, 0.25),  # F#4
        (69, 0.25),  # A4
        (71, 0.25),  # B4
    ],
    answer_at_fifth=False,
    approx_measures=38,
    notes="Compound arpeggio theme in 3/8 time. Lively, dance-like character.",
)

# -------------------------------------------------------------------
# BWV 775 — Invention No.4 in D minor
# Theme: D-E-F-G-A with characteristic trill-like figure
# -------------------------------------------------------------------
GOLD_INVENTIONS[775] = GoldInvention(
    bwv=775,
    number=4,
    title="Invention No. 4 in D minor",
    key_signature="d",
    time_signature="3/8",
    theme_pitches=[
        (62, 0.5),   # D4
        (65, 0.25),  # F4
        (64, 0.25),  # E4
        (65, 0.25),  # F4
        (62, 0.25),  # D4
        (69, 0.5),   # A4
        (62, 0.25),  # D4
        (65, 0.25),  # F4
        (64, 0.25),  # E4
        (62, 0.25),  # D4
        (64, 0.25),  # E4
        (60, 0.25),  # C4
    ],
    answer_at_fifth=True,
    approx_measures=52,
    notes="Graceful dance theme with characteristic trill figure. Answer at the fifth.",
)

# -------------------------------------------------------------------
# BWV 776 — Invention No.5 in Eb major
# -------------------------------------------------------------------
GOLD_INVENTIONS[776] = GoldInvention(
    bwv=776,
    number=5,
    title="Invention No. 5 in E-flat major",
    key_signature="Eb",
    time_signature="4/4",
    theme_pitches=[
        (63, 0.25),  # Eb4
        (67, 0.25),  # G4
        (70, 0.25),  # Bb4
        (75, 0.25),  # Eb5
        (74, 0.25),  # D5
        (75, 0.25),  # Eb5
        (72, 0.25),  # C5
        (70, 0.25),  # Bb4
        (67, 0.25),  # G4
        (70, 0.25),  # Bb4
        (72, 0.25),  # C5
        (63, 0.25),  # Eb4
        (65, 0.5),   # F4
    ],
    answer_at_fifth=False,
    approx_measures=16,
    notes="Brilliant arpeggio theme. Compact form.",
)

# -------------------------------------------------------------------
# BWV 777 — Invention No.6 in E major
# -------------------------------------------------------------------
GOLD_INVENTIONS[777] = GoldInvention(
    bwv=777,
    number=6,
    title="Invention No. 6 in E major",
    key_signature="E",
    time_signature="3/8",
    theme_pitches=[
        (64, 0.25),  # E4
        (68, 0.25),  # G#4
        (71, 0.25),  # B4
        (76, 0.25),  # E5
        (75, 0.25),  # D#5
        (73, 0.25),  # C#5
        (71, 0.25),  # B4
        (73, 0.25),  # C#5
        (69, 0.25),  # A4
        (68, 0.25),  # G#4
        (71, 0.5),   # B4
    ],
    answer_at_fifth=False,
    approx_measures=30,
    notes="Arpeggio-based theme in 3/8. Bright, energetic character.",
)

# -------------------------------------------------------------------
# BWV 778 — Invention No.7 in E minor
# -------------------------------------------------------------------
GOLD_INVENTIONS[778] = GoldInvention(
    bwv=778,
    number=7,
    title="Invention No. 7 in E minor",
    key_signature="e",
    time_signature="4/4",
    theme_pitches=[
        (64, 0.5),   # E4
        (71, 0.25),  # B4
        (69, 0.25),  # A4
        (67, 0.25),  # G4
        (66, 0.25),  # F#4
        (64, 0.25),  # E4
        (67, 0.25),  # G4
        (66, 0.5),   # F#4
        (64, 0.25),  # E4
        (62, 0.25),  # D4
    ],
    answer_at_fifth=True,
    approx_measures=22,
    notes="Expressive descending theme. Rich in suspensions.",
)

# -------------------------------------------------------------------
# BWV 779 — Invention No.8 in F major
# -------------------------------------------------------------------
GOLD_INVENTIONS[779] = GoldInvention(
    bwv=779,
    number=8,
    title="Invention No. 8 in F major",
    key_signature="F",
    time_signature="3/4",
    theme_pitches=[
        (65, 0.5),   # F4
        (69, 0.5),   # A4
        (72, 0.5),   # C5
        (77, 0.25),  # F5
        (76, 0.25),  # E5
        (74, 0.25),  # D5
        (72, 0.25),  # C5
        (74, 0.25),  # D5
        (72, 0.25),  # C5
        (70, 0.25),  # Bb4
        (69, 0.25),  # A4
    ],
    answer_at_fifth=True,
    approx_measures=34,
    notes="Stately theme in 3/4. Rising arpeggio followed by descending scale.",
)

# -------------------------------------------------------------------
# BWV 780 — Invention No.9 in F minor
# -------------------------------------------------------------------
GOLD_INVENTIONS[780] = GoldInvention(
    bwv=780,
    number=9,
    title="Invention No. 9 in F minor",
    key_signature="f",
    time_signature="3/4",
    theme_pitches=[
        (65, 0.5),   # F4
        (63, 0.25),  # Eb4
        (61, 0.25),  # Db4
        (60, 0.25),  # C4
        (61, 0.25),  # Db4
        (63, 0.25),  # Eb4
        (65, 0.25),  # F4
        (68, 0.5),   # Ab4
        (65, 0.25),  # F4
        (72, 0.25),  # C5
        (68, 0.5),   # Ab4
    ],
    answer_at_fifth=False,
    approx_measures=32,
    notes="Lyrical, expressive theme. Chromatic inflections.",
)

# -------------------------------------------------------------------
# BWV 781 — Invention No.10 in G major
# -------------------------------------------------------------------
GOLD_INVENTIONS[781] = GoldInvention(
    bwv=781,
    number=10,
    title="Invention No. 10 in G major",
    key_signature="G",
    time_signature="9/8",
    theme_pitches=[
        (67, 0.25),  # G4
        (71, 0.25),  # B4
        (74, 0.25),  # D5
        (72, 0.25),  # C5
        (71, 0.25),  # B4
        (69, 0.25),  # A4
        (67, 0.25),  # G4
        (71, 0.25),  # B4
        (69, 0.25),  # A4
        (67, 0.25),  # G4
        (66, 0.25),  # F#4
        (69, 0.25),  # A4
    ],
    answer_at_fifth=False,
    approx_measures=22,
    notes="Compound time, flowing pastoral character. Arpeggio-based theme.",
)

# -------------------------------------------------------------------
# BWV 782 — Invention No.11 in G minor
# -------------------------------------------------------------------
GOLD_INVENTIONS[782] = GoldInvention(
    bwv=782,
    number=11,
    title="Invention No. 11 in G minor",
    key_signature="g",
    time_signature="4/4",
    theme_pitches=[
        (67, 0.25),  # G4
        (70, 0.25),  # Bb4
        (74, 0.25),  # D5
        (70, 0.25),  # Bb4
        (72, 0.25),  # C5
        (69, 0.25),  # A4
        (70, 0.25),  # Bb4
        (67, 0.25),  # G4
        (69, 0.25),  # A4
        (65, 0.25),  # F4
        (67, 0.25),  # G4
        (66, 0.25),  # F#4
        (67, 0.5),   # G4
    ],
    answer_at_fifth=False,
    approx_measures=20,
    notes="Descending arpeggio theme. Contrapuntal intensity throughout.",
)

# -------------------------------------------------------------------
# BWV 783 — Invention No.12 in A major
# -------------------------------------------------------------------
GOLD_INVENTIONS[783] = GoldInvention(
    bwv=783,
    number=12,
    title="Invention No. 12 in A major",
    key_signature="A",
    time_signature="12/8",
    theme_pitches=[
        (69, 0.25),  # A4
        (73, 0.25),  # C#5
        (76, 0.25),  # E5
        (73, 0.25),  # C#5
        (74, 0.25),  # D5
        (76, 0.25),  # E5
        (73, 0.25),  # C#5
        (69, 0.25),  # A4
        (71, 0.25),  # B4
        (73, 0.25),  # C#5
        (74, 0.5),   # D5
        (73, 0.25),  # C#5
        (71, 0.25),  # B4
    ],
    answer_at_fifth=False,
    approx_measures=24,
    notes="Compound time (12/8). Jig-like, energetic character.",
)

# -------------------------------------------------------------------
# BWV 784 — Invention No.13 in A minor
# -------------------------------------------------------------------
GOLD_INVENTIONS[784] = GoldInvention(
    bwv=784,
    number=13,
    title="Invention No. 13 in A minor",
    key_signature="a",
    time_signature="4/4",
    theme_pitches=[
        (69, 0.25),  # A4
        (71, 0.25),  # B4
        (72, 0.25),  # C5
        (74, 0.25),  # D5
        (76, 0.25),  # E5
        (72, 0.25),  # C5
        (74, 0.25),  # D5
        (71, 0.25),  # B4
        (72, 0.25),  # C5
        (69, 0.25),  # A4
        (71, 0.25),  # B4
        (68, 0.25),  # G#4
        (69, 0.5),   # A4
    ],
    answer_at_fifth=False,
    approx_measures=20,
    notes="Scalar 16th-note theme similar to Invention 2. Clean counterpoint.",
)

# -------------------------------------------------------------------
# BWV 785 — Invention No.14 in Bb major
# -------------------------------------------------------------------
GOLD_INVENTIONS[785] = GoldInvention(
    bwv=785,
    number=14,
    title="Invention No. 14 in B-flat major",
    key_signature="Bb",
    time_signature="4/4",
    theme_pitches=[
        (70, 0.25),  # Bb4
        (72, 0.25),  # C5
        (74, 0.25),  # D5
        (70, 0.25),  # Bb4
        (77, 0.25),  # F5
        (74, 0.25),  # D5
        (72, 0.25),  # C5
        (74, 0.25),  # D5
        (70, 0.25),  # Bb4
        (69, 0.25),  # A4
        (70, 0.5),   # Bb4
    ],
    answer_at_fifth=False,
    approx_measures=22,
    notes="Playful, scalar theme. Relatively straightforward counterpoint.",
)

# -------------------------------------------------------------------
# BWV 786 — Invention No.15 in B minor
# -------------------------------------------------------------------
GOLD_INVENTIONS[786] = GoldInvention(
    bwv=786,
    number=15,
    title="Invention No. 15 in B minor",
    key_signature="b",
    time_signature="4/4",
    theme_pitches=[
        (59, 0.25),  # B3
        (61, 0.25),  # C#4
        (62, 0.25),  # D4
        (64, 0.25),  # E4
        (66, 0.25),  # F#4
        (62, 0.25),  # D4
        (64, 0.25),  # E4
        (61, 0.25),  # C#4
        (62, 0.25),  # D4
        (59, 0.25),  # B3
        (61, 0.25),  # C#4
        (57, 0.25),  # A3
        (59, 0.5),   # B3
    ],
    answer_at_fifth=False,
    approx_measures=22,
    notes="The final invention. Scalar theme, deeply expressive. Mirror of No.1.",
)


# ═══════════════════════════════════════════════════════════════════════════
# Accessor functions
# ═══════════════════════════════════════════════════════════════════════════

def get_gold_invention(bwv: int) -> Optional[GoldInvention]:
    """Get gold annotation by BWV number."""
    return GOLD_INVENTIONS.get(bwv)


def list_gold_inventions() -> list[GoldInvention]:
    """Return all gold inventions sorted by BWV."""
    return [GOLD_INVENTIONS[k] for k in sorted(GOLD_INVENTIONS)]

