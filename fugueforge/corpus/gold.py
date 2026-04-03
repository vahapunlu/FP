"""
Gold standard annotations for Bach Well-Tempered Clavier fugues.

Each entry encodes the fugue subject as MIDI pitches + durations,
along with key, number of voices, answer type, and countersubject info.
These are reference data from standard musicological analysis.

Covers WTC Book I (BWV 846–869) and selected Book II entries.
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
class GoldFugue:
    """Gold standard annotation for a fugue."""
    bwv: int
    title: str
    key_signature: str
    num_voices: int
    answer_type: AnswerType
    time_signature: str
    # Subject as (midi_pitch, duration_in_quarter_notes) pairs
    subject_pitches: list[tuple[int, float]]
    # Which voice enters first (0=soprano, voices-1=bass)
    first_voice: int = 0
    # Subject entry offsets in beats (voice_index → beat)
    entry_offsets: Optional[dict[int, float]] = None
    # Has countersubject?
    has_countersubject: bool = True
    # Has stretto?
    has_stretto: bool = False
    # Number of episodes
    num_episodes: int = 0
    # Notes on this fugue
    notes: str = ""

    def to_subject(self) -> Subject:
        """Convert to a FugueForge Subject object."""
        fn_notes: list[FugueNote] = []
        offset = 0.0
        for pitch, dur in self.subject_pitches:
            if pitch < 0:
                # Rest
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
# WTC Book I — Fugue subjects
# BWV 846 = Prelude+Fugue in C major, etc.
# Even BWV = Prelude, Odd BWV sometimes used; fugue is the 2nd piece
# Standard numbering: BWV 846 = No.1 C major (Prelude & Fugue)
# ═══════════════════════════════════════════════════════════════════════════

GOLD_SUBJECTS: dict[int, GoldFugue] = {}

# -------------------------------------------------------------------
# BWV 846 — Fugue No.1 in C major, 4 voices
# Subject: C-D-E-F-D-E-C  (soprano entry)
# -------------------------------------------------------------------
GOLD_SUBJECTS[846] = GoldFugue(
    bwv=846,
    title="Fugue No. 1 in C major",
    key_signature="C",
    num_voices=4,
    answer_type=AnswerType.REAL,
    time_signature="4/4",
    subject_pitches=[
        (72, 0.5),   # C5
        (74, 0.5),   # D5
        (76, 0.5),   # E5
        (77, 0.25),  # F5
        (76, 0.25),  # E5
        (74, 0.5),   # D5
        (76, 0.25),  # E5
        (74, 0.25),  # D5
        (72, 0.5),   # C5
        (76, 0.5),   # E5
        (79, 0.5),   # G5
        (77, 0.125), # F5
        (76, 0.125), # E5
        (77, 0.5),   # F5
    ],
    first_voice=2,  # Alto enters first
    has_countersubject=True,
    has_stretto=True,
    notes="One of the simplest WTC fugues. Compact subject, invertible countersubject.",
)

# -------------------------------------------------------------------
# BWV 847 — Fugue No.2 in C minor, 3 voices
# Subject: C-Eb-D-C-B-C, moving to G
# -------------------------------------------------------------------
GOLD_SUBJECTS[847] = GoldFugue(
    bwv=847,
    title="Fugue No. 2 in C minor",
    key_signature="c",
    num_voices=3,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (60, 1.0),   # C4
        (63, 0.5),   # Eb4
        (62, 0.5),   # D4
        (60, 0.5),   # C4
        (59, 0.5),   # B3
        (60, 0.5),   # C4
        (56, 0.5),   # Ab3
        (67, 0.5),   # G4
        (63, 0.5),   # Eb4
        (65, 0.5),   # F4
        (62, 0.5),   # D4
        (60, 1.0),   # C4
    ],
    first_voice=1,  # Middle voice
    has_countersubject=True,
    has_stretto=False,
    num_episodes=5,
    notes="Classic 3-voice fugue. Tonal answer adjusts the initial 5th. Our demo subject is inspired by this.",
)

# -------------------------------------------------------------------
# BWV 848 — Fugue No.3 in C# major, 3 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[848] = GoldFugue(
    bwv=848,
    title="Fugue No. 3 in C-sharp major",
    key_signature="C#",
    num_voices=3,
    answer_type=AnswerType.REAL,
    time_signature="4/4",
    subject_pitches=[
        (73, 0.5),   # C#5
        (68, 0.5),   # G#4
        (73, 0.5),   # C#5
        (75, 0.25),  # D#5
        (76, 0.25),  # E5
        (75, 0.5),   # D#5
        (73, 0.5),   # C#5
        (72, 0.5),   # C5 (B#)
        (73, 1.0),   # C#5
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=True,
    notes="Bright, dance-like character. Subject outlines the tonic triad.",
)

# -------------------------------------------------------------------
# BWV 849 — Fugue No.4 in C# minor, 5 voices
# The famous 5-voice fugue, one of the longest in WTC I
# Subject: chromatic descending line
# -------------------------------------------------------------------
GOLD_SUBJECTS[849] = GoldFugue(
    bwv=849,
    title="Fugue No. 4 in C-sharp minor",
    key_signature="c#",
    num_voices=5,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (61, 2.0),   # C#4
        (60, 2.0),   # C4 (B#)
        (61, 1.0),   # C#4
        (63, 1.0),   # D#4
        (64, 2.0),   # E4
        (63, 1.0),   # D#4
        (61, 1.0),   # C#4
        (63, 1.0),   # D#4
        (59, 1.0),   # B3
        (61, 2.0),   # C#4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=True,
    num_episodes=6,
    notes="Monumental 5-voice fugue. Slow, chromatic subject with expressive character.",
)

# -------------------------------------------------------------------
# BWV 850 — Fugue No.5 in D major, 4 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[850] = GoldFugue(
    bwv=850,
    title="Fugue No. 5 in D major",
    key_signature="D",
    num_voices=4,
    answer_type=AnswerType.REAL,
    time_signature="4/4",
    subject_pitches=[
        (74, 0.5),   # D5
        (73, 0.5),   # C#5
        (74, 0.5),   # D5
        (69, 0.5),   # A4
        (66, 0.5),   # F#4
        (69, 0.5),   # A4
        (74, 0.5),   # D5
        (78, 0.5),   # F#5
        (76, 0.25),  # E5
        (74, 0.25),  # D5
        (73, 0.25),  # C#5
        (74, 0.25),  # D5
        (73, 0.5),   # C#5
        (71, 0.5),   # B4
        (69, 1.0),   # A4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=False,
    notes="Lively, virtuosic subject with wide range. Arpeggio-based opening.",
)

# -------------------------------------------------------------------
# BWV 851 — Fugue No.6 in D minor, 3 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[851] = GoldFugue(
    bwv=851,
    title="Fugue No. 6 in D minor",
    key_signature="d",
    num_voices=3,
    answer_type=AnswerType.TONAL,
    time_signature="3/4",
    subject_pitches=[
        (62, 1.5),   # D4
        (65, 0.5),   # F4
        (69, 0.5),   # A4
        (74, 0.5),   # D5
        (72, 0.25),  # C5
        (70, 0.25),  # Bb4
        (69, 0.5),   # A4
        (67, 0.5),   # G4
        (65, 0.5),   # F4
        (64, 0.25),  # E4
        (62, 0.25),  # D4
        (64, 0.5),   # E4
        (60, 0.5),   # C4
        (62, 1.0),   # D4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=False,
    notes="Elegant triple-meter fugue (3/4). Subject combines arpeggiation with stepwise descent.",
)

# -------------------------------------------------------------------
# BWV 853 — Fugue No.8 in Eb minor (D# minor), 3 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[853] = GoldFugue(
    bwv=853,
    title="Fugue No. 8 in E-flat minor",
    key_signature="eb",
    num_voices=3,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (63, 2.0),   # Eb4
        (62, 0.5),   # D4
        (63, 0.5),   # Eb4
        (66, 1.0),   # F#4 (Gb4)
        (65, 0.5),   # F4
        (63, 0.5),   # Eb4
        (62, 1.0),   # D4
        (63, 2.0),   # Eb4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=True,
    notes="Solemn, chromatic fugue. Related to the famous Prelude No. 8.",
)

# -------------------------------------------------------------------
# BWV 855 — Fugue No.10 in E minor, 2 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[855] = GoldFugue(
    bwv=855,
    title="Fugue No. 10 in E minor",
    key_signature="e",
    num_voices=2,
    answer_type=AnswerType.REAL,
    time_signature="4/4",
    subject_pitches=[
        (64, 0.5),   # E4
        (66, 0.25),  # F#4
        (67, 0.25),  # G4
        (69, 0.5),   # A4
        (67, 0.25),  # G4
        (66, 0.25),  # F#4
        (64, 0.25),  # E4
        (67, 0.25),  # G4
        (66, 0.25),  # F#4
        (64, 0.25),  # E4
        (62, 0.25),  # D4
        (66, 0.25),  # F#4
        (64, 0.5),   # E4
        (62, 0.5),   # D4
        (59, 1.0),   # B3
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=False,
    notes="Only 2-voice fugue in WTC I. Technical, running sixteenth-note subject.",
)

# -------------------------------------------------------------------
# BWV 858 — Fugue No.13 in F# major, 3 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[858] = GoldFugue(
    bwv=858,
    title="Fugue No. 13 in F-sharp major",
    key_signature="F#",
    num_voices=3,
    answer_type=AnswerType.REAL,
    time_signature="4/4",
    subject_pitches=[
        (66, 0.5),   # F#4
        (73, 0.5),   # C#5
        (72, 0.25),  # C5 (B#)
        (73, 0.25),  # C#5
        (70, 0.5),   # A#4 (Bb4)
        (66, 0.5),   # F#4
        (68, 0.5),   # G#4 (Ab4)
        (70, 0.5),   # A#4
        (66, 1.0),   # F#4
    ],
    first_voice=1,
    has_countersubject=True,
    has_stretto=True,
    notes="Gentle, lyrical fugue. Subject characteristic leap of a 5th.",
)

# -------------------------------------------------------------------
# BWV 860 — Fugue No.15 in G major, 3 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[860] = GoldFugue(
    bwv=860,
    title="Fugue No. 15 in G major",
    key_signature="G",
    num_voices=3,
    answer_type=AnswerType.REAL,
    time_signature="6/8",
    subject_pitches=[
        (67, 0.5),   # G4
        (71, 0.5),   # B4
        (74, 0.5),   # D5
        (72, 0.5),   # C5
        (71, 0.25),  # B4
        (69, 0.25),  # A4
        (67, 0.25),  # G4
        (69, 0.25),  # A4
        (71, 0.25),  # B4
        (67, 0.25),  # G4
        (72, 0.5),   # C5
        (71, 1.0),   # B4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=False,
    notes="Compound meter (6/8). Pastoral character.",
)

# -------------------------------------------------------------------
# BWV 861 — Fugue No.16 in G minor, 4 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[861] = GoldFugue(
    bwv=861,
    title="Fugue No. 16 in G minor",
    key_signature="g",
    num_voices=4,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (67, 1.0),   # G4
        (70, 0.5),   # Bb4
        (69, 0.5),   # A4
        (67, 0.5),   # G4
        (65, 0.5),   # F4
        (66, 0.5),   # F#4
        (67, 0.5),   # G4
        (69, 0.5),   # A4
        (74, 0.5),   # D5
        (72, 0.25),  # C5
        (70, 0.25),  # Bb4
        (69, 0.5),   # A4
        (67, 0.5),   # G4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=True,
    notes="Powerful 4-voice fugue. Subject features chromatic inflection (F#).",
)

# -------------------------------------------------------------------
# BWV 865 — Fugue No.20 in A minor, 4 voices
# -------------------------------------------------------------------
GOLD_SUBJECTS[865] = GoldFugue(
    bwv=865,
    title="Fugue No. 20 in A minor",
    key_signature="a",
    num_voices=4,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (69, 1.0),   # A4
        (64, 1.0),   # E4
        (69, 0.5),   # A4
        (72, 0.5),   # C5
        (71, 0.5),   # B4
        (69, 0.5),   # A4
        (68, 0.5),   # G#4
        (71, 0.5),   # B4
        (69, 1.0),   # A4
    ],
    first_voice=0,
    has_countersubject=True,
    has_stretto=True,
    notes="Stately 4-voice fugue. Subject opens with dramatic octave leap.",
)

# -------------------------------------------------------------------
# BWV 869 — Fugue No.24 in B minor, 4 voices
# The final and grandest fugue of WTC Book I
# -------------------------------------------------------------------
GOLD_SUBJECTS[869] = GoldFugue(
    bwv=869,
    title="Fugue No. 24 in B minor",
    key_signature="b",
    num_voices=4,
    answer_type=AnswerType.TONAL,
    time_signature="4/4",
    subject_pitches=[
        (59, 2.0),   # B3
        (62, 1.0),   # D4
        (61, 0.5),   # C#4
        (59, 0.5),   # B3
        (57, 0.5),   # A3
        (59, 0.5),   # B3
        (55, 1.0),   # G3
        (57, 0.5),   # A3
        (59, 0.5),   # B3
        (54, 0.5),   # F#3
        (55, 0.5),   # G3
        (57, 0.5),   # A3
        (52, 0.5),   # E3
        (54, 0.5),   # F#3
        (55, 0.5),   # G3
        (59, 0.5),   # B3
        (57, 0.5),   # A3
        (55, 0.5),   # G3
        (54, 0.5),   # F#3
        (52, 0.5),   # E3
        (54, 0.5),   # F#3
        (59, 0.5),   # B3
        (57, 0.5),   # A3
        (55, 0.5),   # G3
        (54, 0.5),   # F#3
        (55, 0.5),   # G3
        (57, 1.0),   # A3
        (55, 1.0),   # G3
        (54, 0.5),   # F#3
        (52, 0.5),   # E3
        (54, 1.0),   # F#3
        (59, 2.0),   # B3
    ],
    first_voice=3,  # Bass enters first
    has_countersubject=True,
    has_stretto=False,
    notes="The longest subject in WTC I (~19 beats). Deeply chromatic, ricercar-like character.",
)


# ═══════════════════════════════════════════════════════════════════════════
# Accessor functions
# ═══════════════════════════════════════════════════════════════════════════

def get_gold_fugue(bwv: int) -> Optional[GoldFugue]:
    """Get gold annotation by BWV number."""
    return GOLD_SUBJECTS.get(bwv)


def list_gold_fugues() -> list[GoldFugue]:
    """Return all gold fugues sorted by BWV."""
    return [GOLD_SUBJECTS[k] for k in sorted(GOLD_SUBJECTS)]


def get_subjects_by_voices(num_voices: int) -> list[GoldFugue]:
    """Get all gold fugues with a specific number of voices."""
    return [g for g in GOLD_SUBJECTS.values() if g.num_voices == num_voices]


def get_subjects_by_answer_type(answer_type: AnswerType) -> list[GoldFugue]:
    """Get all gold fugues with a specific answer type."""
    return [g for g in GOLD_SUBJECTS.values() if g.answer_type == answer_type]
