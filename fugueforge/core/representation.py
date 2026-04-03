"""
FugueForge Representation Layer

Fugue-aware symbolic music representation.
Handles conversion between music21, internal token format, and MIDI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from music21 import converter, key, meter, note, pitch, stream


# ---------------------------------------------------------------------------
# Enums for structural roles
# ---------------------------------------------------------------------------

class VoiceType(Enum):
    SOPRANO = auto()
    ALTO = auto()
    TENOR = auto()
    BASS = auto()


class SectionType(Enum):
    EXPOSITION = auto()
    MIDDLE_ENTRY = auto()
    EPISODE = auto()
    STRETTO = auto()
    CODA = auto()


class EntryRole(Enum):
    """Role of a melodic fragment within the fugue structure."""
    SUBJECT = auto()
    ANSWER = auto()          # tonal or real
    COUNTERSUBJECT = auto()
    FREE_COUNTERPOINT = auto()
    EPISODE_MATERIAL = auto()
    CODETTA = auto()


class AnswerType(Enum):
    REAL = auto()
    TONAL = auto()


class TransformType(Enum):
    NONE = auto()
    INVERSION = auto()
    AUGMENTATION = auto()
    DIMINUTION = auto()
    RETROGRADE = auto()
    INVERSION_AUGMENTATION = auto()


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class FugueNote:
    """Single note with structural metadata."""
    pitch: int            # MIDI pitch (0-127), -1 for rest
    duration: float       # in quarter-note units
    voice: int            # 0-based voice index
    offset: float         # absolute offset in quarter notes
    role: EntryRole = EntryRole.FREE_COUNTERPOINT
    transform: TransformType = TransformType.NONE
    tied: bool = False

    @property
    def is_rest(self) -> bool:
        return self.pitch == -1

    def to_music21(self) -> note.GeneralNote:
        if self.is_rest:
            r = note.Rest()
            r.quarterLength = self.duration
            return r
        n = note.Note(self.pitch, quarterLength=self.duration)
        return n


@dataclass
class Subject:
    """A fugue subject with analysis metadata."""
    notes: list[FugueNote]
    key_signature: str = "C"       # e.g. "C", "g" (minor)
    time_signature: str = "4/4"
    answer_type: AnswerType = AnswerType.REAL
    tonal_answer_notes: Optional[list[FugueNote]] = None

    @property
    def duration(self) -> float:
        """Total duration in quarter notes."""
        if not self.notes:
            return 0.0
        last = self.notes[-1]
        return last.offset + last.duration - self.notes[0].offset

    @property
    def pitch_range(self) -> tuple[int, int]:
        pitches = [n.pitch for n in self.notes if not n.is_rest]
        return (min(pitches), max(pitches)) if pitches else (0, 0)

    @property
    def interval_sequence(self) -> list[int]:
        """Successive MIDI interval list (semitones)."""
        pitches = [n.pitch for n in self.notes if not n.is_rest]
        return [b - a for a, b in zip(pitches, pitches[1:])]

    def to_stream(self) -> stream.Part:
        part = stream.Part()
        for fn in self.notes:
            el = fn.to_music21()
            part.insert(fn.offset, el)
        ks = key.Key(self.key_signature)
        ts = meter.TimeSignature(self.time_signature)
        part.insert(0, ks)
        part.insert(0, ts)
        return part


@dataclass
class VoiceEntry:
    """Metadata for a single voice entry in the exposition."""
    voice: int
    role: EntryRole
    start_offset: float
    subject_ref: Optional[Subject] = None
    transform: TransformType = TransformType.NONE


@dataclass
class ExpositionPlan:
    """Plan for the fugue exposition."""
    num_voices: int
    entries: list[VoiceEntry] = field(default_factory=list)
    answer_type: AnswerType = AnswerType.REAL
    has_countersubject: bool = True
    codetta_offsets: list[float] = field(default_factory=list)


@dataclass
class SectionPlan:
    """A planned section in the overall fugue structure."""
    section_type: SectionType
    start_offset: float
    estimated_duration: float
    key_area: str = ""
    entries: list[VoiceEntry] = field(default_factory=list)
    notes: str = ""


@dataclass
class FuguePlan:
    """Complete structural plan for a fugue."""
    subject: Subject
    num_voices: int = 3
    key_signature: str = "C"
    time_signature: str = "4/4"
    exposition: Optional[ExpositionPlan] = None
    sections: list[SectionPlan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token types for fugue-aware tokenization
# ---------------------------------------------------------------------------

class TokenType(Enum):
    # Pitch / duration
    NOTE_ON = auto()
    NOTE_OFF = auto()
    REST = auto()
    TIME_SHIFT = auto()
    DURATION = auto()

    # Voice
    VOICE_SELECT = auto()

    # Structural
    SECTION_START = auto()
    SECTION_END = auto()
    SUBJECT_START = auto()
    SUBJECT_END = auto()
    ANSWER_START = auto()
    ANSWER_END = auto()
    COUNTERSUBJECT_START = auto()
    COUNTERSUBJECT_END = auto()
    EPISODE_START = auto()
    EPISODE_END = auto()
    STRETTO_START = auto()
    STRETTO_END = auto()

    # Transform markers
    INVERSION = auto()
    AUGMENTATION = auto()
    DIMINUTION = auto()
    RETROGRADE = auto()

    # Tonal
    KEY_CHANGE = auto()
    CADENCE = auto()

    # Special
    BAR = auto()
    PAD = auto()
    BOS = auto()
    EOS = auto()


@dataclass
class FugueToken:
    """A single token in the fugue-aware representation."""
    type: TokenType
    value: int = 0           # pitch, duration quantized, voice id, etc.
    voice: int = -1          # -1 = global

    def __repr__(self) -> str:
        return f"<{self.type.name}:{self.value} v={self.voice}>"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def stream_to_fuguenotes(part: stream.Part, voice_idx: int = 0) -> list[FugueNote]:
    """Convert a music21 Part to a list of FugueNotes."""
    result: list[FugueNote] = []
    for el in part.recurse().notesAndRests:
        if isinstance(el, note.Rest):
            fn = FugueNote(
                pitch=-1,
                duration=float(el.quarterLength),
                voice=voice_idx,
                offset=float(el.offset),
            )
        elif isinstance(el, note.Note):
            fn = FugueNote(
                pitch=el.pitch.midi,
                duration=float(el.quarterLength),
                voice=voice_idx,
                offset=float(el.offset),
            )
        else:
            continue
        result.append(fn)
    return result


def fuguenotes_to_stream(notes: list[FugueNote]) -> stream.Part:
    """Convert a list of FugueNotes back to a music21 Part."""
    part = stream.Part()
    for fn in notes:
        el = fn.to_music21()
        part.insert(fn.offset, el)
    return part


def load_subject_from_file(filepath: str, voice_idx: int = 0) -> Subject:
    """Load a subject from a MIDI / MusicXML file."""
    score = converter.parse(filepath)
    part = score.parts[0] if score.parts else score
    notes = stream_to_fuguenotes(part, voice_idx)

    # Try to detect key
    k = part.analyze("key")
    ks = k.tonic.name
    if k.mode == "minor":
        ks = ks.lower()

    # Time signature
    ts_obj = part.recurse().getElementsByClass(meter.TimeSignature)
    ts = str(ts_obj[0]) if ts_obj else "4/4"

    return Subject(notes=notes, key_signature=ks, time_signature=ts)
