"""
Corpus loader — load fugues from MIDI/MusicXML files and pair with gold annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from music21 import converter

from fugueforge.core.representation import FugueNote, Subject, stream_to_fuguenotes
from .gold import GoldFugue, GOLD_SUBJECTS


@dataclass
class CorpusEntry:
    """A loaded fugue paired with its gold annotation."""
    bwv: int
    gold: GoldFugue
    voices: dict[int, list[FugueNote]]
    source_path: Optional[Path] = None

    @property
    def num_notes(self) -> int:
        return sum(len(v) for v in self.voices.values())


def load_fugue_from_file(
    path: str | Path,
    bwv: Optional[int] = None,
) -> CorpusEntry:
    """
    Load a fugue from a MIDI or MusicXML file.
    If bwv is provided, pairs with gold annotation.
    """
    path = Path(path)
    score = converter.parse(str(path))

    voices: dict[int, list[FugueNote]] = {}
    for i, part in enumerate(score.parts):
        notes = stream_to_fuguenotes(part, voice=i)
        if notes:
            voices[i] = notes

    gold = GOLD_SUBJECTS.get(bwv) if bwv else None
    if gold is None and bwv:
        # Create a minimal gold entry
        gold = GoldFugue(
            bwv=bwv,
            title=f"BWV {bwv}",
            key_signature="C",
            num_voices=len(voices),
            answer_type=__import__("fugueforge.core.representation", fromlist=["AnswerType"]).AnswerType.REAL,
            time_signature="4/4",
            subject_pitches=[],
        )

    return CorpusEntry(
        bwv=bwv or 0,
        gold=gold,
        voices=voices,
        source_path=path,
    )


def scan_corpus_dir(
    directory: str | Path,
    extensions: tuple[str, ...] = (".mid", ".midi", ".xml", ".mxl", ".musicxml"),
) -> list[Path]:
    """Find all music files in a directory."""
    directory = Path(directory)
    results: list[Path] = []
    for ext in extensions:
        results.extend(directory.rglob(f"*{ext}"))
    return sorted(results)
