"""
FugueForge Analyzer

Analyzes existing fugues to detect:
- Subject boundaries and occurrences
- Answer type (tonal / real)
- Countersubject regions
- Section types (exposition, episode, stretto, coda)
- Cadences and key areas

Uses music21 for parsing and interval analysis, plus custom pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from music21 import analysis, converter, interval, key, note, stream

from .representation import (
    AnswerType,
    EntryRole,
    FugueNote,
    SectionType,
    Subject,
    VoiceEntry,
    stream_to_fuguenotes,
)


# ---------------------------------------------------------------------------
# Subject detection helpers
# ---------------------------------------------------------------------------

def extract_interval_profile(notes: list[FugueNote]) -> list[int]:
    """Extract semitone intervals, ignoring rests."""
    pitches = [n.pitch for n in notes if not n.is_rest]
    return [b - a for a, b in zip(pitches, pitches[1:])]


def extract_rhythm_profile(notes: list[FugueNote]) -> list[float]:
    """Extract duration sequence, ignoring rests."""
    return [n.duration for n in notes if not n.is_rest]


def match_interval_pattern(
    haystack: list[int],
    needle: list[int],
    tolerance: int = 0,
) -> list[int]:
    """
    Find all starting indices where needle matches in haystack.
    tolerance allows minor interval deviations (e.g. tonal answer: 1 semitone).
    """
    matches: list[int] = []
    if not needle or len(needle) > len(haystack):
        return matches
    for i in range(len(haystack) - len(needle) + 1):
        match = True
        for j, n_int in enumerate(needle):
            if abs(haystack[i + j] - n_int) > tolerance:
                match = False
                break
        if match:
            matches.append(i)
    return matches


def match_transposed_pattern(
    haystack: list[int],
    needle: list[int],
) -> list[int]:
    """Find exact transposition matches (intervals identical)."""
    return match_interval_pattern(haystack, needle, tolerance=0)


def match_tonal_answer_pattern(
    haystack: list[int],
    needle: list[int],
) -> list[int]:
    """
    Find tonal answer matches where at most the first/last interval
    differs by 1-2 semitones (scale-degree adjustment at 5th↔1st).
    """
    matches: list[int] = []
    if not needle or len(needle) > len(haystack):
        return matches
    for i in range(len(haystack) - len(needle) + 1):
        diffs = []
        for j, n_int in enumerate(needle):
            diffs.append(abs(haystack[i + j] - n_int))

        # Allow first or last interval to differ by up to 2 semitones
        all_ok = True
        for j, d in enumerate(diffs):
            if d == 0:
                continue
            if d <= 2 and (j == 0 or j == len(diffs) - 1):
                continue
            all_ok = False
            break
        if all_ok:
            matches.append(i)
    return matches


# ---------------------------------------------------------------------------
# Answer type detection
# ---------------------------------------------------------------------------

def detect_answer_type(
    subject_intervals: list[int],
    answer_intervals: list[int],
) -> AnswerType:
    """
    Determine if an answer is real (exact transposition) or tonal
    (adjusted at scale-degree boundaries).
    """
    if subject_intervals == answer_intervals:
        return AnswerType.REAL

    # Check if only first/last intervals differ (tonal adjustment)
    if len(subject_intervals) == len(answer_intervals):
        diffs = [
            abs(a - b)
            for a, b in zip(subject_intervals, answer_intervals)
        ]
        non_zero = [(i, d) for i, d in enumerate(diffs) if d != 0]
        if all(d <= 2 for _, d in non_zero):
            if all(i == 0 or i == len(diffs) - 1 for i, _ in non_zero):
                return AnswerType.TONAL

    return AnswerType.TONAL  # fallback: if different, assume tonal


def compute_tonal_answer(
    subject: Subject,
    transposition_semitones: int = 7,
) -> list[FugueNote]:
    """
    Generate a tonal answer for the subject.

    Real answer: transpose everything up a 5th.
    Tonal answer: adjust the 5th↔1st degree mutation.

    A proper tonal answer replaces the tonic-dominant boundary:
    - Notes that emphasize the 5th degree → answer treats them as the new tonic
    - The leap 1→5 becomes 5→1 (down a 4th instead of up a 5th)
    """
    if not subject.notes:
        return []

    # Simple approach: transpose all notes, then adjust the first note
    # if the subject begins on the tonic and leaps to the dominant
    answer_notes: list[FugueNote] = []

    for fn in subject.notes:
        new_fn = FugueNote(
            pitch=fn.pitch + transposition_semitones if not fn.is_rest else -1,
            duration=fn.duration,
            voice=fn.voice,
            offset=fn.offset,
            role=EntryRole.ANSWER,
        )
        answer_notes.append(new_fn)

    # Tonal adjustment: if subject starts on tonic and goes to dominant,
    # first interval should be adjusted.
    # In a real implementation this needs scale-degree awareness.
    # For now, detect if first interval is +7 (P5 up) and adjust to +5 (P4 up).
    if len(answer_notes) >= 2:
        subj_pitches = [n.pitch for n in subject.notes if not n.is_rest]
        if len(subj_pitches) >= 2:
            first_interval = subj_pitches[1] - subj_pitches[0]
            if first_interval == 7:  # P5 up → adjust to P4 up in answer
                # Shift second note down by 2 semitones
                if not answer_notes[1].is_rest:
                    answer_notes[1] = FugueNote(
                        pitch=answer_notes[1].pitch - 2,
                        duration=answer_notes[1].duration,
                        voice=answer_notes[1].voice,
                        offset=answer_notes[1].offset,
                        role=EntryRole.ANSWER,
                    )
            elif first_interval == -5:  # P4 down → adjust to P5 down
                if not answer_notes[1].is_rest:
                    answer_notes[1] = FugueNote(
                        pitch=answer_notes[1].pitch - 2,
                        duration=answer_notes[1].duration,
                        voice=answer_notes[1].voice,
                        offset=answer_notes[1].offset,
                        role=EntryRole.ANSWER,
                    )

    return answer_notes


# ---------------------------------------------------------------------------
# Subject occurrence finder
# ---------------------------------------------------------------------------

@dataclass
class SubjectOccurrence:
    """A detected occurrence of the subject in the score."""
    voice: int
    start_offset: float
    end_offset: float
    start_note_idx: int
    role: EntryRole  # SUBJECT or ANSWER
    transposition: int = 0  # semitones from original
    answer_type: Optional[AnswerType] = None


def find_subject_occurrences(
    subject: Subject,
    voices: dict[int, list[FugueNote]],
    tolerance: int = 1,
) -> list[SubjectOccurrence]:
    """
    Scan all voices for occurrences of the subject (real transpositions
    and tonal answers).
    """
    subject_intervals = extract_interval_profile(subject.notes)
    if not subject_intervals:
        return []

    subject_rhythm = extract_rhythm_profile(subject.notes)
    occurrences: list[SubjectOccurrence] = []

    for v, notes in voices.items():
        v_intervals = extract_interval_profile(notes)
        v_pitches = [n.pitch for n in notes if not n.is_rest]
        non_rest_notes = [n for n in notes if not n.is_rest]

        # Real transpositions (exact intervals)
        real_matches = match_transposed_pattern(v_intervals, subject_intervals)
        for idx in real_matches:
            start_n = non_rest_notes[idx]
            end_n = non_rest_notes[min(idx + len(subject_intervals), len(non_rest_notes) - 1)]
            transp = start_n.pitch - subject.notes[0].pitch
            occurrences.append(SubjectOccurrence(
                voice=v,
                start_offset=start_n.offset,
                end_offset=end_n.offset + end_n.duration,
                start_note_idx=idx,
                role=EntryRole.SUBJECT if (transp % 12 == 0) else EntryRole.SUBJECT,
                transposition=transp,
            ))

        # Tonal answer matches
        tonal_matches = match_tonal_answer_pattern(v_intervals, subject_intervals)
        for idx in tonal_matches:
            if idx in real_matches:
                continue  # Already counted as real
            start_n = non_rest_notes[idx]
            end_n = non_rest_notes[min(idx + len(subject_intervals), len(non_rest_notes) - 1)]
            transp = start_n.pitch - subject.notes[0].pitch
            occurrences.append(SubjectOccurrence(
                voice=v,
                start_offset=start_n.offset,
                end_offset=end_n.offset + end_n.duration,
                start_note_idx=idx,
                role=EntryRole.ANSWER,
                transposition=transp,
                answer_type=AnswerType.TONAL,
            ))

        # Inverted subject matches (negated interval profile)
        inverted_intervals = [-iv for iv in subject_intervals]
        inv_matches = match_transposed_pattern(v_intervals, inverted_intervals)
        matched_indices = set(real_matches) | set(tonal_matches)
        for idx in inv_matches:
            if idx in matched_indices:
                continue
            start_n = non_rest_notes[idx]
            end_n = non_rest_notes[min(idx + len(subject_intervals), len(non_rest_notes) - 1)]
            transp = start_n.pitch - subject.notes[0].pitch
            occurrences.append(SubjectOccurrence(
                voice=v,
                start_offset=start_n.offset,
                end_offset=end_n.offset + end_n.duration,
                start_note_idx=idx,
                role=EntryRole.SUBJECT,
                transposition=transp,
            ))

    # Sort by offset
    occurrences.sort(key=lambda o: o.start_offset)
    return occurrences


# ---------------------------------------------------------------------------
# Section detection (heuristic)
# ---------------------------------------------------------------------------

@dataclass
class DetectedSection:
    """A detected section in the fugue."""
    section_type: SectionType
    start_offset: float
    end_offset: float
    key_area: str = ""
    subject_entries: int = 0


def detect_sections(
    subject: Subject,
    voices: dict[int, list[FugueNote]],
    total_duration: float,
) -> list[DetectedSection]:
    """
    Heuristic section detection based on subject occurrence density.

    - Dense subject entries at the start → exposition
    - Gaps between entries → episodes
    - Dense entries at the end with overlaps → stretto
    - Final measures → coda
    """
    occurrences = find_subject_occurrences(subject, voices)
    if not occurrences:
        return [DetectedSection(
            section_type=SectionType.EXPOSITION,
            start_offset=0.0,
            end_offset=total_duration,
        )]

    sections: list[DetectedSection] = []
    subject_dur = subject.duration

    # Group occurrences into clusters (entries within subject_dur gap)
    clusters: list[list[SubjectOccurrence]] = []
    current_cluster: list[SubjectOccurrence] = [occurrences[0]]

    for occ in occurrences[1:]:
        prev_end = current_cluster[-1].end_offset
        if occ.start_offset - prev_end < subject_dur * 1.5:
            current_cluster.append(occ)
        else:
            clusters.append(current_cluster)
            current_cluster = [occ]
    clusters.append(current_cluster)

    # First cluster is exposition
    if clusters:
        exp = clusters[0]
        sections.append(DetectedSection(
            section_type=SectionType.EXPOSITION,
            start_offset=exp[0].start_offset,
            end_offset=exp[-1].end_offset,
            subject_entries=len(exp),
        ))

    # Middle clusters: check for stretto (overlapping entries)
    for cluster in clusters[1:]:
        has_overlap = False
        for i in range(len(cluster) - 1):
            if cluster[i + 1].start_offset < cluster[i].end_offset:
                has_overlap = True
                break

        stype = SectionType.STRETTO if has_overlap else SectionType.MIDDLE_ENTRY
        sections.append(DetectedSection(
            section_type=stype,
            start_offset=cluster[0].start_offset,
            end_offset=cluster[-1].end_offset,
            subject_entries=len(cluster),
        ))

    # Fill gaps with episodes
    episode_sections: list[DetectedSection] = []
    for i in range(len(sections) - 1):
        gap_start = sections[i].end_offset
        gap_end = sections[i + 1].start_offset
        if gap_end - gap_start > subject_dur * 0.5:
            episode_sections.append(DetectedSection(
                section_type=SectionType.EPISODE,
                start_offset=gap_start,
                end_offset=gap_end,
            ))

    sections.extend(episode_sections)
    sections.sort(key=lambda s: s.start_offset)

    return sections


# ---------------------------------------------------------------------------
# Cadence detection (simplified)
# ---------------------------------------------------------------------------

@dataclass
class DetectedCadence:
    """A detected cadence point."""
    offset: float
    type: str  # "PAC", "IAC", "HC", "DC", "PC"
    key_area: str = ""


def detect_cadences_simple(
    voices: dict[int, list[FugueNote]],
    key_sig: str = "C",
) -> list[DetectedCadence]:
    """
    Simple cadence detection: look for V→I bass motions followed
    by rhythmic elongation.
    """
    # Get bass voice (highest numbered)
    bass_voice = max(voices.keys()) if voices else 0
    bass_notes = voices.get(bass_voice, [])
    if len(bass_notes) < 2:
        return []

    # Determine tonic and dominant pitches
    from music21 import key as m21key
    k = m21key.Key(key_sig)
    tonic_pc = k.tonic.pitchClass
    dominant_pc = k.getDominant().pitchClass

    cadences: list[DetectedCadence] = []
    non_rest = [n for n in bass_notes if not n.is_rest]

    for i in range(len(non_rest) - 1):
        n1, n2 = non_rest[i], non_rest[i + 1]
        pc1 = n1.pitch % 12
        pc2 = n2.pitch % 12

        if pc1 == dominant_pc and pc2 == tonic_pc:
            # V → I motion
            if n2.duration >= 2.0:
                cadences.append(DetectedCadence(
                    offset=n2.offset,
                    type="PAC",
                    key_area=key_sig,
                ))
            else:
                cadences.append(DetectedCadence(
                    offset=n2.offset,
                    type="IAC",
                    key_area=key_sig,
                ))
        elif pc2 == dominant_pc and n2.duration >= 2.0:
            cadences.append(DetectedCadence(
                offset=n2.offset,
                type="HC",
                key_area=key_sig,
            ))

    return cadences


# ---------------------------------------------------------------------------
# Full fugue analysis
# ---------------------------------------------------------------------------

@dataclass
class FugueAnalysis:
    """Complete analysis of a fugue."""
    subject: Optional[Subject] = None
    subject_occurrences: list[SubjectOccurrence] = field(default_factory=list)
    sections: list[DetectedSection] = field(default_factory=list)
    cadences: list[DetectedCadence] = field(default_factory=list)
    key_areas: list[str] = field(default_factory=list)
    num_voices: int = 0


def analyze_fugue(
    score: stream.Score,
    subject_end_offset: Optional[float] = None,
) -> FugueAnalysis:
    """
    Analyze a fugue score.

    If subject_end_offset is given, uses the first voice's notes up to
    that offset as the subject. Otherwise, heuristically detects the
    subject as the first voice's notes until the second voice enters.
    """
    parts = list(score.parts)
    if not parts:
        return FugueAnalysis()

    num_voices = len(parts)

    # Convert all parts to FugueNotes
    all_voices: dict[int, list[FugueNote]] = {}
    for i, part in enumerate(parts):
        all_voices[i] = stream_to_fuguenotes(part, voice_idx=i)

    # Detect subject: first voice until second voice enters
    first_voice = all_voices.get(0, [])
    if not first_voice:
        return FugueAnalysis(num_voices=num_voices)

    if subject_end_offset is not None:
        subject_notes = [n for n in first_voice if n.offset < subject_end_offset]
    elif num_voices > 1:
        second_voice = all_voices.get(1, [])
        non_rest_second = [n for n in second_voice if not n.is_rest]
        if non_rest_second:
            subject_end_offset = non_rest_second[0].offset
            subject_notes = [n for n in first_voice if n.offset < subject_end_offset]
        else:
            subject_notes = first_voice[:8]  # fallback
    else:
        subject_notes = first_voice[:8]

    # Detect key
    k = score.analyze("key")
    ks = k.tonic.name
    if k.mode == "minor":
        ks = ks.lower()

    subject = Subject(notes=subject_notes, key_signature=ks)

    # Find occurrences
    occurrences = find_subject_occurrences(subject, all_voices)

    # Detect total duration
    total_dur = max(
        (n.offset + n.duration for notes in all_voices.values() for n in notes),
        default=0.0,
    )

    # Detect sections
    sections = detect_sections(subject, all_voices, total_dur)

    # Detect cadences
    cadences = detect_cadences_simple(all_voices, ks)

    return FugueAnalysis(
        subject=subject,
        subject_occurrences=occurrences,
        sections=sections,
        cadences=cadences,
        num_voices=num_voices,
    )


def analyze_fugue_file(filepath: str) -> FugueAnalysis:
    """Load and analyze a fugue from a file."""
    score = converter.parse(filepath)
    return analyze_fugue(score)
