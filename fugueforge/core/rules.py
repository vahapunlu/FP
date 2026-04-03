"""
FugueForge Rule Engine

Species counterpoint and fugue-specific rules checker.
Scores passages for voice-leading violations, forbidden motions,
dissonance handling, tessitura, and cadence quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from music21 import interval, pitch

from .representation import FugueNote


# ---------------------------------------------------------------------------
# Rule violation types
# ---------------------------------------------------------------------------

class ViolationType(Enum):
    PARALLEL_FIFTHS = auto()
    PARALLEL_OCTAVES = auto()
    PARALLEL_UNISONS = auto()
    HIDDEN_FIFTHS = auto()
    HIDDEN_OCTAVES = auto()
    VOICE_CROSSING = auto()
    VOICE_OVERLAP = auto()
    LARGE_LEAP_UNRESOLVED = auto()
    DISSONANCE_UNPREPARED = auto()
    DISSONANCE_UNRESOLVED = auto()
    TESSITURA_EXCEEDED = auto()
    SPACING_TOO_WIDE = auto()
    DIRECT_MOTION_ALL_VOICES = auto()
    AUGMENTED_INTERVAL = auto()
    TRITONE_UNRESOLVED = auto()
    CADENCE_WEAK = auto()


class Severity(Enum):
    ERROR = auto()       # Hard rule, must never happen
    WARNING = auto()     # Soft rule, penalized but not fatal
    INFO = auto()        # Stylistic suggestion


@dataclass
class Violation:
    type: ViolationType
    severity: Severity
    offset: float
    voices: tuple[int, ...]
    message: str = ""
    penalty: float = 0.0


# ---------------------------------------------------------------------------
# Voice ranges (MIDI note numbers)
# ---------------------------------------------------------------------------

VOICE_RANGES: dict[int, tuple[int, int]] = {
    0: (60, 81),  # Soprano: C4 – A5
    1: (53, 74),  # Alto:    F3 – D5
    2: (48, 69),  # Tenor:   C3 – A4
    3: (36, 60),  # Bass:    C2 – C4
}

# For 3-voice, use indices 0, 1, 3 (SA + B) or 0, 2, 3 (S + TB)
# The planner decides which voices are active.

# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

# Perfect consonances
PERFECT_UNISON = 0
PERFECT_FIFTH = 7
PERFECT_OCTAVE = 12

PERFECT_CONSONANCES = {0, 7, 12, 19, 24}  # unison, P5, P8, P12, P15

# Imperfect consonances
IMPERFECT_CONSONANCES = {3, 4, 8, 9, 15, 16}  # m3, M3, m6, M6, ...

# Dissonances
DISSONANCES = {1, 2, 5, 6, 10, 11}  # m2, M2, P4 (above bass), tritone, m7, M7


def interval_class(semitones: int) -> int:
    """Reduce to simple interval (mod 12), preserve sign for motion detection."""
    return abs(semitones) % 12


def is_perfect_consonance(semitones: int) -> bool:
    return interval_class(semitones) in {0, 7}


def is_consonance(semitones: int) -> bool:
    ic = interval_class(semitones)
    return ic in {0, 3, 4, 7, 8, 9}


def is_dissonance(semitones: int) -> bool:
    return not is_consonance(semitones)


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

class MotionType(Enum):
    PARALLEL = auto()
    SIMILAR = auto()
    CONTRARY = auto()
    OBLIQUE = auto()


def classify_motion(
    lower1: int, lower2: int,
    upper1: int, upper2: int,
) -> MotionType:
    """Classify the motion between two voice pairs across two beats."""
    motion_lower = lower2 - lower1
    motion_upper = upper2 - upper1

    if motion_lower == 0 and motion_upper == 0:
        return MotionType.OBLIQUE
    if motion_lower == 0 or motion_upper == 0:
        return MotionType.OBLIQUE
    if motion_lower == motion_upper:
        return MotionType.PARALLEL
    if (motion_lower > 0) == (motion_upper > 0):
        return MotionType.SIMILAR
    return MotionType.CONTRARY


# ---------------------------------------------------------------------------
# Snapshot: a vertical slice of all voices at one beat
# ---------------------------------------------------------------------------

@dataclass
class BeatSnapshot:
    """Pitches sounding at a given offset, indexed by voice."""
    offset: float
    pitches: dict[int, int]  # voice -> MIDI pitch (-1 = rest)


def build_snapshots(
    voices: dict[int, list[FugueNote]],
    resolution: float = 0.5,
) -> list[BeatSnapshot]:
    """
    Build a time-grid of snapshots at the given resolution.
    Each snapshot records which pitch each voice is playing.
    """
    # Find total duration
    max_end = 0.0
    for notes in voices.values():
        for n in notes:
            max_end = max(max_end, n.offset + n.duration)

    snapshots: list[BeatSnapshot] = []
    t = 0.0
    while t < max_end + 0.001:
        pitches: dict[int, int] = {}
        for v, notes in voices.items():
            sounding = -1
            for n in notes:
                if n.offset <= t < n.offset + n.duration:
                    sounding = n.pitch
                    break
            pitches[v] = sounding
        snapshots.append(BeatSnapshot(offset=t, pitches=pitches))
        t += resolution

    return snapshots


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------

class CounterpointRuleChecker:
    """
    Checks a multi-voice passage for counterpoint violations.
    Works on beat snapshots to detect parallel/hidden intervals,
    voice crossing, spacing, tessitura, and dissonance issues.
    """

    def __init__(
        self,
        voice_ranges: Optional[dict[int, tuple[int, int]]] = None,
        resolution: float = 0.5,
        strict: bool = True,
    ):
        self.voice_ranges = voice_ranges or VOICE_RANGES
        self.resolution = resolution
        self.strict = strict

    def check(self, voices: dict[int, list[FugueNote]]) -> list[Violation]:
        """Run all checks and return a list of violations."""
        snapshots = build_snapshots(voices, self.resolution)
        violations: list[Violation] = []

        violations.extend(self._check_parallels(snapshots))
        violations.extend(self._check_hidden_intervals(snapshots))
        violations.extend(self._check_voice_crossing(snapshots))
        violations.extend(self._check_tessitura(voices))
        violations.extend(self._check_spacing(snapshots))
        violations.extend(self._check_dissonance(snapshots))

        return violations

    def score(self, voices: dict[int, list[FugueNote]]) -> float:
        """
        Return a quality score 0..100.
        100 = no violations, lower = more violations.
        Uses diminishing penalty: each violation hurts less as total grows,
        so the score remains informative across quality levels.
        """
        violations = self.check(voices)
        if not violations:
            return 100.0

        # Weighted penalties by severity
        error_penalty = sum(v.penalty for v in violations if v.severity == Severity.ERROR)
        warn_penalty = sum(v.penalty for v in violations if v.severity == Severity.WARNING)
        info_penalty = sum(v.penalty for v in violations if v.severity == Severity.INFO)

        # Count total beats for normalization
        total_beats = 0.0
        for v_notes in voices.values():
            if v_notes:
                total_beats = max(total_beats,
                    max(n.offset + n.duration for n in v_notes))
        num_voices = len(voices)
        total_slots = max(1, total_beats * num_voices)

        # Normalize: violations per beat-voice slot
        error_rate = error_penalty / total_slots
        warn_rate = warn_penalty / total_slots
        info_rate = info_penalty / total_slots

        # Diminishing returns: 100 * e^(-k * rate)
        import math
        score = 100.0 * math.exp(-0.5 * error_rate - 0.15 * warn_rate - 0.05 * info_rate)
        return max(0.0, min(100.0, score))

    # -- Parallel fifths / octaves / unisons --

    def _check_parallels(self, snapshots: list[BeatSnapshot]) -> list[Violation]:
        violations: list[Violation] = []
        voice_ids = sorted(
            set().union(*(s.pitches.keys() for s in snapshots))
        )

        for i in range(len(snapshots) - 1):
            s1, s2 = snapshots[i], snapshots[i + 1]
            for vi_idx in range(len(voice_ids)):
                for vj_idx in range(vi_idx + 1, len(voice_ids)):
                    vi, vj = voice_ids[vi_idx], voice_ids[vj_idx]
                    p1_lo = s1.pitches.get(vj, -1)
                    p1_hi = s1.pitches.get(vi, -1)
                    p2_lo = s2.pitches.get(vj, -1)
                    p2_hi = s2.pitches.get(vi, -1)

                    # Skip rests
                    if -1 in (p1_lo, p1_hi, p2_lo, p2_hi):
                        continue

                    # Skip if no movement
                    if p1_lo == p2_lo and p1_hi == p2_hi:
                        continue

                    int1 = interval_class(p1_hi - p1_lo)
                    int2 = interval_class(p2_hi - p2_lo)
                    motion = classify_motion(p1_lo, p2_lo, p1_hi, p2_hi)

                    if motion == MotionType.PARALLEL:
                        if int1 == int2 == 7:  # parallel fifths
                            violations.append(Violation(
                                type=ViolationType.PARALLEL_FIFTHS,
                                severity=Severity.ERROR,
                                offset=s2.offset,
                                voices=(vi, vj),
                                message=f"Parallel 5ths at beat {s2.offset}",
                                penalty=10.0,
                            ))
                        elif int1 == int2 == 0:  # parallel unisons
                            violations.append(Violation(
                                type=ViolationType.PARALLEL_UNISONS,
                                severity=Severity.ERROR,
                                offset=s2.offset,
                                voices=(vi, vj),
                                message=f"Parallel unisons at beat {s2.offset}",
                                penalty=10.0,
                            ))
                        elif int1 == int2 and int1 in (0, 12):  # parallel octaves
                            violations.append(Violation(
                                type=ViolationType.PARALLEL_OCTAVES,
                                severity=Severity.ERROR,
                                offset=s2.offset,
                                voices=(vi, vj),
                                message=f"Parallel 8ves at beat {s2.offset}",
                                penalty=10.0,
                            ))

        return violations

    # -- Hidden (direct) fifths / octaves --

    def _check_hidden_intervals(self, snapshots: list[BeatSnapshot]) -> list[Violation]:
        violations: list[Violation] = []
        voice_ids = sorted(
            set().union(*(s.pitches.keys() for s in snapshots))
        )

        for i in range(len(snapshots) - 1):
            s1, s2 = snapshots[i], snapshots[i + 1]
            for vi_idx in range(len(voice_ids)):
                for vj_idx in range(vi_idx + 1, len(voice_ids)):
                    vi, vj = voice_ids[vi_idx], voice_ids[vj_idx]
                    p1_lo = s1.pitches.get(vj, -1)
                    p1_hi = s1.pitches.get(vi, -1)
                    p2_lo = s2.pitches.get(vj, -1)
                    p2_hi = s2.pitches.get(vi, -1)

                    if -1 in (p1_lo, p1_hi, p2_lo, p2_hi):
                        continue

                    int2 = interval_class(p2_hi - p2_lo)
                    motion = classify_motion(p1_lo, p2_lo, p1_hi, p2_hi)

                    if motion == MotionType.SIMILAR and int2 in (0, 7):
                        # Hidden 5th or 8ve: similar motion into a perfect interval
                        vtype = (ViolationType.HIDDEN_FIFTHS
                                 if int2 == 7
                                 else ViolationType.HIDDEN_OCTAVES)
                        violations.append(Violation(
                            type=vtype,
                            severity=Severity.WARNING,
                            offset=s2.offset,
                            voices=(vi, vj),
                            message=f"Hidden {'5th' if int2 == 7 else '8ve'} at beat {s2.offset}",
                            penalty=3.0,
                        ))

        return violations

    # -- Voice crossing --

    def _check_voice_crossing(self, snapshots: list[BeatSnapshot]) -> list[Violation]:
        violations: list[Violation] = []
        for snap in snapshots:
            voice_ids = sorted(snap.pitches.keys())
            for i in range(len(voice_ids) - 1):
                vi, vj = voice_ids[i], voice_ids[i + 1]
                pi = snap.pitches.get(vi, -1)
                pj = snap.pitches.get(vj, -1)
                if pi == -1 or pj == -1:
                    continue
                # Higher-numbered voice should be lower
                if pi < pj:
                    violations.append(Violation(
                        type=ViolationType.VOICE_CROSSING,
                        severity=Severity.WARNING,
                        offset=snap.offset,
                        voices=(vi, vj),
                        message=f"Voice crossing v{vi}/v{vj} at beat {snap.offset}",
                        penalty=2.0,
                    ))
        return violations

    # -- Tessitura --

    def _check_tessitura(self, voices: dict[int, list[FugueNote]]) -> list[Violation]:
        violations: list[Violation] = []
        for v, notes in voices.items():
            lo, hi = self.voice_ranges.get(v, (0, 127))
            for n in notes:
                if n.is_rest:
                    continue
                if n.pitch < lo or n.pitch > hi:
                    violations.append(Violation(
                        type=ViolationType.TESSITURA_EXCEEDED,
                        severity=Severity.WARNING,
                        offset=n.offset,
                        voices=(v,),
                        message=f"Voice {v} pitch {n.pitch} outside range [{lo},{hi}]",
                        penalty=2.0,
                    ))
        return violations

    # -- Spacing --

    def _check_spacing(self, snapshots: list[BeatSnapshot]) -> list[Violation]:
        """Adjacent upper voices should not be more than an octave apart."""
        violations: list[Violation] = []
        for snap in snapshots:
            voice_ids = sorted(snap.pitches.keys())
            for i in range(len(voice_ids) - 1):
                vi, vj = voice_ids[i], voice_ids[i + 1]
                pi = snap.pitches.get(vi, -1)
                pj = snap.pitches.get(vj, -1)
                if pi == -1 or pj == -1:
                    continue
                gap = abs(pi - pj)
                # Bass can be farther from tenor, but upper voices <= octave
                if vj < 3 and gap > 12:
                    violations.append(Violation(
                        type=ViolationType.SPACING_TOO_WIDE,
                        severity=Severity.INFO,
                        offset=snap.offset,
                        voices=(vi, vj),
                        message=f"Spacing {gap} semitones between v{vi}/v{vj}",
                        penalty=1.0,
                    ))
        return violations

    # -- Dissonance handling --

    def _check_dissonance(self, snapshots: list[BeatSnapshot]) -> list[Violation]:
        """Flag dissonances on strong beats that don't resolve."""
        violations: list[Violation] = []
        for idx, snap in enumerate(snapshots):
            voice_ids = sorted(snap.pitches.keys())
            for i in range(len(voice_ids)):
                for j in range(i + 1, len(voice_ids)):
                    vi, vj = voice_ids[i], voice_ids[j]
                    pi = snap.pitches.get(vi, -1)
                    pj = snap.pitches.get(vj, -1)
                    if pi == -1 or pj == -1:
                        continue
                    semitones = abs(pi - pj)
                    if is_dissonance(semitones):
                        # Check if on a strong beat (offset is integer)
                        if snap.offset == int(snap.offset):
                            # Check if next beat resolves
                            resolved = False
                            if idx + 1 < len(snapshots):
                                ns = snapshots[idx + 1]
                                npi = ns.pitches.get(vi, -1)
                                npj = ns.pitches.get(vj, -1)
                                if npi != -1 and npj != -1:
                                    if is_consonance(abs(npi - npj)):
                                        resolved = True
                            if not resolved:
                                violations.append(Violation(
                                    type=ViolationType.DISSONANCE_UNRESOLVED,
                                    severity=Severity.ERROR if self.strict else Severity.WARNING,
                                    offset=snap.offset,
                                    voices=(vi, vj),
                                    message=f"Unresolved dissonance ({semitones}st) at beat {snap.offset}",
                                    penalty=5.0,
                                ))
        return violations


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def check_counterpoint(
    voices: dict[int, list[FugueNote]],
    strict: bool = True,
    voice_ranges: Optional[dict[int, tuple[int, int]]] = None,
) -> tuple[float, list[Violation]]:
    """
    Check multi-voice passage and return (score, violations).
    Score is 0..100 where 100 = clean.
    """
    checker = CounterpointRuleChecker(strict=strict, voice_ranges=voice_ranges)
    violations = checker.check(voices)
    score = checker.score(voices)
    return score, violations
