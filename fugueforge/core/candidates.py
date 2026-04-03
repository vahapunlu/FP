"""
FugueForge Candidate Generation & Scoring

Generates candidate pitches for note-by-note voice generation and scores
them using inline constraint checking.  Hard violations (parallel 5/8,
voice crossing) veto candidates; soft violations reduce score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .representation import FugueNote
from .rules import (
    VOICE_RANGES,
    MotionType,
    classify_motion,
    interval_class,
    is_consonance,
    is_dissonance,
)


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Parameters for the generation process."""
    beam_width: int = 8
    max_candidates_per_step: int = 24
    temperature: float = 0.5
    step_resolution: float = 0.5     # generate in half-beat steps
    prefer_stepwise: float = 0.7     # weight for stepwise motion
    max_leap: int = 12               # max interval in semitones
    consonance_weight: float = 4.0   # bonus for consonant intervals
    dissonance_penalty: float = 8.0  # penalty for dissonance on strong beat
    parallel_veto: bool = True       # hard-reject parallel 5/8/unisons
    crossing_veto: bool = True       # hard-reject voice crossing
    leap_resolution_bonus: float = 12.0
    contrary_motion_bonus: float = 1.5
    voice_ranges: dict[int, tuple[int, int]] = field(
        default_factory=lambda: dict(VOICE_RANGES),
    )


# ---------------------------------------------------------------------------
# Voice state tracker (for inline constraint checking)
# ---------------------------------------------------------------------------

@dataclass
class VoiceState:
    """Tracks the previous pitch per voice for motion detection."""
    prev_pitches: dict[int, int] = field(default_factory=dict)
    curr_pitches: dict[int, int] = field(default_factory=dict)

    def advance(self, voice: int, new_pitch: int) -> None:
        self.prev_pitches[voice] = self.curr_pitches.get(voice, new_pitch)
        self.curr_pitches[voice] = new_pitch

    def set_current(self, voice: int, pitch: int) -> None:
        self.curr_pitches[voice] = pitch


# ---------------------------------------------------------------------------
# Candidate note generation
# ---------------------------------------------------------------------------

def _candidate_pitches(
    prev_pitch: int,
    voice: int,
    config: GenerationConfig,
) -> list[int]:
    """Generate candidate pitches for the next note in a voice."""
    lo, hi = config.voice_ranges.get(voice, (36, 84))
    candidates: list[int] = []

    # Stepwise motion (-2, -1, 0, +1, +2) — highest priority
    for delta in range(-2, 3):
        p = prev_pitch + delta
        if lo <= p <= hi:
            candidates.append(p)

    # Thirds (-3, -4, +3, +4)
    for delta in [-4, -3, 3, 4]:
        p = prev_pitch + delta
        if lo <= p <= hi and p not in candidates:
            candidates.append(p)

    # Larger consonant leaps (P4, P5, m6, M6, P8)
    for delta in [-5, 5, -7, 7, -8, 8, -9, 9, -12, 12]:
        p = prev_pitch + delta
        if lo <= p <= hi and p not in candidates:
            candidates.append(p)

    return candidates


# ---------------------------------------------------------------------------
# Inline constraint checking
# ---------------------------------------------------------------------------

def _check_parallel_motion(
    voice: int,
    candidate: int,
    prev_pitch: int,
    other_voice: int,
    other_prev: int,
    other_curr: int,
) -> bool:
    """
    Return True if placing `candidate` in `voice` creates parallel 5ths,
    8ves, or unisons with `other_voice`. These are HARD vetoes.
    """
    if prev_pitch == candidate and other_prev == other_curr:
        return False
    if prev_pitch == candidate or other_prev == other_curr:
        return False

    int_before = interval_class(prev_pitch - other_prev)
    int_after = interval_class(candidate - other_curr)

    if int_before == 0 and int_after == 0:
        return True
    if int_before == 7 and int_after == 7:
        return True
    if int_before in (0, 7) and int_after == int_before:
        motion = classify_motion(
            min(other_prev, prev_pitch), min(other_curr, candidate),
            max(other_prev, prev_pitch), max(other_curr, candidate),
        )
        if motion == MotionType.PARALLEL:
            return True

    return False


def _check_hidden_fifth_octave(
    voice: int,
    candidate: int,
    prev_pitch: int,
    other_voice: int,
    other_prev: int,
    other_curr: int,
) -> bool:
    """Return True if similar motion into a perfect 5th or octave."""
    if prev_pitch == candidate or other_prev == other_curr:
        return False

    motion = classify_motion(
        min(other_prev, prev_pitch), min(other_curr, candidate),
        max(other_prev, prev_pitch), max(other_curr, candidate),
    )
    if motion != MotionType.SIMILAR:
        return False

    int_after = interval_class(candidate - other_curr)
    return int_after in (0, 7)


def _check_voice_crossing(
    voice: int,
    candidate: int,
    other_voices_current: dict[int, int],
) -> bool:
    """Return True if candidate causes voice crossing."""
    for ov, op in other_voices_current.items():
        if op == -1:
            continue
        if voice < ov and candidate < op:
            return True
        if voice > ov and candidate > op:
            return True
    return False


# ---------------------------------------------------------------------------
# Candidate scoring with full constraint integration
# ---------------------------------------------------------------------------

def _score_candidate(
    candidate: int,
    prev_pitch: int,
    voice: int,
    other_voices_current: dict[int, int],
    other_voices_prev: dict[int, int],
    is_strong_beat: bool,
    prev_interval: int,
    config: GenerationConfig,
) -> tuple[float, bool]:
    """
    Score a candidate pitch. Returns (score, vetoed).
    If vetoed=True, the candidate MUST NOT be used.
    """
    score = 0.0
    vetoed = False

    # === HARD CONSTRAINTS (veto) ===

    # 1. Parallel 5ths/8ves/unisons
    if config.parallel_veto:
        for ov, oc in other_voices_current.items():
            if oc == -1:
                continue
            op = other_voices_prev.get(ov, oc)
            if _check_parallel_motion(voice, candidate, prev_pitch, ov, op, oc):
                vetoed = True
                return -100.0, vetoed

    # 2. Voice crossing
    if config.crossing_veto:
        if _check_voice_crossing(voice, candidate, other_voices_current):
            vetoed = True
            return -100.0, vetoed

    # === SOFT CONSTRAINTS (penalties/bonuses) ===

    mel_interval = candidate - prev_pitch
    abs_interval = abs(mel_interval)

    # 3. Melodic quality
    if abs_interval <= 2:
        score += config.prefer_stepwise * 4.0
    elif abs_interval <= 4:
        score += 2.0
    elif abs_interval == 5 or abs_interval == 7:
        score += 1.0
    elif abs_interval > 7:
        score -= 2.0

    # 4. Leap resolution: after a leap, prefer step in opposite direction
    if abs(prev_interval) >= 5:
        if abs_interval <= 2 and (mel_interval * prev_interval < 0):
            score += config.leap_resolution_bonus
        elif abs_interval <= 2:
            score += config.leap_resolution_bonus * 0.3
        elif abs_interval <= 4:
            score -= 3.0
        else:
            score -= 8.0

    # 5. Augmented/diminished interval check
    if abs_interval == 6:
        score -= 3.0

    # 6. Harmonic quality against each sounding voice
    num_other = len([v for v in other_voices_current.values() if v != -1])
    for ov, op in other_voices_current.items():
        if op == -1:
            continue
        vert = interval_class(candidate - op)

        if vert in (0, 3, 4, 7, 8, 9):
            score += config.consonance_weight
            if vert in (3, 4, 8, 9):
                score += 1.0
        else:
            diss_mult = 1.0 + (num_other - 1) * 0.3
            if is_strong_beat:
                score -= config.dissonance_penalty * diss_mult
            else:
                score -= config.dissonance_penalty * 0.4

    # 7. Contrary motion bonus
    for ov, oc in other_voices_current.items():
        if oc == -1:
            continue
        op = other_voices_prev.get(ov, oc)
        other_motion = oc - op
        if mel_interval != 0 and other_motion != 0:
            if (mel_interval > 0) != (other_motion > 0):
                score += config.contrary_motion_bonus

    # 8. Hidden 5ths/8ves (soft penalty)
    for ov, oc in other_voices_current.items():
        if oc == -1:
            continue
        op = other_voices_prev.get(ov, oc)
        if _check_hidden_fifth_octave(voice, candidate, prev_pitch, ov, op, oc):
            score -= 6.0

    # 9. Avoid repeated note (except for ties/suspensions)
    if candidate == prev_pitch:
        score -= 1.0

    # 10. Slight randomness for variety
    score += random.gauss(0, config.temperature)

    return score, False
