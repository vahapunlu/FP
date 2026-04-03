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
    temperature: float = 0.4
    step_resolution: float = 0.5     # generate in half-beat steps
    prefer_stepwise: float = 0.7     # weight for stepwise motion
    max_leap: int = 12               # max interval in semitones
    consonance_weight: float = 3.0   # bonus for consonant intervals (was 4.0)
    dissonance_penalty: float = 8.0  # penalty for dissonance on strong beat
    parallel_veto: bool = True       # hard-reject parallel 5/8/unisons
    crossing_veto: bool = True       # hard-reject voice crossing
    leap_resolution_bonus: float = 12.0
    contrary_motion_bonus: float = 1.5
    scale_bonus: float = 8.0         # bonus for in-scale pitches (was 10.0)
    chromatic_penalty: float = 8.0   # penalty for out-of-scale pitches
    chord_tone_bonus: float = 5.0    # bonus for chord tones (was 6.0)
    non_chord_penalty: float = 3.0   # penalty for non-chord tones on strong beats
    scale_pcs: frozenset[int] = frozenset()  # pitch classes in the current key
    voice_ranges: dict[int, tuple[int, int]] = field(
        default_factory=lambda: dict(VOICE_RANGES),
    )
    # --- Phase 2: Melodic quality ---
    stepwise_bonus: float = 8.0       # strong bonus for steps (1-2 semitones)
    third_bonus: float = 3.5          # moderate bonus for thirds
    fourth_fifth_penalty: float = 1.0 # slight penalty for P4/P5 leaps
    large_leap_penalty: float = 8.0   # strong penalty for >P5
    consecutive_leap_penalty: float = 10.0  # penalty for 2+ leaps in a row
    direction_momentum: float = 3.0   # bonus for continuing melodic direction (2-3 notes)
    direction_reversal: float = 4.0   # bonus for reversing after 4+ notes same direction
    register_gravity: float = 2.0     # pull toward voice center
    phrase_length: float = 4.0        # phrase length in beats for contour shaping


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
    chord_pcs: frozenset[int] = frozenset(),
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

    # 3. Melodic quality — STRONG stepwise preference (Phase 2)
    if abs_interval == 0:
        score += 1.0  # repetition: mild
    elif abs_interval <= 2:
        score += config.stepwise_bonus  # +8: steps are the backbone of melody
    elif abs_interval <= 4:
        score += config.third_bonus     # +3.5: thirds are good melodic intervals
    elif abs_interval == 5 or abs_interval == 7:
        score -= config.fourth_fifth_penalty  # -1: acceptable but less common
    elif abs_interval > 7:
        score -= config.large_leap_penalty    # -8: large leaps are rare in Bach

    # 4. Leap resolution: after a leap, MUST resolve by step in opposite direction
    if abs(prev_interval) >= 5:
        if abs_interval <= 2 and (mel_interval * prev_interval < 0):
            score += config.leap_resolution_bonus      # +12: correct resolution
        elif abs_interval <= 2:
            score += config.leap_resolution_bonus * 0.3  # step same dir OK-ish
        elif abs_interval <= 4:
            score -= 5.0   # third after leap: poor
        else:
            score -= 12.0  # leap after leap: very bad

    # 4b. Consecutive leap penalty: 2+ leaps in a row sound mechanical
    if abs(prev_interval) >= 3 and abs_interval >= 3:
        score -= config.consecutive_leap_penalty  # -10: consecutive leaps
        # Even worse if same direction (zigzag slightly better than staircase)
        if (mel_interval > 0) == (prev_interval > 0):
            score -= config.consecutive_leap_penalty * 0.5  # -5 extra

    # 5. Augmented/diminished interval check
    if abs_interval == 6:
        score -= 5.0  # tritone leap: avoid

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
                score -= config.dissonance_penalty * 0.75

    # 7. Contrary motion bonus
    for ov, oc in other_voices_current.items():
        if oc == -1:
            continue
        op = other_voices_prev.get(ov, oc)
        other_motion = oc - op
        if mel_interval != 0 and other_motion != 0:
            if (mel_interval > 0) != (other_motion > 0):
                score += config.contrary_motion_bonus

    # 8. Scale/tonality: strong bonus for in-key pitches, penalty for chromatic
    if config.scale_pcs:
        candidate_pc = candidate % 12
        if candidate_pc in config.scale_pcs:
            score += config.scale_bonus
        else:
            # Allow chromatic passing tones on weak beats only
            if is_strong_beat:
                score -= config.chromatic_penalty * 1.5
            else:
                # Weak-beat chromatic is tolerable only if stepwise
                if abs_interval <= 2:
                    score -= config.chromatic_penalty * 0.3
                else:
                    score -= config.chromatic_penalty

    # 8b. Chord tone awareness: bonus for hitting the current chord
    if chord_pcs:
        candidate_pc = candidate % 12
        if candidate_pc in chord_pcs:
            if is_strong_beat:
                score += config.chord_tone_bonus * 1.5  # Strong beat chord tone
            else:
                score += config.chord_tone_bonus * 0.5  # Weak beat chord tone
        else:
            # Non-chord tone on strong beat — penalize unless it's a step
            # (passing tone / neighbor tone are OK on weak beats)
            if is_strong_beat:
                score -= config.non_chord_penalty

    # 9. Hidden 5ths/8ves (soft penalty)
    for ov, oc in other_voices_current.items():
        if oc == -1:
            continue
        op = other_voices_prev.get(ov, oc)
        if _check_hidden_fifth_octave(voice, candidate, prev_pitch, ov, op, oc):
            score -= 6.0

    # 9. Avoid repeated note (except for ties/suspensions)
    if candidate == prev_pitch:
        score -= 2.0

    # 10. Register gravity: pull toward voice center to prevent drifting
    lo, hi = config.voice_ranges.get(voice, (36, 84))
    center = (lo + hi) / 2
    dist_from_center = abs(candidate - center)
    range_half = (hi - lo) / 2
    if range_half > 0:
        # Penalty increases as pitch moves toward range extremes
        score -= config.register_gravity * (dist_from_center / range_half) ** 2

    # 11. Slight randomness for variety (reduced from before)
    score += random.gauss(0, config.temperature)

    return score, False
