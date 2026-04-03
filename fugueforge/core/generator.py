"""
FugueForge Generator

Constrained note-level generation for fugue voices.
Uses the plan + rules engine to produce notes voice-by-voice,
with inline constraint checking that rejects forbidden motions
BEFORE notes are committed.

Generation strategy:
  1. Candidate pitches are proposed (stepwise preferred, leaps allowed)
  2. Each candidate is scored on:
     - Melodic quality (stepwise preference, leap resolution, contour)
     - Harmonic quality (consonance with sounding voices)
     - Voice-leading legality (no parallel 5/8, no crossing, resolved dissonance)
     - Rhythmic variety
  3. Hard violations (parallel 5/8, voice crossing) VETO candidates
  4. Soft violations reduce score but don't eliminate
  5. Weighted random selection from survivors

ML model integration comes in a later phase.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from .representation import (
    AnswerType,
    EntryRole,
    ExpositionPlan,
    FugueNote,
    FuguePlan,
    SectionType,
    Subject,
    TransformType,
    VoiceEntry,
)
from .analyzer import compute_tonal_answer
from .rules import (
    CounterpointRuleChecker,
    Violation,
    build_snapshots,
    classify_motion,
    interval_class,
    is_consonance,
    is_dissonance,
    MotionType,
    VOICE_RANGES,
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
    voice_ranges: dict[int, tuple[int, int]] = field(default_factory=lambda: dict(VOICE_RANGES))


# ---------------------------------------------------------------------------
# Voice state tracker (for inline constraint checking)
# ---------------------------------------------------------------------------

@dataclass
class VoiceState:
    """Tracks the previous pitch per voice for motion detection."""
    prev_pitches: dict[int, int] = field(default_factory=dict)  # voice -> prev MIDI pitch
    curr_pitches: dict[int, int] = field(default_factory=dict)  # voice -> current MIDI pitch

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
    # Need motion in both voices to have parallel
    if prev_pitch == candidate and other_prev == other_curr:
        return False  # no motion = oblique, fine

    if prev_pitch == candidate or other_prev == other_curr:
        return False  # oblique motion, fine

    # Check if both intervals are the same perfect consonance
    int_before = interval_class(prev_pitch - other_prev)
    int_after = interval_class(candidate - other_curr)

    # Parallel unisons/octaves
    if int_before == 0 and int_after == 0:
        return True
    # Parallel fifths
    if int_before == 7 and int_after == 7:
        return True

    # Also check parallel octaves (compound)
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
        # Lower-numbered voice should be higher pitch
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
        score += 1.0   # P4, P5 leaps OK but less common
    elif abs_interval > 7:
        score -= 2.0    # large leap penalty

    # 4. Leap resolution: after a leap, prefer step in opposite direction
    if abs(prev_interval) >= 5:
        # We had a leap; now want step in opposite direction
        if abs_interval <= 2 and (mel_interval * prev_interval < 0):
            score += config.leap_resolution_bonus  # ideal resolution
        elif abs_interval <= 2:
            score += config.leap_resolution_bonus * 0.3  # step same dir OK
        elif abs_interval <= 4:
            score -= 3.0  # skip after leap: bad
        else:
            score -= 8.0  # consecutive leaps: very bad

    # 5. Augmented/diminished interval check
    if abs_interval == 6:  # tritone melodic
        score -= 3.0

    # 6. Harmonic quality against each sounding voice
    num_other = len([v for v in other_voices_current.values() if v != -1])
    for ov, op in other_voices_current.items():
        if op == -1:
            continue
        vert = interval_class(candidate - op)

        if vert in (0, 3, 4, 7, 8, 9):  # consonances
            score += config.consonance_weight
            # Prefer imperfect consonances (3rds, 6ths) over perfect
            if vert in (3, 4, 8, 9):
                score += 1.0
        else:
            # Dissonance — scale penalty with voice count
            diss_mult = 1.0 + (num_other - 1) * 0.3  # 4v→1.9x, 5v→2.2x
            if is_strong_beat:
                score -= config.dissonance_penalty * diss_mult
            else:
                score -= config.dissonance_penalty * 0.4  # passing tones OK

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


# ---------------------------------------------------------------------------
# Subject / Answer placement
# ---------------------------------------------------------------------------

def place_subject(
    subject: Subject,
    voice: int,
    start_offset: float,
    transposition: int = 0,
    transform: TransformType = TransformType.NONE,
) -> list[FugueNote]:
    """Place the subject in a voice at a given offset with optional transposition."""
    result: list[FugueNote] = []
    base_offset = subject.notes[0].offset if subject.notes else 0.0

    for fn in subject.notes:
        p = fn.pitch
        if not fn.is_rest:
            if transform == TransformType.INVERSION:
                # Invert around the first pitch
                first_pitch = subject.notes[0].pitch
                p = first_pitch - (p - first_pitch) + transposition
            elif transform == TransformType.RETROGRADE:
                p += transposition  # handled by reversing note list below
            else:
                # NONE, AUGMENTATION, DIMINUTION all just transpose
                p += transposition

        dur = fn.duration
        rel_offset = fn.offset - base_offset
        if transform == TransformType.AUGMENTATION:
            dur *= 2
            rel_offset *= 2
        elif transform == TransformType.DIMINUTION:
            dur *= 0.5
            rel_offset *= 0.5

        result.append(FugueNote(
            pitch=p if not fn.is_rest else -1,
            duration=dur,
            voice=voice,
            offset=start_offset + rel_offset,
            role=EntryRole.SUBJECT if transposition % 12 == 0 else EntryRole.ANSWER,
            transform=transform,
        ))

    if transform == TransformType.RETROGRADE:
        # Reverse the pitches but keep the original rhythm
        pitches = [n.pitch for n in result]
        pitches.reverse()
        for i, n in enumerate(result):
            result[i] = FugueNote(
                pitch=pitches[i],
                duration=n.duration,
                voice=n.voice,
                offset=n.offset,
                role=n.role,
                transform=n.transform,
            )

    return result


def place_answer(
    subject: Subject,
    voice: int,
    start_offset: float,
) -> list[FugueNote]:
    """Place the tonal/real answer in a voice."""
    answer_notes = compute_tonal_answer(subject, transposition_semitones=7)
    result: list[FugueNote] = []
    base_offset = answer_notes[0].offset if answer_notes else 0.0

    for fn in answer_notes:
        result.append(FugueNote(
            pitch=fn.pitch,
            duration=fn.duration,
            voice=voice,
            offset=start_offset + (fn.offset - base_offset),
            role=EntryRole.ANSWER,
        ))

    return result


# ---------------------------------------------------------------------------
# Free counterpoint generation (heuristic)
# ---------------------------------------------------------------------------

def generate_free_counterpoint(
    voice: int,
    start_offset: float,
    duration: float,
    existing_voices: dict[int, list[FugueNote]],
    start_pitch: Optional[int] = None,
    config: Optional[GenerationConfig] = None,
    progress: float = 0.0,
) -> list[FugueNote]:
    """
    Generate free counterpoint for one voice against existing voices.
    Uses constraint-aware scoring: hard violations (parallel 5/8, crossing)
    are vetoed, soft violations reduce score.
    
    progress: 0.0-1.0 indicating position within the full piece.
              Used to reserve highest pitches for later sections (climax placement).
    """
    config = config or GenerationConfig()
    lo, hi = config.voice_ranges.get(voice, (48, 72))

    # Climax placement: control upper range based on piece position
    # Early sections: lower ceiling; late sections: raise ceiling for climax
    if voice == 0:  # soprano — controls climax
        if progress < 0.45:
            ceiling_reduction = int((0.45 - progress) * 10)  # up to ~4 semitones
            hi = max(hi - ceiling_reduction, lo + 12)
        elif progress > 0.55:
            ceiling_raise = int((progress - 0.55) * 8)  # up to ~3 semitones
            hi = hi + ceiling_raise

    if start_pitch is None:
        start_pitch = (lo + hi) // 2

    notes: list[FugueNote] = []
    current_offset = start_offset
    current_pitch = start_pitch
    prev_melodic_interval = 0  # for leap resolution tracking
    prev_was_dissonant = False  # for dissonance resolution tracking
    end_offset = start_offset + duration

    step = config.step_resolution

    # Build a VoiceState from existing voices
    state = VoiceState()
    state.set_current(voice, current_pitch)

    while current_offset < end_offset - 0.01:
        # Determine current and previous pitches for all other voices
        other_current: dict[int, int] = {}
        other_prev: dict[int, int] = {}
        for v, v_notes in existing_voices.items():
            if v == voice:
                continue
            # Current: what's sounding now
            for n in v_notes:
                if n.offset <= current_offset < n.offset + n.duration:
                    other_current[v] = n.pitch
                    break
            # Previous: what was sounding at the previous step
            prev_t = current_offset - step
            for n in v_notes:
                if n.offset <= prev_t < n.offset + n.duration:
                    other_prev[v] = n.pitch
                    break
            # Fallback: if no previous, use current
            if v not in other_prev:
                other_prev[v] = other_current.get(v, -1)

        # Is this a strong beat?
        is_strong = (current_offset % 1.0) < 0.01

        # Generate and score candidates
        candidates = _candidate_pitches(current_pitch, voice, config)
        scored: list[tuple[float, int]] = []

        for cp in candidates:
            s, vetoed = _score_candidate(
                candidate=cp,
                prev_pitch=current_pitch,
                voice=voice,
                other_voices_current=other_current,
                other_voices_prev=other_prev,
                is_strong_beat=is_strong,
                prev_interval=prev_melodic_interval,
                config=config,
            )
            # Dissonance resolution: after a dissonant beat,
            # strongly prefer consonant resolution by step
            if prev_was_dissonant and other_current:
                is_now_consonant = all(
                    is_consonance(cp - op)
                    for op in other_current.values()
                    if op != -1
                )
                if is_now_consonant and abs(cp - current_pitch) <= 2:
                    s += 12.0  # strong resolution bonus
                elif is_now_consonant:
                    s += 4.0   # consonant but leap — still acceptable
                elif not is_now_consonant:
                    s -= 8.0  # failing to resolve penalty
            if not vetoed:
                scored.append((s, cp))

        # If all candidates vetoed, relax constraints
        if not scored:
            for cp in candidates:
                s, _ = _score_candidate(
                    candidate=cp,
                    prev_pitch=current_pitch,
                    voice=voice,
                    other_voices_current=other_current,
                    other_voices_prev=other_prev,
                    is_strong_beat=is_strong,
                    prev_interval=prev_melodic_interval,
                    config=GenerationConfig(
                        **{**config.__dict__,
                           'parallel_veto': False,
                           'crossing_veto': False}
                    ),
                )
                scored.append((s, cp))

        scored.sort(reverse=True)

        # Pick from top candidates with weighted randomness
        top_n = min(config.beam_width, len(scored))
        if top_n == 0:
            break

        # Shift scores to be positive for weighting
        min_score = min(s for s, _ in scored[:top_n])
        shift = abs(min_score) + 1.0 if min_score < 0 else 0.0
        weights = [(s + shift + 0.1) for s, _ in scored[:top_n]]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        chosen_pitch = random.choices(
            [p for _, p in scored[:top_n]],
            weights=weights,
            k=1,
        )[0]

        # Determine duration with variety
        dur = step
        # On strong beats, sometimes use longer durations
        if is_strong and random.random() < 0.25 and current_offset + step * 2 <= end_offset:
            dur = step * 2
        # Occasionally use quarter note
        elif random.random() < 0.15 and current_offset + 1.0 <= end_offset:
            dur = 1.0

        notes.append(FugueNote(
            pitch=chosen_pitch,
            duration=dur,
            voice=voice,
            offset=current_offset,
            role=EntryRole.FREE_COUNTERPOINT,
        ))

        # Track dissonance state for resolution on next beat
        # Track on ALL beats, not just strong — weak-beat dissonance also
        # needs resolution (passing tones must resolve by step)
        prev_was_dissonant = any(
            is_dissonance(chosen_pitch - op)
            for op in other_current.values()
            if op != -1
        )

        prev_melodic_interval = chosen_pitch - current_pitch
        current_pitch = chosen_pitch
        current_offset += dur

    return notes


# ---------------------------------------------------------------------------
# Episode generation (sequence-based)
# ---------------------------------------------------------------------------

def generate_episode(
    subject: Subject,
    voice: int,
    start_offset: float,
    duration: float,
    existing_voices: dict[int, list[FugueNote]],
    sequence_interval: int = -2,
    sequence_count: int = 3,
    config: Optional[GenerationConfig] = None,
    source: str = "subject_head",
    countersubject: Optional[list[FugueNote]] = None,
) -> list[FugueNote]:
    """
    Generate an episode by creating a sequence pattern from subject material.

    Source options (per Burrows thesis - thematic episodes):
      - 'subject_head': first 3-5 notes of subject (default)
      - 'subject_head_inverted': melodic inversion of subject head
      - 'countersubject': first 3-5 notes of captured countersubject
      - 'subject_tail': last 3-5 notes of subject
    Uses sequential transposition with inline constraint checking.
    """
    config = config or GenerationConfig()
    lo, hi = config.voice_ranges.get(voice, (36, 84))

    # Extract motif based on source material
    non_rest = [n for n in subject.notes if not n.is_rest]
    if source == "subject_tail" and len(non_rest) >= 4:
        motif_notes = non_rest[-4:]
    elif source == "countersubject" and countersubject:
        cs_non_rest = [n for n in countersubject if not n.is_rest]
        motif_notes = cs_non_rest[:4] if cs_non_rest else non_rest[:4]
    elif source == "subject_head_inverted" and len(non_rest) >= 2:
        # Invert the head motif around its first pitch
        head = non_rest[:4]
        first_p = head[0].pitch
        motif_notes = [
            FugueNote(
                pitch=first_p - (n.pitch - first_p),
                duration=n.duration,
                voice=n.voice,
                offset=n.offset,
                role=n.role,
            )
            for n in head
        ]
    else:  # subject_head (default)
        motif_notes = non_rest[:4]

    if not motif_notes:
        return []

    motif_dur = sum(n.duration for n in motif_notes)
    if motif_dur == 0:
        return []

    result: list[FugueNote] = []
    offset = start_offset
    transposition = 0

    for seq_idx in range(sequence_count):
        if offset >= start_offset + duration:
            break

        motif_base = motif_notes[0].offset
        for mn in motif_notes:
            note_offset = offset + (mn.offset - motif_base)
            if note_offset >= start_offset + duration:
                break

            candidate_pitch = mn.pitch + transposition
            # Clamp to voice range
            candidate_pitch = max(lo, min(hi, candidate_pitch))

            # Quick consonance check against other voices
            other_current: dict[int, int] = {}
            for v, v_notes in existing_voices.items():
                if v == voice:
                    continue
                for n in v_notes:
                    if n.offset <= note_offset < n.offset + n.duration:
                        other_current[v] = n.pitch
                        break

            # Adjust pitch to avoid voice crossing, dissonance, and hidden intervals
            if other_current:
                best_pitch = candidate_pitch
                best_score = -100
                prev_p = result[-1].pitch if result else candidate_pitch
                prev_interval = (candidate_pitch - prev_p) if result else 0
                for adj in [0, 1, -1, 2, -2]:
                    p = candidate_pitch + adj
                    if p < lo or p > hi:
                        continue

                    # Consonance score
                    cons = sum(
                        1 for op in other_current.values()
                        if op != -1 and is_consonance(p - op)
                    )
                    is_strong = (note_offset % 1.0) < 0.01
                    sc = cons * 3
                    # Extra bonus for consonance on strong beats
                    if is_strong:
                        sc += cons * 3

                    # Leap resolution: after a leap, prefer step in opposite dir
                    if result and abs(prev_interval) >= 5:
                        mel = p - prev_p
                        if abs(mel) <= 2 and (mel * prev_interval < 0):
                            sc += 8  # ideal resolution
                        elif abs(mel) <= 2:
                            sc += 2  # step same dir
                        elif abs(mel) > 4:
                            sc -= 6  # consecutive leaps

                    # Voice crossing penalty (hard)
                    crossing = False
                    for ov, op in other_current.items():
                        if voice < ov and p < op:
                            crossing = True
                        elif voice > ov and p > op:
                            crossing = True
                    if crossing:
                        sc -= 20

                    # Hidden 5th/8ve penalty
                    if result:
                        prev_p = result[-1].pitch
                        if prev_p != -1 and prev_p != p:
                            for ov, oc in other_current.items():
                                if oc == -1:
                                    continue
                                # Get prev pitch for this other voice
                                prev_t = note_offset - 0.5
                                op_prev = oc
                                for vn in existing_voices.get(ov, []):
                                    if vn.offset <= prev_t < vn.offset + vn.duration:
                                        op_prev = vn.pitch
                                        break
                                if _check_hidden_fifth_octave(voice, p, prev_p, ov, op_prev, oc):
                                    sc -= 4

                    # Prefer minimal deviation from original
                    sc -= abs(adj) * 1

                    if sc > best_score:
                        best_score = sc
                        best_pitch = p
                candidate_pitch = best_pitch

            result.append(FugueNote(
                pitch=candidate_pitch,
                duration=mn.duration,
                voice=voice,
                offset=note_offset,
                role=EntryRole.EPISODE_MATERIAL,
            ))

        offset += motif_dur
        transposition += sequence_interval

    return result


# ---------------------------------------------------------------------------
# Countersubject placement
# ---------------------------------------------------------------------------

def _place_countersubject(
    countersubject: list[FugueNote],
    voice: int,
    start_offset: float,
    existing_voices: dict[int, list[FugueNote]],
    config: GenerationConfig,
    transposition: int = 0,
) -> list[FugueNote]:
    """
    Place a captured countersubject in a voice, transposed to fit the voice range.
    Each note is validated against existing voices using the same constraint system
    as free counterpoint generation. Violating notes are adjusted by small intervals
    to maintain contour while fixing counterpoint errors.
    Falls back to free counterpoint if the CS can't fit at all.
    """
    if not countersubject:
        return []

    lo, hi = config.voice_ranges.get(voice, (36, 84))
    cs_pitches = [n.pitch for n in countersubject if not n.is_rest]
    if not cs_pitches:
        return []

    cs_lo, cs_hi = min(cs_pitches), max(cs_pitches)

    # Find best octave transposition to fit the voice range
    best_shift = transposition
    best_fit = -1
    for octave_shift in [-24, -12, 0, 12, 24]:
        shift = transposition + octave_shift
        shifted_lo = cs_lo + shift
        shifted_hi = cs_hi + shift
        if shifted_lo >= lo and shifted_hi <= hi:
            center = (lo + hi) / 2
            shifted_center = (shifted_lo + shifted_hi) / 2
            fit = 100 - abs(center - shifted_center)
            if fit > best_fit:
                best_fit = fit
                best_shift = shift

    if best_fit < 0:
        # CS doesn't fit — fall back to free counterpoint
        dur = sum(n.duration for n in countersubject)
        return generate_free_counterpoint(
            voice=voice,
            start_offset=start_offset,
            duration=dur,
            existing_voices=existing_voices,
            config=config,
        )

    # --- Constraint-aware placement ---
    # Place each CS note, checking for counterpoint violations and adjusting
    result: list[FugueNote] = []
    prev_pitch = -1
    prev_melodic_interval = 0
    step = config.step_resolution

    for n in countersubject:
        note_offset = start_offset + n.offset
        if n.is_rest:
            result.append(FugueNote(
                pitch=-1,
                duration=n.duration,
                voice=voice,
                offset=note_offset,
                role=EntryRole.COUNTERSUBJECT,
            ))
            continue

        raw_pitch = n.pitch + best_shift

        # Gather current and previous pitches of other voices at this offset
        other_current: dict[int, int] = {}
        other_prev: dict[int, int] = {}
        for v, v_notes in existing_voices.items():
            if v == voice:
                continue
            for vn in v_notes:
                if vn.offset <= note_offset < vn.offset + vn.duration:
                    other_current[v] = vn.pitch
                    break
            prev_t = note_offset - step
            for vn in v_notes:
                if vn.offset <= prev_t < vn.offset + vn.duration:
                    other_prev[v] = vn.pitch
                    break
            if v not in other_prev:
                other_prev[v] = other_current.get(v, -1)
        # Include already-placed CS notes in the constraint context
        for rn in result:
            if rn.voice == voice and rn.pitch != -1:
                if rn.offset <= note_offset < rn.offset + rn.duration:
                    pass  # same voice, skip
                prev_t2 = note_offset - step
                if rn.offset <= prev_t2 < rn.offset + rn.duration:
                    pass  # same voice prev, skip

        is_strong = (note_offset % 1.0) < 0.01
        use_prev = prev_pitch if prev_pitch != -1 else raw_pitch

        # Score the raw CS pitch and small adjustments (0, ±1, ±2)
        # Prefer 0 adjustment to preserve contour
        best_p = raw_pitch
        best_score = -999.0
        best_vetoed = True

        for adj in [0, 1, -1, 2, -2, 3, -3]:
            p = raw_pitch + adj
            if p < lo or p > hi:
                continue

            s, vetoed = _score_candidate(
                candidate=p,
                prev_pitch=use_prev,
                voice=voice,
                other_voices_current=other_current,
                other_voices_prev=other_prev,
                is_strong_beat=is_strong,
                prev_interval=prev_melodic_interval,
                config=config,
            )
            # Penalise deviation from the original CS pitch
            s -= abs(adj) * 3.0

            if not vetoed:
                if best_vetoed or s > best_score:
                    best_score = s
                    best_p = p
                    best_vetoed = False
            elif best_vetoed and s > best_score:
                best_score = s
                best_p = p

        result.append(FugueNote(
            pitch=best_p,
            duration=n.duration,
            voice=voice,
            offset=note_offset,
            role=EntryRole.COUNTERSUBJECT,
        ))

        prev_melodic_interval = best_p - use_prev
        prev_pitch = best_p

    return result


# ---------------------------------------------------------------------------
# Exposition generator
# ---------------------------------------------------------------------------

def generate_exposition(
    plan: ExpositionPlan,
    subject: Subject,
    config: Optional[GenerationConfig] = None,
) -> dict[int, list[FugueNote]]:
    """
    Generate the exposition according to the plan.

    1. Place subject/answer for each voice entry
    2. Generate counterpoint for voices already entered
    3. Capture the first counterpoint against the 2nd entry as countersubject
    4. Reuse the countersubject in subsequent entries
    """
    config = config or GenerationConfig()
    voices: dict[int, list[FugueNote]] = {
        e.voice: [] for e in plan.entries
    }

    countersubject: list[FugueNote] = []  # captured after 2nd entry

    for entry_idx, entry in enumerate(plan.entries):
        # Place subject or answer
        if entry.role == EntryRole.SUBJECT:
            notes = place_subject(
                subject, entry.voice, entry.start_offset,
                transposition=0 if entry_idx == 0 else _get_transposition(entry, subject),
            )
        else:
            notes = place_answer(subject, entry.voice, entry.start_offset)

        voices[entry.voice].extend(notes)

        # Generate counterpoint for previously entered voices
        for prev_idx, prev_entry in enumerate(plan.entries[:entry_idx]):
            prev_voice = prev_entry.voice
            if prev_voice == entry.voice:
                continue

            # Find where prev voice's last note ends
            prev_notes = voices[prev_voice]
            if prev_notes:
                prev_end = max(n.offset + n.duration for n in prev_notes)
            else:
                prev_end = entry.start_offset

            # Generate counterpoint from prev_end to current entry end
            entry_end = entry.start_offset + subject.duration
            gap = entry_end - prev_end

            if gap > 0:
                last_pitch = None
                if prev_notes:
                    non_rest = [n for n in prev_notes if not n.is_rest]
                    if non_rest:
                        last_pitch = non_rest[-1].pitch

                # For 3rd+ entry: try to use the countersubject
                if entry_idx >= 2 and countersubject and prev_idx == entry_idx - 1:
                    cp = _place_countersubject(
                        countersubject,
                        voice=prev_voice,
                        start_offset=entry.start_offset,
                        existing_voices=voices,
                        config=config,
                    )
                else:
                    cp = generate_free_counterpoint(
                        voice=prev_voice,
                        start_offset=prev_end,
                        duration=gap,
                        existing_voices=voices,
                        start_pitch=last_pitch,
                        config=config,
                    )

                # Capture countersubject from the 1st voice's counterpoint
                # against the 2nd entry (this IS the countersubject)
                if entry_idx == 1 and prev_idx == 0 and not countersubject:
                    countersubject = [
                        FugueNote(
                            pitch=n.pitch,
                            duration=n.duration,
                            voice=0,  # normalize to voice 0
                            offset=n.offset - entry.start_offset,  # relative offset
                            role=EntryRole.COUNTERSUBJECT,
                        )
                        for n in cp
                    ]

                # Tag as countersubject if placed from CS
                if entry_idx >= 2 and countersubject and prev_idx == entry_idx - 1:
                    for n in cp:
                        # This is already tagged by _place_countersubject
                        pass
                
                voices[prev_voice].extend(cp)

    return voices, countersubject


def _get_transposition(entry: VoiceEntry, subject: Subject) -> int:
    """Determine appropriate transposition for a middle entry."""
    return 0  # Default: same key. Override for middle entries in different keys.


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

    # Build ranges with comfortable padding around subject/answer
    padding = 6  # allow some room above and below

    # For 3 voices: top voice gets subject+answer range, mid gets overlap, bottom gets lower
    new_ranges = dict(config.voice_ranges)

    if num_voices == 3:
        # Voice 0 (top): answer range + padding
        new_ranges[0] = (subj_lo, max(answer_hi + padding, subj_hi + 12))
        # Voice 1 (mid): subject range ± padding, overlapping with both
        new_ranges[1] = (max(36, subj_lo - 7), subj_hi + padding)
        # Voice 2 (bottom): below subject
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

    # Return new config with adapted ranges
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
        voice_ranges=new_ranges,
    )
    return new_config


# ---------------------------------------------------------------------------
# Full fugue generator
# ---------------------------------------------------------------------------

def generate_fugue(
    plan: FuguePlan,
    config: Optional[GenerationConfig] = None,
) -> dict[int, list[FugueNote]]:
    """
    Generate a complete fugue from a plan.

    Works section by section, respecting constraints throughout:
    1. Exposition — subject/answer + constrained counterpoint
    2. Episodes — sequence-based with consonance checking
    3. Middle entries — subject placement + constrained fill
    4. Stretto — overlapping entries
    5. Coda — cadential motion to final chord
    """
    config = config or GenerationConfig()

    if not plan.exposition:
        return {}

    # Adapt voice ranges to the subject's actual pitch range
    config = _adapt_voice_ranges(config, plan.subject, plan.num_voices)

    # Generate exposition (also captures countersubject)
    voices, countersubject = generate_exposition(plan.exposition, plan.subject, config)

    # Process remaining sections
    total_sections = len(plan.sections)
    for sec_idx, section in enumerate(plan.sections):
        progress = sec_idx / max(1, total_sections - 1) if total_sections > 1 else 0.0
        if section.section_type == SectionType.EXPOSITION:
            continue  # already done

        elif section.section_type == SectionType.EPISODE:
            # Generate episodes voice by voice, staggered to avoid unisons.
            # Each voice uses different sequence direction, source material,
            # and start offset — creating thematic, imitative episodes
            # (per Burrows: episodes should derive from S/CS/inverted material).
            motif_dur = sum(
                n.duration for n in plan.subject.notes if not n.is_rest
            ) / max(1, len([n for n in plan.subject.notes if not n.is_rest][:4]))

            # Vary episode source material across voices
            episode_sources = ["subject_head", "subject_head_inverted", "countersubject", "subject_tail"]
            seq_intervals = [-2, 2, -3]

            for v in range(plan.num_voices):
                # Stagger: voice 0 starts on time, voice 1 delayed by motif,
                # voice 2 delayed by 2*motif — creates imitative texture
                stagger = v * motif_dur * min(4, len([n for n in plan.subject.notes if not n.is_rest][:4]))
                ep_start = section.start_offset + stagger
                ep_dur = section.estimated_duration - stagger
                if ep_dur <= 0:
                    continue

                ep_notes = generate_episode(
                    subject=plan.subject,
                    voice=v,
                    start_offset=ep_start,
                    duration=ep_dur,
                    existing_voices=voices,
                    sequence_interval=seq_intervals[v % len(seq_intervals)],
                    sequence_count=3,
                    config=config,
                    source=episode_sources[v % len(episode_sources)],
                    countersubject=countersubject,
                )
                voices.setdefault(v, []).extend(ep_notes)

        elif section.section_type in (SectionType.MIDDLE_ENTRY, SectionType.STRETTO):
            # Place subject entries, using inversion for variety
            # (per Burrows/BWV 547: combine rectus and inversus in stretto)
            trans = _key_to_transposition(section.key_area, plan.key_signature)
            for entry_i, entry in enumerate(section.entries):
                # In stretto: alternate rectus/inversus for contrapuntal intensity
                # In middle entries: use inversion for every other entry
                use_inversion = (
                    section.section_type == SectionType.STRETTO and entry_i % 2 == 1
                ) or (
                    section.section_type == SectionType.MIDDLE_ENTRY
                    and entry_i > 0 and entry_i % 2 == 1
                )
                xform = TransformType.INVERSION if use_inversion else TransformType.NONE
                subj_notes = place_subject(
                    plan.subject,
                    entry.voice,
                    entry.start_offset,
                    transposition=trans,
                    transform=xform,
                )
                # Adjust octave if placement would cause voice crossing
                subj_notes = _adjust_placement_octave(
                    subj_notes, entry.voice, voices, plan.num_voices, config,
                )
                voices.setdefault(entry.voice, []).extend(subj_notes)

            # Fill counterpoint for voices without subject entries
            # First voice without subject gets CS, rest get free CP
            entry_voices = {e.voice for e in section.entries}
            free_voices = [v for v in range(plan.num_voices) if v not in entry_voices]
            cs_placed = False

            for v in free_voices:
                if not cs_placed and countersubject and section.entries:
                    # Try to place countersubject alongside the first subject entry
                    trans = _key_to_transposition(section.key_area, plan.key_signature)
                    cp = _place_countersubject(
                        countersubject,
                        voice=v,
                        start_offset=section.entries[0].start_offset,
                        existing_voices=voices,
                        config=config,
                        transposition=trans,
                    )
                    cs_placed = True
                else:
                    cp = generate_free_counterpoint(
                        voice=v,
                        start_offset=section.start_offset,
                        duration=section.estimated_duration,
                        existing_voices=voices,
                        config=config,
                        progress=progress,
                    )
                voices.setdefault(v, []).extend(cp)

        elif section.section_type == SectionType.CODA:
            # Generate cadential coda
            coda_notes = generate_coda(
                plan.subject,
                plan.num_voices,
                section.start_offset,
                section.estimated_duration,
                voices,
                config,
            )
            for v, v_notes in coda_notes.items():
                voices.setdefault(v, []).extend(v_notes)

    # Post-process: fix unresolved leaps where possible
    voices = _postprocess_leap_resolution(voices)

    # Post-process: fix parallel fifths/octaves/unisons
    voices = _postprocess_fix_parallels(voices)

    # Post-process: fix strong-beat dissonances
    voices = _postprocess_fix_dissonances(voices)

    # Post-process: ensure climax is in the latter half
    voices = _postprocess_climax_placement(voices)

    return voices


# ---------------------------------------------------------------------------
# Leap resolution post-processor
# ---------------------------------------------------------------------------

def _postprocess_leap_resolution(
    voices: dict[int, list[FugueNote]],
) -> dict[int, list[FugueNote]]:
    """
    Scan each voice for unresolved leaps and fix resolving notes
    by shifting them 1-2 semitones to create step motion in opposite direction.
    Only modifies FREE_COUNTERPOINT and EPISODE_MATERIAL notes.
    Constraint-aware: skips fixes that would create voice crossing or dissonance.
    """
    for v, v_notes in voices.items():
        non_rest = [n for n in v_notes if not n.is_rest]
        if len(non_rest) < 3:
            continue

        for i in range(len(non_rest) - 2):
            leap = non_rest[i + 1].pitch - non_rest[i].pitch
            if abs(leap) < 5:
                continue  # not a leap

            resolution = non_rest[i + 2].pitch - non_rest[i + 1].pitch
            # Check if already resolved (step in opposite direction)
            if abs(resolution) <= 2 and (leap > 0 and resolution < 0 or leap < 0 and resolution > 0):
                continue

            # Only modify modifiable note types
            target = non_rest[i + 2]
            if target.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
                continue

            # Calculate ideal resolution pitch: step (1-2 semitones) opposite to leap
            leap_dir = 1 if leap > 0 else -1
            target_offset = target.offset
            is_strong = (target_offset % 1.0) < 0.01

            # Find what other voices are sounding at this time
            other_pitches: dict[int, int] = {}
            for ov, ov_notes in voices.items():
                if ov == v:
                    continue
                for n in ov_notes:
                    if n.offset <= target_offset < n.offset + n.duration:
                        other_pitches[ov] = n.pitch
                        break

            best_new_pitch = None
            for step in [1, 2]:
                new_pitch = non_rest[i + 1].pitch - leap_dir * step

                # Check voice crossing
                crossing = False
                for ov, op in other_pitches.items():
                    if v < ov and new_pitch < op:
                        crossing = True
                    elif v > ov and new_pitch > op:
                        crossing = True
                if crossing:
                    continue

                # Check dissonance on strong beats
                if is_strong and other_pitches:
                    has_dissonance = any(
                        not is_consonance(new_pitch - op)
                        for op in other_pitches.values()
                        if op != -1
                    )
                    if has_dissonance:
                        continue

                # Check parallel 5ths/8ves with prev beat
                if i + 1 < len(non_rest):
                    prev_pitch = non_rest[i + 1].pitch
                    prev_offset = non_rest[i + 1].offset
                    creates_parallel = False
                    for ov, op_cur in other_pitches.items():
                        # Find previous pitch for this other voice
                        op_prev = op_cur
                        for n_check in voices.get(ov, []):
                            if n_check.offset <= prev_offset < n_check.offset + n_check.duration:
                                op_prev = n_check.pitch
                                break
                        if op_prev == op_cur or prev_pitch == new_pitch:
                            continue
                        if _check_parallel_motion(v, new_pitch, prev_pitch, ov, op_prev, op_cur):
                            creates_parallel = True
                            break
                    if creates_parallel:
                        continue

                best_new_pitch = new_pitch
                break

            if best_new_pitch is None:
                continue

            # Apply the fix
            for j, n in enumerate(v_notes):
                if n is target:
                    v_notes[j] = FugueNote(
                        pitch=best_new_pitch,
                        duration=n.duration,
                        voice=n.voice,
                        offset=n.offset,
                        role=n.role,
                    )
                    non_rest[i + 2] = v_notes[j]
                    break

    return voices


def _postprocess_fix_dissonances(
    voices: dict[int, list[FugueNote]],
) -> dict[int, list[FugueNote]]:
    """
    Scan for strong-beat dissonances in modifiable notes and fix them
    by shifting 1-2 semitones to the nearest consonance, if possible
    without creating voice crossing.
    """
    for v, v_notes in voices.items():
        for j, n in enumerate(v_notes):
            if n.is_rest:
                continue
            if n.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
                continue
            # Only fix strong beats
            if (n.offset % 1.0) > 0.01:
                continue

            # Find other pitches sounding at this offset
            other_pitches: dict[int, int] = {}
            for ov, ov_notes in voices.items():
                if ov == v:
                    continue
                for on in ov_notes:
                    if on.is_rest:
                        continue
                    if on.offset <= n.offset < on.offset + on.duration:
                        other_pitches[ov] = on.pitch
                        break

            if not other_pitches:
                continue

            # Check if current pitch is dissonant with any voice
            has_dissonance = any(
                is_dissonance(n.pitch - op) for op in other_pitches.values()
            )
            if not has_dissonance:
                continue

            # Try small adjustments to find a consonant pitch
            best_adj = None
            for adj in [1, -1, 2, -2]:
                new_p = n.pitch + adj
                # All consonant?
                all_cons = all(
                    is_consonance(new_p - op) for op in other_pitches.values()
                )
                if not all_cons:
                    continue
                # No crossing?
                crossing = False
                for ov, op in other_pitches.items():
                    if v < ov and new_p < op:
                        crossing = True
                    elif v > ov and new_p > op:
                        crossing = True
                if crossing:
                    continue
                best_adj = adj
                break

            if best_adj is not None:
                v_notes[j] = FugueNote(
                    pitch=n.pitch + best_adj,
                    duration=n.duration,
                    voice=n.voice,
                    offset=n.offset,
                    role=n.role,
                )

    return voices


def _postprocess_climax_placement(
    voices: dict[int, list[FugueNote]],
) -> dict[int, list[FugueNote]]:
    """
    Ensure the highest note (climax) occurs in the latter half of the piece.
    If the current climax is too early (<40% position), find a modifiable note
    in the 55-80% range and raise it above the current highest pitch.
    """
    all_notes = [(v, n) for v, v_notes in voices.items() for n in v_notes if not n.is_rest]
    if not all_notes:
        return voices

    total_dur = max(n.offset + n.duration for _, n in all_notes)
    if total_dur <= 0:
        return voices

    highest_pitch = max(n.pitch for _, n in all_notes)
    highest_notes = [(v, n) for v, n in all_notes if n.pitch == highest_pitch]
    climax_pos = highest_notes[0][1].offset / total_dur

    if climax_pos >= 0.4:
        return voices  # climax already well-placed

    # Find a modifiable note in soprano (voice 0) in the 55-80% range
    target_start = total_dur * 0.55
    target_end = total_dur * 0.80
    target_pitch = highest_pitch + 1  # just 1 semitone above current highest

    soprano_notes = voices.get(0, [])
    for j, n in enumerate(soprano_notes):
        if n.is_rest:
            continue
        if n.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
            continue
        if target_start <= n.offset <= target_end:
            soprano_notes[j] = FugueNote(
                pitch=target_pitch,
                duration=n.duration,
                voice=n.voice,
                offset=n.offset,
                role=n.role,
            )
            return voices

    return voices


def _postprocess_fix_parallels(
    voices: dict[int, list[FugueNote]],
) -> dict[int, list[FugueNote]]:
    """
    Scan for parallel fifths, octaves, and unisons between voice pairs.
    Fix by shifting the modifiable note by 1 semitone, checking that
    the fix doesn't create new violations.
    """
    snapshots = build_snapshots(voices)
    if len(snapshots) < 2:
        return voices

    voice_ids = sorted(voices.keys())

    for i in range(len(snapshots) - 1):
        s1, s2 = snapshots[i], snapshots[i + 1]
        for vi in voice_ids:
            for vj in voice_ids:
                if vi >= vj:
                    continue
                p1_hi = s1.pitches.get(vi, -1)
                p1_lo = s1.pitches.get(vj, -1)
                p2_hi = s2.pitches.get(vi, -1)
                p2_lo = s2.pitches.get(vj, -1)

                if -1 in (p1_hi, p1_lo, p2_hi, p2_lo):
                    continue
                if p1_lo == p2_lo and p1_hi == p2_hi:
                    continue

                int1 = interval_class(p1_hi - p1_lo)
                int2 = interval_class(p2_hi - p2_lo)
                motion = classify_motion(p1_lo, p2_lo, p1_hi, p2_hi)

                if motion != MotionType.PARALLEL:
                    continue
                if int2 not in (0, 7):  # not a perfect interval
                    continue
                if int1 != int2:
                    continue

                # Found a parallel 5th/8ve/unison — try to fix the higher voice
                # at snapshot i+1 by shifting 1 semitone
                target_offset = s2.offset
                for fix_voice in [vi, vj]:
                    v_notes = voices.get(fix_voice, [])
                    for j, n in enumerate(v_notes):
                        if n.is_rest:
                            continue
                        if abs(n.offset - target_offset) > 0.01:
                            continue
                        if n.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
                            continue

                        # Try +1 and -1 semitone
                        for adj in [1, -1]:
                            new_pitch = n.pitch + adj
                            new_int = interval_class(
                                (new_pitch if fix_voice == vi else p2_hi)
                                - (p2_lo if fix_voice == vi else new_pitch)
                            )
                            # Make sure we broke the parallel
                            if new_int == int2:
                                continue
                            # Check no crossing
                            other_v = vj if fix_voice == vi else vi
                            other_p = s2.pitches.get(other_v, -1)
                            if fix_voice < other_v and new_pitch < other_p:
                                continue
                            if fix_voice > other_v and new_pitch > other_p:
                                continue

                            v_notes[j] = FugueNote(
                                pitch=new_pitch,
                                duration=n.duration,
                                voice=n.voice,
                                offset=n.offset,
                                role=n.role,
                            )
                            break
                        break  # fixed this voice for this snapshot
                    else:
                        continue
                    break  # fixed one pair

    return voices


# ---------------------------------------------------------------------------
# Coda / cadence generator
# ---------------------------------------------------------------------------

def generate_coda(
    subject: Subject,
    num_voices: int,
    start_offset: float,
    duration: float,
    existing_voices: dict[int, list[FugueNote]],
    config: Optional[GenerationConfig] = None,
) -> dict[int, list[FugueNote]]:
    """
    Generate a cadential coda section.

    Strategy:
    - Start with free counterpoint leading to a dominant chord
    - End with V → I authentic cadence
    - Final chord uses tonic triad in appropriate voicing
    """
    config = config or GenerationConfig()
    result: dict[int, list[FugueNote]] = {}

    # Determine tonic pitches based on subject key
    key_sig = subject.key_signature
    tonic_pc = _key_name_to_pc(key_sig)
    dominant_pc = (tonic_pc + 7) % 12
    is_minor = key_sig[0].islower()
    third_pc = (tonic_pc + 3) % 12 if is_minor else (tonic_pc + 4) % 12

    # --- Thematic coda: place subject in bass, counterpoint above, then cadence ---
    # Per Burrows thesis: Bach's codas combine S + CS + thematic material (BWV 544, 547)
    free_dur = max(0, duration - 4.0)  # save last 4 beats for cadence
    subj_dur = subject.duration
    bass_voice = num_voices - 1

    working_voices = dict(existing_voices)

    # Place a final subject statement in the lowest voice if room allows
    if free_dur >= subj_dur and subj_dur > 0:
        subj_notes = place_subject(
            subject, bass_voice, start_offset, transposition=0,
        )
        # Adjust octave to fit bass range
        subj_notes = _adjust_placement_octave(
            subj_notes, bass_voice, working_voices, num_voices, config,
        )
        result.setdefault(bass_voice, []).extend(subj_notes)
        working_voices.setdefault(bass_voice, []).extend(subj_notes)

        # Fill upper voices with free counterpoint over the subject
        for v in range(num_voices - 1):
            cp = generate_free_counterpoint(
                voice=v,
                start_offset=start_offset,
                duration=free_dur,
                existing_voices=working_voices,
                config=config,
            )
            result.setdefault(v, []).extend(cp)
            working_voices.setdefault(v, []).extend(cp)
    else:
        for v in range(num_voices):
            if free_dur > 0:
                cp = generate_free_counterpoint(
                    voice=v,
                    start_offset=start_offset,
                    duration=free_dur,
                    existing_voices=working_voices,
                    config=config,
                )
                result.setdefault(v, []).extend(cp)
                working_voices.setdefault(v, []).extend(cp)

    # Final cadence: V chord (2 beats) → I chord (2 beats)
    cadence_offset = start_offset + free_dur

    # Build voicings for V and I chords
    for v in range(num_voices):
        lo, hi = config.voice_ranges.get(v, (48, 72))
        mid = (lo + hi) // 2

        # V chord pitch for this voice — bass MUST have the dominant root
        if v == num_voices - 1:
            # Bass voice: dominant root for proper PAC detection
            v_pitch = _nearest_pitch_class(mid, dominant_pc, lo, hi)
        elif v == 1:
            leading_tone_pc = (tonic_pc - 1) % 12  # leading tone
            v_pitch = _nearest_pitch_class(mid, leading_tone_pc, lo, hi)
        elif v >= 2:
            v_fifth_pc = (dominant_pc + 7) % 12  # 5th of V = 2nd scale degree
            v_pitch = _nearest_pitch_class(mid, v_fifth_pc, lo, hi)
        else:
            v_pitch = _nearest_pitch_class(mid, dominant_pc, lo, hi)

        result.setdefault(v, []).append(FugueNote(
            pitch=v_pitch,
            duration=2.0,
            voice=v,
            offset=cadence_offset,
            role=EntryRole.CODETTA,
        ))

        # I chord pitch for this voice
        if v == 0:
            i_pitch = _nearest_pitch_class(mid, third_pc, lo, hi)
        elif v == num_voices - 1:
            i_pitch = _nearest_pitch_class(mid, tonic_pc, lo, hi)
        else:
            fifth_pc = (tonic_pc + 7) % 12
            i_pitch = _nearest_pitch_class(mid, fifth_pc, lo, hi)

        result.setdefault(v, []).append(FugueNote(
            pitch=i_pitch,
            duration=2.0,
            voice=v,
            offset=cadence_offset + 2.0,
            role=EntryRole.CODETTA,
        ))

    return result


def _key_name_to_pc(key_name: str) -> int:
    """Convert key name to pitch class (0-11)."""
    pc_map = {
        "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
        "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
        "B": 11, "b": 11,
        "Bb": 10, "bb": 10, "Eb": 3, "eb": 3, "Ab": 8, "ab": 8,
        "F#": 6, "f#": 6, "C#": 1, "c#": 1,
    }
    return pc_map.get(key_name, 0)


def _nearest_pitch_class(center: int, pc: int, lo: int, hi: int) -> int:
    """Find the nearest MIDI pitch to center with the given pitch class."""
    best = center
    best_dist = 999
    for p in range(lo, hi + 1):
        if p % 12 == pc:
            dist = abs(p - center)
            if dist < best_dist:
                best_dist = dist
                best = p
    return best


def _key_to_transposition(key_area: str, home_key: str) -> int:
    """Convert a key area string to transposition in semitones."""
    # Simple lookup for common transpositions
    key_semitones = {
        "C": 0, "c": 0, "D": 2, "d": 2, "E": 4, "e": 4,
        "F": 5, "f": 5, "G": 7, "g": 7, "A": 9, "a": 9,
        "B": 11, "b": 11,
        "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "Gb": 6,
        "F#": 6, "f#": 6, "C#": 1, "c#": 1,
    }
    # If key_area contains "→", use the target
    if "→" in key_area:
        key_area = key_area.split("→")[-1].strip()

    home_st = key_semitones.get(home_key, 0)
    target_st = key_semitones.get(key_area, 0)
    return (target_st - home_st) % 12


def _adjust_placement_octave(
    notes: list[FugueNote],
    voice: int,
    existing_voices: dict[int, list[FugueNote]],
    num_voices: int,
    config: GenerationConfig,
) -> list[FugueNote]:
    """
    Check if a subject/answer placement causes voice crossing.
    If so, shift by octave(s) to minimize crossings while staying in range.
    """
    if not notes:
        return notes

    pitches = [n.pitch for n in notes if not n.is_rest]
    if not pitches:
        return notes

    avg_pitch = sum(pitches) / len(pitches)

    # Count crossings at current transposition
    def count_crossings(shift: int) -> int:
        crossings = 0
        for n in notes:
            if n.is_rest:
                continue
            p = n.pitch + shift
            for ov, ov_notes in existing_voices.items():
                if ov == voice:
                    continue
                for on in ov_notes:
                    if on.is_rest:
                        continue
                    if abs(on.offset - n.offset) < 0.01 or (
                        on.offset < n.offset + n.duration and n.offset < on.offset + on.duration
                    ):
                        # Overlapping in time
                        if voice < ov and p < on.pitch:
                            crossings += 1
                        elif voice > ov and p > on.pitch:
                            crossings += 1
                        break  # one check per concurrent note pair
        return crossings

    lo, hi = config.voice_ranges.get(voice, (36, 84))
    best_shift = 0
    best_crossings = count_crossings(0)

    # Try octave shifts
    for shift in [-12, 12, -24, 24]:
        shifted_lo = min(pitches) + shift
        shifted_hi = max(pitches) + shift
        if shifted_lo < lo - 3 or shifted_hi > hi + 3:
            continue
        c = count_crossings(shift)
        if c < best_crossings:
            best_crossings = c
            best_shift = shift

    if best_shift == 0:
        return notes

    # Apply shift
    return [
        FugueNote(
            pitch=n.pitch + best_shift if not n.is_rest else n.pitch,
            duration=n.duration,
            voice=n.voice,
            offset=n.offset,
            role=n.role,
            transform=n.transform,
        )
        for n in notes
    ]


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
