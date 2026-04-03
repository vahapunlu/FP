"""
FugueForge Counterpoint Generation

Free counterpoint generation and countersubject placement.
Both use constraint-aware scoring to produce voice-leading-legal lines.
"""

from __future__ import annotations

import random
from typing import Optional

from .representation import EntryRole, FugueNote, Subject
from .rules import is_consonance, is_dissonance
from .candidates import (
    GenerationConfig,
    VoiceState,
    _candidate_pitches,
    _score_candidate,
)
from .harmony import ChordLabel, get_chord_at


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
    harmonic_skeleton: Optional[list[ChordLabel]] = None,
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
    if voice == 0:  # soprano — controls climax
        if progress < 0.45:
            ceiling_reduction = int((0.45 - progress) * 10)
            hi = max(hi - ceiling_reduction, lo + 12)
        elif progress > 0.55:
            ceiling_raise = int((progress - 0.55) * 8)
            hi = hi + ceiling_raise

    if start_pitch is None:
        start_pitch = (lo + hi) // 2

    notes: list[FugueNote] = []
    current_offset = start_offset
    current_pitch = start_pitch
    prev_melodic_interval = 0
    prev_was_dissonant = False
    end_offset = start_offset + duration

    step = config.step_resolution

    state = VoiceState()
    state.set_current(voice, current_pitch)

    while current_offset < end_offset - 0.01:
        # Determine current and previous pitches for all other voices
        other_current: dict[int, int] = {}
        other_prev: dict[int, int] = {}
        for v, v_notes in existing_voices.items():
            if v == voice:
                continue
            for n in v_notes:
                if n.offset <= current_offset < n.offset + n.duration:
                    other_current[v] = n.pitch
                    break
            prev_t = current_offset - step
            for n in v_notes:
                if n.offset <= prev_t < n.offset + n.duration:
                    other_prev[v] = n.pitch
                    break
            if v not in other_prev:
                other_prev[v] = other_current.get(v, -1)

        is_strong = (current_offset % 1.0) < 0.01

        # Look up current chord from harmonic skeleton
        current_chord_pcs: frozenset[int] = frozenset()
        if harmonic_skeleton:
            chord = get_chord_at(harmonic_skeleton, current_offset)
            if chord:
                current_chord_pcs = chord.chord_pcs

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
                chord_pcs=current_chord_pcs,
            )
            # Dissonance resolution bonus/penalty
            if prev_was_dissonant and other_current:
                is_now_consonant = all(
                    is_consonance(cp - op)
                    for op in other_current.values()
                    if op != -1
                )
                if is_now_consonant and abs(cp - current_pitch) <= 2:
                    s += 15.0   # Strong reward for stepwise resolution
                elif is_now_consonant:
                    s += 6.0    # Reward for any resolution
                elif not is_now_consonant:
                    s -= 15.0   # Heavy penalty for consecutive dissonance
            if not vetoed:
                scored.append((s, cp))

        # If all candidates vetoed, relax hard vetoes but keep dissonance penalty
        if not scored:
            relaxed_config = GenerationConfig(
                **{**config.__dict__,
                   'parallel_veto': False,
                   'crossing_veto': False,
                   'dissonance_penalty': config.dissonance_penalty * 1.5,
                   'scale_pcs': config.scale_pcs,
                   'scale_bonus': config.scale_bonus,
                   'chromatic_penalty': config.chromatic_penalty}
            )
            for cp in candidates:
                s, _ = _score_candidate(
                    candidate=cp,
                    prev_pitch=current_pitch,
                    voice=voice,
                    other_voices_current=other_current,
                    other_voices_prev=other_prev,
                    is_strong_beat=is_strong,
                    prev_interval=prev_melodic_interval,
                    config=relaxed_config,
                )
                scored.append((s, cp))

        scored.sort(reverse=True)

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
        if is_strong and random.random() < 0.25 and current_offset + step * 2 <= end_offset:
            dur = step * 2
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

        is_strong = (note_offset % 1.0) < 0.01
        use_prev = prev_pitch if prev_pitch != -1 else raw_pitch

        # Score the raw CS pitch and small adjustments (0, ±1, ±2, ±3)
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
                chord_pcs=frozenset(),  # don't constrain CS to chords
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
