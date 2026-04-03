"""
FugueForge Invention Generator — v3: Harmony-First Melody

CORE PRINCIPLE:
  Melodies are BORN from harmony, not adjusted after the fact.

  Strong beats → chord tone (mandatory)
  Weak beats   → diatonic step following subject contour
  Every beat   → consonance with other voice checked

  Subject provides: RHYTHM + CONTOUR DIRECTION
  Harmonic skeleton provides: ACTUAL PITCHES (chord tones)

This replaces the note-by-note random scoring approach with a
deterministic harmony-driven algorithm:

  1. Extract rhythm pattern (durations) and contour (up/down) from subject
  2. At every time position, follow contour with a diatonic step
  3. On strong beats, SNAP to the nearest chord tone
  4. On weak beats, allow passing tones (diatonic steps are OK)
  5. Consonance with other voice filters all decisions
"""

from __future__ import annotations

import copy
from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    FuguePlan,
    SectionType,
    Subject,
    TransformType,
)
from .candidates import GenerationConfig
from .voice_utils import _key_to_transposition, key_to_scale_pcs
from .placement import place_subject, _adjust_placement_octave
from .harmony import (
    generate_harmonic_skeleton,
    key_name_to_pc,
    key_is_minor,
    ChordLabel,
    get_chord_at,
)
from .postprocess import (
    _postprocess_fix_dissonances,
    _postprocess_fix_parallels,
    _postprocess_leap_resolution,
)
from .rules import interval_class


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Config
# ═══════════════════════════════════════════════════════════════════════════

def _invention_config() -> GenerationConfig:
    """GenerationConfig for 2-voice inventions (used by postprocessors)."""
    return GenerationConfig(
        beam_width=8,
        max_candidates_per_step=24,
        temperature=0.10,
        step_resolution=0.25,
        prefer_stepwise=0.85,
        max_leap=9,
        consonance_weight=6.0,
        dissonance_penalty=15.0,
        chord_tone_bonus=7.0,
        non_chord_penalty=4.0,
        parallel_veto=True,
        crossing_veto=True,
        leap_resolution_bonus=4.0,
        contrary_motion_bonus=5.0,
        scale_bonus=12.0,
        chromatic_penalty=8.0,
        stepwise_bonus=12.0,
        third_bonus=5.0,
        fourth_fifth_penalty=2.0,
        large_leap_penalty=12.0,
        consecutive_leap_penalty=15.0,
        direction_momentum=3.0,
        direction_reversal=4.0,
        register_gravity=3.0,
        phrase_length=4.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Voice ranges — clear register separation
# ═══════════════════════════════════════════════════════════════════════════

def _adapt_invention_ranges(
    config: GenerationConfig,
    subject: Subject,
    answer_trans: int,
) -> GenerationConfig:
    """
    Voice 0 (treble): subject register + padding above
    Voice 1 (bass):   answer register (octave or 5th below) + padding below
    Gap between V0 bottom and V1 top >= 3 semitones.
    """
    subj_lo, subj_hi = subject.pitch_range

    # Voice 0 (treble)
    v0_lo = subj_lo - 3
    v0_hi = subj_hi + 7
    if v0_hi - v0_lo < 15:
        v0_hi = v0_lo + 15

    # Voice 1 (bass): around the answer register
    if answer_trans > -8:
        # Answer at 5th: only 5 semitones below.
        # Push V0 up and V1 down to create a 5-semitone gap.
        v0_lo = subj_lo          # no lower padding — keep treble high
        v0_hi = max(subj_hi + 8, v0_lo + 15)
        v1_hi = v0_lo - 5        # 5-semitone gap
        v1_lo = max(36, v1_hi - 18)
        if v1_hi - v1_lo < 14:
            v1_lo = max(36, v1_hi - 14)
    else:
        ans_lo = subj_lo + answer_trans - 3
        ans_hi = subj_hi + answer_trans + 3
        v1_hi = min(ans_hi, v0_lo - 3)
        v1_lo = max(36, v1_hi - 18)
        if v1_hi - v1_lo < 14:
            v1_lo = max(36, v1_hi - 14)

    new_ranges = dict(config.voice_ranges)
    new_ranges[0] = (v0_lo, v0_hi)
    new_ranges[1] = (v1_lo, v1_hi)

    return GenerationConfig(
        **{**config.__dict__,
           'voice_ranges': new_ranges,
           'scale_pcs': config.scale_pcs or key_to_scale_pcs(subject.key_signature)}
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Musical analysis helpers
# ═══════════════════════════════════════════════════════════════════════════

def _beat_strength(offset: float) -> int:
    """0 = weak (16th), 1 = semi (8th), 2 = strong (quarter)."""
    frac = offset % 1.0
    if frac < 0.01 or frac > 0.99:
        return 2          # quarter-note downbeat
    if abs(frac - 0.5) < 0.01:
        return 1          # 8th-note offbeat
    return 0              # 16th-note subdivision


def _extract_contour(subject: Subject) -> list[int]:
    """
    Extract directional contour from subject: list of +1 / -1 / 0.
    Length = len(non_rest_notes) - 1.
    """
    pitches = [n.pitch for n in subject.notes if not n.is_rest]
    contour: list[int] = []
    for i in range(1, len(pitches)):
        diff = pitches[i] - pitches[i - 1]
        contour.append(1 if diff > 0 else (-1 if diff < 0 else 0))
    return contour


def _extract_rhythm(subject: Subject) -> list[float]:
    """Extract the duration pattern from subject (non-rest notes)."""
    return [n.duration for n in subject.notes if not n.is_rest]


def _cp_rhythm(subject: Subject) -> list[float]:
    """
    Build a *slower* rhythm pattern for the counterpoint voice.

    While the subject/running voice uses fast figuration (8th/16th notes),
    the accompanying voice provides contrast with longer values.
    Strategy: group subject notes in pairs/triples → merge into one longer note.
    Ensures each CP note is at least 0.5 beats (8th note).
    """
    src = _extract_rhythm(subject)
    if not src:
        return [1.0]
    # Merge consecutive short notes into longer ones
    # Target: roughly half the number of notes
    result: list[float] = []
    acc = 0.0
    for i, d in enumerate(src):
        acc += d
        # Merge until we reach at least 0.5, or at phrase boundary (every 4th)
        if acc >= 0.5 or i == len(src) - 1:
            result.append(round(acc, 4))
            acc = 0.0
    # If everything ended up as one giant note, split
    if len(result) == 1 and result[0] > 2.0:
        half = round(result[0] / 2, 4)
        result = [half, result[0] - half]
    return result


def _get_sounding_pitch(
    voices: dict[int, list[FugueNote]],
    exclude_voice: int,
    offset: float,
) -> int | None:
    """Return the pitch of the OTHER voice sounding at *offset*."""
    for v, notes in voices.items():
        if v == exclude_voice:
            continue
        for n in notes:
            if n.offset <= offset < n.offset + n.duration and not n.is_rest:
                return n.pitch
    return None


def _is_consonant(p1: int, p2: int) -> bool:
    return interval_class(p1 - p2) in (0, 3, 4, 7, 8, 9)


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Motif extraction and sequence generation (Bach-style)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_motif(subject: Subject, max_notes: int = 6) -> list[tuple[int, float]]:
    """
    Extract the subject head as a list of (interval, duration) pairs.

    The *interval* is relative to the first note (semitones from motif root).
    The *duration* is the original note duration.
    Returns at most *max_notes* entries; the first always has interval = 0.
    """
    notes = [n for n in subject.notes if not n.is_rest][:max_notes]
    if not notes:
        return [(0, 0.5)]
    root = notes[0].pitch
    return [(n.pitch - root, n.duration) for n in notes]


def _invert_motif(motif: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Invert a motif: flip intervals around the root."""
    return [(-iv, dur) for iv, dur in motif]


def _place_motif(
    motif: list[tuple[int, float]],
    root_pitch: int,
    start_offset: float,
    voice: int,
    lo: int,
    hi: int,
    role: EntryRole = EntryRole.EPISODE_MATERIAL,
) -> list[FugueNote]:
    """Place a motif at a given root pitch and offset, clamping to range."""
    result = []
    t = start_offset
    for iv, dur in motif:
        p = root_pitch + iv
        p = max(lo, min(hi, p))
        result.append(FugueNote(
            pitch=p, duration=dur, voice=voice,
            offset=round(t, 4), role=role,
        ))
        t += dur
    return result


def _motif_duration(motif: list[tuple[int, float]]) -> float:
    """Total duration of a motif."""
    return sum(dur for _, dur in motif)


def _generate_sequence(
    motif: list[tuple[int, float]],
    start_pitch: int,
    start_offset: float,
    total_duration: float,
    voice: int,
    scale_pcs: frozenset[int],
    lo: int,
    hi: int,
    direction: int = -1,
    other_voice_notes: list[FugueNote] | None = None,
) -> list[FugueNote]:
    """
    Generate a diatonic SEQUENCE: repeat the motif at successively
    higher or lower diatonic pitch levels.

    This is THE core technique of Bach's invention episodes:
    a short motif fragment is stated, then transposed down (or up)
    by a diatonic step, then again, creating a stepwise chain.

    direction: -1 = descending sequence (most common), +1 = ascending
    """
    m_dur = _motif_duration(motif)
    if m_dur <= 0:
        return []

    result: list[FugueNote] = []
    root = start_pitch
    t = start_offset
    end = start_offset + total_duration

    # Build sorted scale for diatonic transposition
    scale = sorted(scale_pcs) if scale_pcs else list(range(12))

    while t + m_dur <= end + 0.05:
        notes = _place_motif(motif, root, t, voice, lo, hi, EntryRole.EPISODE_MATERIAL)

        # Consonance check: if any note clashes badly with the other voice,
        # try shifting the whole motif ±1 diatonic step
        if other_voice_notes:
            clash_count = 0
            for n in notes:
                for on in other_voice_notes:
                    if on.is_rest:
                        continue
                    if on.offset < n.offset + n.duration and on.offset + on.duration > n.offset:
                        iv = interval_class(n.pitch - on.pitch)
                        if iv in (1, 2, 6, 10, 11):
                            clash_count += 1
            if clash_count >= 2:
                # Try one step further in sequence direction
                alt_root = _next_diatonic_pitch(root, direction, scale, lo, hi)
                alt_notes = _place_motif(motif, alt_root, t, voice, lo, hi)
                alt_clash = 0
                for n in alt_notes:
                    for on in other_voice_notes:
                        if on.is_rest:
                            continue
                        if on.offset < n.offset + n.duration and on.offset + on.duration > n.offset:
                            iv = interval_class(n.pitch - on.pitch)
                            if iv in (1, 2, 6, 10, 11):
                                alt_clash += 1
                if alt_clash < clash_count:
                    notes = alt_notes
                    root = alt_root

        result.extend(notes)
        t += m_dur
        # Move root DOWN (or UP) by one diatonic step for next iteration
        root = _next_diatonic_pitch(root, direction, scale, lo, hi)

    return result


def _next_diatonic_pitch(
    current: int, direction: int, scale: list[int], lo: int, hi: int,
) -> int:
    """Find the next diatonic pitch in the given direction."""
    step = 1 if direction > 0 else -1
    p = current + step
    for _ in range(4):  # max 4 chromatic steps to find scale tone
        if lo <= p <= hi and (p % 12) in scale:
            return p
        p += step
    # Fallback: just move by 2 semitones (whole step)
    return max(lo, min(hi, current + step * 2))


def _generate_sustained_bass(
    start_offset: float,
    duration: float,
    voice: int,
    lo: int,
    hi: int,
    start_pitch: int,
    scale_pcs: frozenset[int],
    harmonic_skeleton: list[ChordLabel] | None,
    other_voice_notes: list[FugueNote] | None = None,
    role: EntryRole = EntryRole.EPISODE_MATERIAL,
) -> list[FugueNote]:
    """
    Generate sustained harmonic support (one chord tone per beat or half-bar).
    Each note is a chord tone from the harmonic skeleton, held for ~1 beat.
    Checks consonance against the other voice's notes.
    """
    result = []
    t = start_offset
    end = start_offset + duration
    current = start_pitch
    beat_dur = 1.0  # one note per beat

    while t < end - 0.05:
        dur = min(beat_dur, end - t)
        if dur < 0.1:
            break

        # Get chord tone
        chord = get_chord_at(harmonic_skeleton, t) if harmonic_skeleton else None
        chord_pcs = chord.chord_pcs if chord else frozenset()

        # Find best chord tone near current pitch
        best_p = current
        best_sc = -9999.0
        for delta in range(-7, 8):
            p = current + delta
            if p < lo or p > hi:
                continue
            sc = -abs(delta) * 2.0  # proximity
            if chord_pcs and (p % 12) in chord_pcs:
                sc += 8.0
            elif scale_pcs and (p % 12) in scale_pcs:
                sc += 2.0
            else:
                sc -= 5.0
            # Consonance with other voice
            if other_voice_notes:
                for on in other_voice_notes:
                    if on.is_rest:
                        continue
                    if on.offset < t + dur and on.offset + on.duration > t:
                        iv = interval_class(p - on.pitch)
                        if iv in (3, 4, 8, 9):
                            sc += 3.0
                        elif iv in (1, 11):
                            sc -= 12.0
                        elif iv == 6:
                            sc -= 8.0
                        elif iv in (2, 10):
                            sc -= 5.0
            if sc > best_sc:
                best_sc = sc
                best_p = p

        result.append(FugueNote(
            pitch=best_p, duration=round(dur, 4), voice=voice,
            offset=round(t, 4), role=role,
        ))
        current = best_p
        t += dur

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Harmony-first pitch selection (for free counterpoint)
# ═══════════════════════════════════════════════════════════════════════════

def _snap_to_chord_tone(
    ideal: int,
    chord_pcs: frozenset[int],
    scale_pcs: frozenset[int],
    other_pitch: int | None,
    lo: int,
    hi: int,
) -> int:
    """
    Find the best chord tone near *ideal* on a STRONG beat.

    Scoring:
      -3 * |adjustment|   proximity to ideal pitch
      +8   imperfect consonance (3rd/6th) with other voice
      +3   perfect consonance (unison/5th/octave) with other voice
      -15  dissonance with other voice
      -5   unison/octave with other voice (avoid doubling)
    """
    if not chord_pcs:
        # Fallback: use scale tones
        chord_pcs = scale_pcs

    best_p, best_sc = ideal, -9999.0

    for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5):
        p = ideal + delta
        if p < lo or p > hi:
            continue
        if chord_pcs and (p % 12) not in chord_pcs:
            continue

        sc = -abs(delta) * 3.0          # proximity

        if other_pitch is not None:
            iv = interval_class(p - other_pitch)
            if iv in (3, 4, 8, 9):      # imperfect consonance — BEST
                sc += 8.0
            elif iv == 7:               # perfect 5th — OK
                sc += 3.0
            elif iv == 0:               # unison/octave — avoid
                sc -= 5.0
            else:                       # dissonance — BAD on strong beat
                sc -= 15.0

        if sc > best_sc:
            best_sc = sc
            best_p = p

    return best_p


def _diatonic_step(
    current: int,
    direction: int,
    scale_pcs: frozenset[int],
    lo: int,
    hi: int,
    other_pitch: int | None = None,
) -> int:
    """
    Take one diatonic step in the contour *direction* (+1 up, -1 down, 0 neighbor).

    On weak beats dissonance is tolerated (passing tones), but consonance
    is still mildly preferred.
    """
    if direction == 0:
        # Neighbor note: step away briefly
        for delta in (1, -1, 2, -2):
            p = current + delta
            if lo <= p <= hi and (not scale_pcs or (p % 12) in scale_pcs):
                if other_pitch is None or _is_consonant(p, other_pitch):
                    return p
        # No consonant neighbor — try any scale neighbor
        for delta in (1, -1, 2, -2):
            p = current + delta
            if lo <= p <= hi and (not scale_pcs or (p % 12) in scale_pcs):
                return p
        return current

    step = 1 if direction > 0 else -1

    # If near the range edge, reverse direction to avoid getting stuck
    center = (lo + hi) / 2
    if direction > 0 and current >= hi - 3:
        step = -1       # forced reversal at top
    elif direction < 0 and current <= lo + 3:
        step = 1        # forced reversal at bottom

    # Collect candidate steps, score them lightly
    best_p, best_sc = current, -9999.0
    for delta in range(1, 5):
        p = current + step * delta
        if p < lo or p > hi:
            continue

        sc = 0.0
        # Scale-tone preference
        if scale_pcs and (p % 12) in scale_pcs:
            sc += 6.0
        else:
            sc -= 3.0
        # Smaller step preferred
        sc -= delta * 2.0
        # Register gravity: pull toward center (avoid extremes)
        dist = abs(p - center) / max(1, (hi - lo) / 2)
        sc -= dist * 3.0
        # Consonance: STRONG avoidance of dissonance (must outweigh scale bonus)
        if other_pitch is not None:
            iv = interval_class(p - other_pitch)
            if iv in (3, 4, 8, 9):
                sc += 4.0       # imperfect consonance — BEST
            elif iv in (0, 7):
                sc += 2.0       # perfect consonance
            elif iv in (1, 11):
                sc -= 12.0      # m2/M7 — harshest, must always lose
            elif iv == 6:
                sc -= 10.0      # tritone
            elif iv in (2, 10):
                sc -= 8.0       # M2/m7
            elif iv == 5:
                sc -= 6.0       # P4 (mild dissonance in 2-voice)

        if sc > best_sc:
            best_sc = sc
            best_p = p

    if best_p == current:
        # Range limit: try opposite direction
        for delta in range(1, 4):
            p = current - step * delta
            if lo <= p <= hi and (not scale_pcs or (p % 12) in scale_pcs):
                return p

    return best_p


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Core: Harmonic melody generator
# ═══════════════════════════════════════════════════════════════════════════

def _generate_harmonic_melody(
    subject: Subject,
    voice: int,
    start_offset: float,
    duration: float,
    existing_voices: dict[int, list[FugueNote]],
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None = None,
    start_pitch: int | None = None,
    role: EntryRole = EntryRole.FREE_COUNTERPOINT,
    rhythm_override: list[float] | None = None,
) -> list[FugueNote]:
    """
    Generate counterpoint whose pitches are BORN from the harmonic skeleton.

    Algorithm:
      1. Build a time grid from rhythm_override (or subject rhythm).
      2. Extract contour (up/down) from the subject's interval sequence.
      3. At each time slot:
         a) Follow contour with a diatonic step  →  *ideal pitch*
         b) If strong beat: SNAP ideal to nearest chord tone
         c) If weak beat: keep ideal (passing / neighbor tone OK)
      4. Check consonance with other voice at every step.

    rhythm_override: if given, uses this rhythm instead of the subject's.
                     Use _cp_rhythm(subject) for slower accompaniment.
    """
    if duration <= 0:
        return []

    lo, hi = config.voice_ranges.get(voice, (36, 84))
    scale_pcs = config.scale_pcs

    # ── Build time grid ──
    rhythm = rhythm_override if rhythm_override else _extract_rhythm(subject)
    if not rhythm:
        return []

    time_grid: list[tuple[float, float]] = []  # (offset, dur)
    t = start_offset
    end = start_offset + duration
    r_idx = 0
    while t < end - 0.05:
        dur = rhythm[r_idx % len(rhythm)]
        dur = min(dur, end - t)
        if dur < 0.05:
            break
        time_grid.append((t, dur))
        t += dur
        r_idx += 1

    if not time_grid:
        return []

    # ── Extract contour ──
    contour = _extract_contour(subject)
    if not contour:
        contour = [1, -1]  # default: oscillate

    # ── Starting pitch ──
    current = start_pitch if start_pitch is not None else (lo + hi) // 2
    current = max(lo + 2, min(hi - 2, current))

    # ── Generate notes ──
    result: list[FugueNote] = []

    for i, (t, dur) in enumerate(time_grid):
        c_dir = contour[i % len(contour)]
        other_p = _get_sounding_pitch(existing_voices, voice, t)
        strength = _beat_strength(t)

        # Step A: follow contour with a diatonic step
        ideal = _diatonic_step(current, c_dir, scale_pcs, lo, hi, other_p)

        # Step B: on strong beats *or* motif-cycle starts, snap to chord tone
        motif_start = (i % len(rhythm) == 0) if rhythm else False
        if (strength >= 1 or motif_start) and harmonic_skeleton:
            chord = get_chord_at(harmonic_skeleton, t)
            chord_pcs = chord.chord_pcs if chord else frozenset()
            ideal = _snap_to_chord_tone(
                ideal, chord_pcs, scale_pcs, other_p, lo, hi,
            )

        result.append(FugueNote(
            pitch=ideal, duration=dur, voice=voice,
            offset=t, role=role,
        ))
        current = ideal

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Section generators
# ═══════════════════════════════════════════════════════════════════════════

# -------- Exposition --------

def _invention_exposition(
    plan: FuguePlan,
    subject: Subject,
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None,
    answer_trans: int,
) -> tuple[dict[int, list[FugueNote]], list[FugueNote]]:
    """
    1. Voice 0: subject at original pitch
    2. Voice 1: theme transposed DOWN (octave or 5th)
    3. Voice 0 gets harmonic-melody CP while Voice 1 plays
    4. That CP is captured as the countersubject
    """
    voices: dict[int, list[FugueNote]] = {0: [], 1: []}

    # 1. Subject in Voice 0
    subj_notes = place_subject(subject, voice=0, start_offset=0.0, transposition=0)
    voices[0].extend(subj_notes)

    # 2. Answer in Voice 1 (transposed down)
    ans_notes = place_subject(
        subject, voice=1, start_offset=subject.duration,
        transposition=answer_trans,
    )
    ans_notes = _adjust_to_voice_range(ans_notes, 1, config)
    # Re-tag as ANSWER
    ans_notes = [
        FugueNote(pitch=n.pitch, duration=n.duration, voice=n.voice,
                  offset=n.offset, role=EntryRole.ANSWER, transform=n.transform)
        for n in ans_notes
    ]
    voices[1].extend(ans_notes)

    # 3. Counterpoint for Voice 0 while Voice 1 plays answer.
    #    Use INVERTED subject head as a running motif sequence —
    #    this creates the imitative texture Bach uses from bar 1.
    last_subj_p = subj_notes[-1].pitch if subj_notes else 72
    motif = _extract_motif(subject, max_notes=4)
    inv_motif = _invert_motif(motif)
    cp_notes = _generate_sequence(
        inv_motif, last_subj_p, subject.duration, subject.duration,
        voice=0, scale_pcs=config.scale_pcs,
        lo=config.voice_ranges.get(0, (60, 84))[0],
        hi=config.voice_ranges.get(0, (60, 84))[1],
        direction=-1,
        other_voice_notes=ans_notes,
    )
    voices[0].extend(cp_notes)

    # 4. Capture countersubject
    cs_base = subject.duration
    countersubject = [
        FugueNote(pitch=n.pitch, duration=n.duration, voice=0,
                  offset=n.offset - cs_base, role=EntryRole.COUNTERSUBJECT)
        for n in cp_notes
    ]

    return voices, countersubject


# -------- Middle Entry --------

def _generate_invention_entry(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    countersubject: list[FugueNote],
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None,
) -> None:
    """
    One voice states the theme, the other plays motif-derived counterpoint.

    The CP voice uses the INVERTED subject head as a running sequence,
    creating the imitative texture that characterises Bach's entries.
    Falls back to sustained harmonic support if the inverted sequence
    would create too many dissonances.
    """
    trans = _key_to_transposition(section.key_area, plan.key_signature)

    # Place theme
    theme_notes_all = []
    for entry in section.entries:
        subj_notes = place_subject(
            plan.subject, entry.voice, entry.start_offset,
            transposition=trans, transform=entry.transform,
        )
        subj_notes = _adjust_to_voice_range(subj_notes, entry.voice, config)
        voices.setdefault(entry.voice, []).extend(subj_notes)
        theme_notes_all.extend(subj_notes)

    # CP voice: inverted motif sequence against the theme
    entry_voices = {e.voice for e in section.entries}
    motif = _extract_motif(plan.subject, max_notes=4)
    inv_motif = _invert_motif(motif)

    for v in range(2):
        if v in entry_voices:
            continue
        lo, hi = config.voice_ranges.get(v, (36, 84))
        prev_notes = sorted(voices.get(v, []), key=lambda n: n.offset)
        start_p = prev_notes[-1].pitch if prev_notes else (lo + hi) // 2

        # Use inverted motif sequence as CP
        cp_notes = _generate_sequence(
            inv_motif, start_p, section.start_offset,
            section.estimated_duration, voice=v,
            scale_pcs=config.scale_pcs, lo=lo, hi=hi,
            direction=-1,
            other_voice_notes=theme_notes_all,
        )

        # If sequence is too short (didn't fill), pad with sustained notes
        seq_end = cp_notes[-1].offset + cp_notes[-1].duration if cp_notes else section.start_offset
        if seq_end < section.start_offset + section.estimated_duration - 0.1:
            pad_start = seq_end
            pad_dur = section.start_offset + section.estimated_duration - seq_end
            pad_p = cp_notes[-1].pitch if cp_notes else start_p
            pad = _generate_sustained_bass(
                pad_start, pad_dur, v, lo, hi, pad_p,
                config.scale_pcs, harmonic_skeleton, theme_notes_all,
            )
            cp_notes.extend(pad)

        voices.setdefault(v, []).extend(cp_notes)


# -------- Episode --------

def _generate_invention_episode(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None,
) -> None:
    """
    Bach-style motivic episode:

      The subject head motif is SEQUENCED (repeated at successively
      lower/higher diatonic pitch levels). One voice runs the sequence
      while the other provides sustained harmonic support.
      Halfway through, the voices swap roles.

    This is the single most important technique in Bach's inventions.
    """
    ep_start = section.start_offset
    ep_dur = section.estimated_duration
    half = round(ep_dur / 2.0 / 0.25) * 0.25
    if half < 0.5:
        half = ep_dur

    # Extract head motif (first ~4 notes) — the building block of episodes
    motif = _extract_motif(plan.subject, max_notes=4)
    inv_motif = _invert_motif(motif)
    scale_pcs = config.scale_pcs

    # Determine sequence direction from section's key movement
    # Descending sequences are most common (moving toward dominant)
    seq_dir = -1

    # --- First half: V0 runs sequence, V1 sustains ---
    lo0, hi0 = config.voice_ranges.get(0, (60, 84))
    lo1, hi1 = config.voice_ranges.get(1, (36, 60))

    prev0 = sorted(voices.get(0, []), key=lambda n: n.offset)
    prev1 = sorted(voices.get(1, []), key=lambda n: n.offset)
    start_p0 = prev0[-1].pitch if prev0 else (lo0 + hi0) // 2
    start_p1 = prev1[-1].pitch if prev1 else (lo1 + hi1) // 2

    # V0: descending sequence with subject head motif
    seq_notes_0 = _generate_sequence(
        motif, start_p0, ep_start, half, voice=0,
        scale_pcs=scale_pcs, lo=lo0, hi=hi0, direction=seq_dir,
    )
    # V1: sustained harmonic support, checking against V0's sequence
    bass_notes_1 = _generate_sustained_bass(
        ep_start, half, voice=1, lo=lo1, hi=hi1,
        start_pitch=start_p1, scale_pcs=scale_pcs,
        harmonic_skeleton=harmonic_skeleton,
        other_voice_notes=seq_notes_0,
    )

    voices.setdefault(0, []).extend(seq_notes_0)
    voices.setdefault(1, []).extend(bass_notes_1)

    # --- Second half: V1 runs INVERTED sequence, V0 sustains ---
    if half < ep_dur:
        remaining = ep_dur - half
        # V1's starting pitch for second half
        sp1 = bass_notes_1[-1].pitch if bass_notes_1 else start_p1
        sp0 = seq_notes_0[-1].pitch if seq_notes_0 else start_p0

        # V1: ascending sequence with inverted motif (contrary motion)
        seq_notes_1 = _generate_sequence(
            inv_motif, sp1, ep_start + half, remaining, voice=1,
            scale_pcs=scale_pcs, lo=lo1, hi=hi1, direction=-seq_dir,
        )
        # V0: sustained support
        bass_notes_0 = _generate_sustained_bass(
            ep_start + half, remaining, voice=0, lo=lo0, hi=hi0,
            start_pitch=sp0, scale_pcs=scale_pcs,
            harmonic_skeleton=harmonic_skeleton,
            other_voice_notes=seq_notes_1,
        )

        voices[1].extend(seq_notes_1)
        voices[0].extend(bass_notes_0)


# -------- Coda --------

def _generate_invention_coda(
    subject: Subject,
    voices: dict[int, list[FugueNote]],
    start_offset: float,
    duration: float,
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None,
) -> None:
    """Motif lead-in  ->  V chord  ->  I chord."""
    ks = subject.key_signature
    tonic_pc = key_name_to_pc(ks)
    dominant_pc = (tonic_pc + 7) % 12
    is_minor = ks[0].islower()
    leading_tone_pc = (tonic_pc - 1) % 12

    cadence_beats = min(2.0, duration)
    free_dur = max(0, duration - cadence_beats)

    # Lead-in — descending motif sequence winding toward cadence
    if free_dur > 0.5:
        motif = _extract_motif(subject, max_notes=4)
        lo0, hi0 = config.voice_ranges.get(0, (60, 84))
        lo1, hi1 = config.voice_ranges.get(1, (36, 60))

        prev0 = sorted(voices.get(0, []), key=lambda n: n.offset)
        prev1 = sorted(voices.get(1, []), key=lambda n: n.offset)
        sp0 = prev0[-1].pitch if prev0 else (lo0 + hi0) // 2
        sp1 = prev1[-1].pitch if prev1 else (lo1 + hi1) // 2

        # V0: descending sequence
        seq0 = _generate_sequence(
            motif, sp0, start_offset, free_dur, voice=0,
            scale_pcs=config.scale_pcs, lo=lo0, hi=hi0, direction=-1,
        )
        # V1: sustained support under V0's sequence
        bass1 = _generate_sustained_bass(
            start_offset, free_dur, voice=1, lo=lo1, hi=hi1,
            start_pitch=sp1, scale_pcs=config.scale_pcs,
            harmonic_skeleton=harmonic_skeleton,
            other_voice_notes=seq0,
        )
        voices.setdefault(0, []).extend(seq0)
        voices.setdefault(1, []).extend(bass1)

    # V -> I cadence
    cad_off = start_offset + free_dur
    cad_half = cadence_beats / 2.0
    for v in range(2):
        lo, hi = config.voice_ranges.get(v, (48, 72))
        mid = (lo + hi) // 2

        # V chord
        v_pc = leading_tone_pc if v == 0 else dominant_pc
        vp = _nearest_pc(mid, v_pc, lo, hi)
        voices.setdefault(v, []).append(FugueNote(
            pitch=vp, duration=cad_half, voice=v,
            offset=cad_off, role=EntryRole.CODETTA,
        ))

        # I chord
        ip = _nearest_pc(mid, tonic_pc, lo, hi)
        voices.setdefault(v, []).append(FugueNote(
            pitch=ip, duration=cad_half, voice=v,
            offset=cad_off + cad_half, role=EntryRole.CODETTA,
        ))


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _adjust_to_voice_range(
    notes: list[FugueNote], voice: int, config: GenerationConfig,
) -> list[FugueNote]:
    """Shift notes by octave(s) so their average pitch is within voice range."""
    pitches = [n.pitch for n in notes if not n.is_rest]
    if not pitches:
        return notes

    lo, hi = config.voice_ranges.get(voice, (36, 84))
    avg = sum(pitches) / len(pitches)
    center = (lo + hi) / 2

    shift = 0
    while avg + shift > hi:
        shift -= 12
    while avg + shift < lo:
        shift += 12

    best_shift = shift
    for alt in (shift - 12, shift, shift + 12):
        if lo <= avg + alt <= hi:
            if abs(avg + alt - center) < abs(avg + best_shift - center):
                best_shift = alt

    if best_shift == 0:
        return notes

    return [
        FugueNote(pitch=n.pitch + best_shift if not n.is_rest else n.pitch,
                  duration=n.duration, voice=n.voice, offset=n.offset,
                  role=n.role, transform=n.transform)
        for n in notes
    ]


def _nearest_pc(center: int, pc: int, lo: int, hi: int) -> int:
    best, best_d = center, 999
    for p in range(lo, hi + 1):
        if p % 12 == pc and abs(p - center) < best_d:
            best, best_d = p, abs(p - center)
    return best


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def generate_invention(
    plan: FuguePlan,
    config: GenerationConfig | None = None,
    answer_at_fifth: bool = False,
) -> dict[int, list[FugueNote]]:
    """
    Generate a complete Two-Part Invention.

    Harmony-first pipeline:
      1. Build harmonic skeleton for the entire piece
      2. Exposition: subject + answer (correct octave) + harmonic CP
      3. Episodes: harmonic melody with sequential modulation
      4. Middle entries: theme + harmonic CP
      5. Coda: harmonic lead-in + V->I cadence
    """
    config = config or _invention_config()

    if not plan.exposition:
        return {}

    answer_trans = -5 if answer_at_fifth else -12
    config = _adapt_invention_ranges(config, plan.subject, answer_trans)

    tonic_pc = key_name_to_pc(plan.key_signature)
    is_minor = key_is_minor(plan.key_signature)
    total_dur = sum(s.estimated_duration for s in plan.sections)
    full_skeleton = generate_harmonic_skeleton(
        tonic_pc, is_minor, 0.0, total_dur,
        beats_per_chord=2.0, section_type="normal",
    )

    # 1. Exposition
    voices, countersubject = _invention_exposition(
        plan, plan.subject, config, full_skeleton, answer_trans,
    )

    # 2-N. Sections
    for section in plan.sections:
        if section.section_type == SectionType.EXPOSITION:
            continue
        elif section.section_type == SectionType.EPISODE:
            _generate_invention_episode(plan, section, voices, config, full_skeleton)
        elif section.section_type == SectionType.MIDDLE_ENTRY:
            _generate_invention_entry(plan, section, voices, countersubject,
                                      config, full_skeleton)
        elif section.section_type == SectionType.CODA:
            _generate_invention_coda(plan.subject, voices,
                                     section.start_offset,
                                     section.estimated_duration,
                                     config, full_skeleton)

    # Post-process (run dissonance fix twice for iterative convergence)
    voices = _postprocess_leap_resolution(voices)
    voices = _postprocess_fix_parallels(voices)
    voices = _postprocess_fix_dissonances(voices, scale_pcs=config.scale_pcs)
    voices = _postprocess_fix_dissonances(voices, scale_pcs=config.scale_pcs)

    return voices
