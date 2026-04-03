"""
FugueForge Post-Processors

Post-processing passes applied after initial generation to fix
counterpoint violations: unresolved leaps, parallel 5ths/8ves,
strong-beat dissonances, and climax placement.
"""

from __future__ import annotations

from .representation import EntryRole, FugueNote
from .rules import (
    MotionType,
    build_snapshots,
    classify_motion,
    interval_class,
    is_consonance,
    is_dissonance,
)
from .candidates import _check_parallel_motion


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
                continue

            resolution = non_rest[i + 2].pitch - non_rest[i + 1].pitch
            if abs(resolution) <= 2 and (
                (leap > 0 and resolution < 0)
                or (leap < 0 and resolution > 0)
            ):
                continue

            target = non_rest[i + 2]
            if target.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
                continue

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
                        op_prev = op_cur
                        for n_check in voices.get(ov, []):
                            if n_check.offset <= prev_offset < n_check.offset + n_check.duration:
                                op_prev = n_check.pitch
                                break
                        if op_prev == op_cur or prev_pitch == new_pitch:
                            continue
                        if _check_parallel_motion(
                            v, new_pitch, prev_pitch, ov, op_prev, op_cur,
                        ):
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


# ---------------------------------------------------------------------------
# Dissonance fixer
# ---------------------------------------------------------------------------

def _postprocess_fix_dissonances(
    voices: dict[int, list[FugueNote]],
    scale_pcs: frozenset[int] = frozenset(),
) -> dict[int, list[FugueNote]]:
    """
    Scan for dissonances in modifiable notes and fix them
    by shifting 1-2 semitones to the nearest consonance, if possible
    without creating voice crossing. Prefers in-scale adjustments.

    Checks EVERY vertical sounding moment: both note-onset and sustained
    overlap (a long note sounding while the other voice moves).
    """
    # Build a quick lookup: for each voice, list of (offset, end, pitch, index)
    def _sounding_at(v_notes: list[FugueNote], t: float) -> int | None:
        for i, n in enumerate(v_notes):
            if not n.is_rest and n.offset <= t < n.offset + n.duration:
                return i
        return None

    for v, v_notes in voices.items():
        for j, n in enumerate(v_notes):
            if n.is_rest:
                continue
            if n.role not in (EntryRole.FREE_COUNTERPOINT, EntryRole.EPISODE_MATERIAL):
                continue

            # Collect ALL other-voice pitches that sound during this note's lifespan
            other_pitches_at_onset: dict[int, int] = {}
            other_during: list[tuple[int, int]] = []  # (other_voice, other_pitch)
            for ov, ov_notes in voices.items():
                if ov == v:
                    continue
                for on in ov_notes:
                    if on.is_rest:
                        continue
                    # Does this other note overlap with n at all?
                    if on.offset < n.offset + n.duration and on.offset + on.duration > n.offset:
                        other_during.append((ov, on.pitch))
                        # Also check onset specifically
                        if on.offset <= n.offset < on.offset + on.duration:
                            other_pitches_at_onset[ov] = on.pitch

            if not other_during:
                continue

            # Check for dissonance at onset OR during sustain
            has_dissonance = False
            for _, op in other_during:
                if is_dissonance(n.pitch - op):
                    has_dissonance = True
                    break

            is_strong = abs(n.offset - round(n.offset)) < 0.01
            is_chromatic = scale_pcs and (n.pitch % 12 not in scale_pcs)
            if not has_dissonance and not (is_chromatic and is_strong):
                continue

            best_adj = None
            best_score = -999
            for adj in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
                if adj == 0 and has_dissonance:
                    continue
                new_p = n.pitch + adj
                # Must be consonant with ALL notes that overlap during lifespan
                all_cons = all(
                    is_consonance(new_p - op) for _, op in other_during
                )
                if not all_cons:
                    continue
                crossing = False
                for ov, op in other_during:
                    if v < ov and new_p < op:
                        crossing = True
                    elif v > ov and new_p > op:
                        crossing = True
                if crossing:
                    continue
                # Score: prefer in-scale, minimal movement
                sc = -abs(adj) * 2
                if scale_pcs and (new_p % 12 in scale_pcs):
                    sc += 10
                if best_adj is None or sc > best_score:
                    best_adj = adj
                    best_score = sc

            if best_adj is not None and best_adj != 0:
                v_notes[j] = FugueNote(
                    pitch=n.pitch + best_adj,
                    duration=n.duration,
                    voice=n.voice,
                    offset=n.offset,
                    role=n.role,
                )

    return voices


# ---------------------------------------------------------------------------
# Climax placement post-processor
# ---------------------------------------------------------------------------

def _postprocess_climax_placement(
    voices: dict[int, list[FugueNote]],
) -> dict[int, list[FugueNote]]:
    """
    Ensure the highest note (climax) occurs in the latter half of the piece.
    If the current climax is too early (<40% position), find a modifiable note
    in the 55-80% range and raise it above the current highest pitch.
    """
    all_notes = [
        (v, n) for v, v_notes in voices.items() for n in v_notes if not n.is_rest
    ]
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

    target_start = total_dur * 0.55
    target_end = total_dur * 0.80
    target_pitch = highest_pitch + 1

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


# ---------------------------------------------------------------------------
# Parallel 5ths/8ves fixer
# ---------------------------------------------------------------------------

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
                if int2 not in (0, 7):
                    continue
                if int1 != int2:
                    continue

                # Found a parallel 5th/8ve/unison — try to fix
                target_offset = s2.offset
                for fix_voice in [vi, vj]:
                    v_notes = voices.get(fix_voice, [])
                    for j, n in enumerate(v_notes):
                        if n.is_rest:
                            continue
                        if abs(n.offset - target_offset) > 0.01:
                            continue
                        if n.role not in (
                            EntryRole.FREE_COUNTERPOINT,
                            EntryRole.EPISODE_MATERIAL,
                        ):
                            continue

                        for adj in [1, -1]:
                            new_pitch = n.pitch + adj
                            new_int = interval_class(
                                (new_pitch if fix_voice == vi else p2_hi)
                                - (p2_lo if fix_voice == vi else new_pitch)
                            )
                            if new_int == int2:
                                continue
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
                        break
                    else:
                        continue
                    break

    return voices
