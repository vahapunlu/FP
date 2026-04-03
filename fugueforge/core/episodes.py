"""
FugueForge Episode Generation

Sequence-based episode generation derived from subject material.
Supports multiple source types: subject head, tail, inversion,
and countersubject fragments.
"""

from __future__ import annotations

import random
from typing import Optional

from .representation import EntryRole, FugueNote, Subject
from .rules import is_consonance
from .candidates import (
    GenerationConfig,
    _check_hidden_fifth_octave,
)
from .harmony import ChordLabel, get_chord_at


# ---------------------------------------------------------------------------
# Rhythmic variation patterns for episode sequences
# ---------------------------------------------------------------------------

def _vary_rhythm(motif_notes: list[FugueNote], seq_idx: int) -> list[FugueNote]:
    """
    Apply subtle rhythmic variation to a motif repetition.
    Bach's sequences vary rhythm slightly to maintain interest.
    seq_idx 0 = original, 1+ = progressively more varied.
    """
    if seq_idx == 0 or not motif_notes:
        return motif_notes  # first statement: original rhythm

    varied = []
    for i, mn in enumerate(motif_notes):
        dur = mn.duration
        # Apply subtle variation based on sequence index and note position
        if seq_idx >= 2 and i == len(motif_notes) - 1:
            # Last note of later sequences: slightly longer (phrase breathing)
            dur = dur * 1.5
        elif seq_idx >= 1 and i == 0 and random.random() < 0.3:
            # First note occasionally lengthened (agogic accent)
            dur = dur * 1.5
        elif seq_idx >= 1 and random.random() < 0.15:
            # Occasional dotted rhythm
            dur = dur * (1.5 if random.random() < 0.5 else 0.75)

        # Ensure minimum duration
        dur = max(0.25, dur)

        varied.append(FugueNote(
            pitch=mn.pitch,
            duration=dur,
            voice=mn.voice,
            offset=mn.offset,
            role=mn.role,
        ))
    return varied


def _vary_interval(base_pitch: int, motif_pitch: int, seq_idx: int,
                   scale_pcs: frozenset[int]) -> int:
    """
    Apply subtle intervallic variation to sequence repetitions.
    Returns adjusted pitch. Later repetitions have more freedom.
    """
    if seq_idx == 0:
        return motif_pitch  # first statement: exact

    # Small chance of ornamental variation (neighbor note, passing tone)
    if random.random() < 0.12 * seq_idx:  # 12% per repetition
        adj = random.choice([-1, 1, -2, 2])
        varied = motif_pitch + adj
        # Prefer in-scale adjustments
        if scale_pcs and (varied % 12) not in scale_pcs:
            varied = motif_pitch - adj  # try opposite
            if scale_pcs and (varied % 12) not in scale_pcs:
                varied = motif_pitch  # give up, use original
        return varied

    return motif_pitch


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
    harmonic_skeleton: Optional[list[ChordLabel]] = None,
) -> list[FugueNote]:
    """
    Generate an episode by creating a sequence pattern from subject material.

    Source options (per Burrows thesis — thematic episodes):
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

    motif_dur = motif_notes[-1].offset + motif_notes[-1].duration - motif_notes[0].offset
    if motif_dur <= 0:
        motif_dur = sum(n.duration for n in motif_notes)
    if motif_dur == 0:
        return []

    result: list[FugueNote] = []
    offset = start_offset
    transposition = 0

    for seq_idx in range(sequence_count):
        if offset >= start_offset + duration:
            break

        # Apply rhythmic variation to this sequence repetition
        varied_motif = _vary_rhythm(motif_notes, seq_idx)

        # Recalculate motif duration for varied rhythm
        if varied_motif:
            seq_dur = sum(n.duration for n in varied_motif)
        else:
            seq_dur = motif_dur

        motif_base = varied_motif[0].offset if varied_motif else 0.0
        note_time = offset
        for mi, mn in enumerate(varied_motif):
            if note_time >= start_offset + duration:
                break

            # Apply intervallic variation
            raw_pitch = mn.pitch + transposition
            candidate_pitch = _vary_interval(
                mn.pitch, raw_pitch, seq_idx, config.scale_pcs,
            )
            # Clamp to voice range
            candidate_pitch = max(lo, min(hi, candidate_pitch))

            # Quick consonance check against other voices
            other_current: dict[int, int] = {}
            for v, v_notes in existing_voices.items():
                if v == voice:
                    continue
                for n in v_notes:
                    if n.offset <= note_time < n.offset + n.duration:
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

                    mel = p - prev_p
                    abs_mel = abs(mel)

                    # Melodic quality — stepwise strongly preferred
                    if abs_mel <= 2:
                        sc = 8.0   # step: excellent
                    elif abs_mel <= 4:
                        sc = 3.0   # third: good
                    elif abs_mel <= 7:
                        sc = -2.0  # P4/P5: acceptable
                    else:
                        sc = -8.0  # large leap: bad

                    # Consecutive leap penalty
                    if result and abs(prev_interval) >= 3 and abs_mel >= 3:
                        sc -= 8.0
                        if (mel > 0) == (prev_interval > 0):
                            sc -= 4.0  # same direction leaps: worse

                    # Consonance score
                    cons = sum(
                        1 for op in other_current.values()
                        if op != -1 and is_consonance(p - op)
                    )
                    is_strong = abs(note_time - round(note_time)) < 0.01
                    sc += cons * 2.5
                    if is_strong:
                        sc += cons * 2.5

                    # Scale/tonality: strongly prefer in-key pitches
                    if config.scale_pcs:
                        if p % 12 in config.scale_pcs:
                            sc += 7
                        else:
                            sc -= 10 if is_strong else 5

                    # Chord tone bonus from harmonic skeleton
                    if harmonic_skeleton:
                        chord = get_chord_at(harmonic_skeleton, note_time)
                        if chord:
                            if p % 12 in chord.chord_pcs:
                                sc += 5 if is_strong else 2.5
                            elif is_strong:
                                sc -= 3

                    # Leap resolution
                    if result and abs(prev_interval) >= 5:
                        if abs_mel <= 2 and (mel * prev_interval < 0):
                            sc += 10
                        elif abs_mel <= 2:
                            sc += 3
                        elif abs_mel > 4:
                            sc -= 8

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
                                prev_t = note_time - 0.5
                                op_prev = oc
                                for vn in existing_voices.get(ov, []):
                                    if vn.offset <= prev_t < vn.offset + vn.duration:
                                        op_prev = vn.pitch
                                        break
                                if _check_hidden_fifth_octave(
                                    voice, p, prev_p, ov, op_prev, oc,
                                ):
                                    sc -= 4

                    # Prefer minimal deviation from original (but melodic quality matters more)
                    sc -= abs(adj) * 0.5

                    # Register gravity: pull toward voice center
                    voice_center = (lo + hi) / 2
                    range_half = (hi - lo) / 2
                    if range_half > 0:
                        sc -= 1.5 * (abs(p - voice_center) / range_half) ** 2

                    if sc > best_score:
                        best_score = sc
                        best_pitch = p
                candidate_pitch = best_pitch

            result.append(FugueNote(
                pitch=candidate_pitch,
                duration=mn.duration,
                voice=voice,
                offset=note_time,
                role=EntryRole.EPISODE_MATERIAL,
            ))
            note_time += mn.duration

        offset += seq_dur
        transposition += sequence_interval

    return result
