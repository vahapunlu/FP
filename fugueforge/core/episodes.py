"""
FugueForge Episode Generation

Sequence-based episode generation derived from subject material.
Supports multiple source types: subject head, tail, inversion,
and countersubject fragments.
"""

from __future__ import annotations

from typing import Optional

from .representation import EntryRole, FugueNote, Subject
from .rules import is_consonance
from .candidates import (
    GenerationConfig,
    _check_hidden_fifth_octave,
)


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
                    if is_strong:
                        sc += cons * 3

                    # Leap resolution
                    if result and abs(prev_interval) >= 5:
                        mel = p - prev_p
                        if abs(mel) <= 2 and (mel * prev_interval < 0):
                            sc += 8
                        elif abs(mel) <= 2:
                            sc += 2
                        elif abs(mel) > 4:
                            sc -= 6

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
                                prev_t = note_offset - 0.5
                                op_prev = oc
                                for vn in existing_voices.get(ov, []):
                                    if vn.offset <= prev_t < vn.offset + vn.duration:
                                        op_prev = vn.pitch
                                        break
                                if _check_hidden_fifth_octave(
                                    voice, p, prev_p, ov, op_prev, oc,
                                ):
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
