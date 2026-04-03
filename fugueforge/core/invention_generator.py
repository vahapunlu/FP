"""
FugueForge Invention Generator

Generates Two-Part Inventions using the existing infrastructure
but with tighter constraints optimized for 2-voice texture:

  - Stronger consonance enforcement (2 voices = every interval audible)
  - Higher stepwise preference (smoother melodic lines)
  - Tighter voice independence (avoid parallel motion)
  - Better register separation between voices

Reuses: candidates, counterpoint, placement, episodes, exposition, coda, harmony
"""

from __future__ import annotations

import copy
from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    FuguePlan,
    SectionType,
    TransformType,
)
from .candidates import GenerationConfig
from .voice_utils import (
    _adapt_voice_ranges,
    _key_to_transposition,
    key_to_scale_pcs,
    voices_to_score,
)
from .placement import _adjust_placement_octave, place_subject
from .counterpoint import _place_countersubject, generate_free_counterpoint
from .episodes import generate_episode
from .exposition import generate_exposition
from .coda import generate_coda
from .harmony import (
    generate_harmonic_skeleton,
    key_name_to_pc,
    key_is_minor,
    ChordLabel,
)
from .postprocess import (
    _postprocess_climax_placement,
    _postprocess_fix_dissonances,
    _postprocess_fix_parallels,
    _postprocess_leap_resolution,
)


# ---------------------------------------------------------------------------
# Invention-specific config: tighter constraints for 2-voice clarity
# ---------------------------------------------------------------------------

def _invention_config() -> GenerationConfig:
    """
    Return a GenerationConfig tuned for 2-voice inventions.

    Key differences from fugue config:
      - Higher consonance weight (every vertical interval is audible)
      - Higher dissonance penalty
      - Stronger stepwise preference
      - Lower temperature (more predictable choices)
      - Tighter leap penalties
    """
    return GenerationConfig(
        beam_width=8,
        max_candidates_per_step=24,
        temperature=0.30,          # more predictable (fugue: 0.40)
        step_resolution=0.5,
        prefer_stepwise=0.8,       # high stepwise (fugue: 0.7)
        max_leap=10,               # smaller max leap (fugue: 12)

        # Harmony — stronger for 2 voices
        consonance_weight=5.0,     # up from 3.0
        dissonance_penalty=12.0,   # up from 8.0
        chord_tone_bonus=6.0,      # up from 5.0
        non_chord_penalty=3.0,     # up from 2.0
        scale_bonus=10.0,          # up from 8.0
        chromatic_penalty=6.0,     # up from 4.0

        # Counterpoint — stricter
        parallel_veto=True,
        crossing_veto=True,
        leap_resolution_bonus=3.0,
        contrary_motion_bonus=4.0, # up from 2.0

        # Melodic quality — Phase 2 features
        stepwise_bonus=10.0,       # up from 8.0
        third_bonus=4.0,           # up from 3.5
        fourth_fifth_penalty=2.0,  # up from 1.0
        large_leap_penalty=10.0,   # up from 8.0
        consecutive_leap_penalty=12.0,  # up from 10.0
        direction_momentum=3.0,
        direction_reversal=4.0,
        register_gravity=2.5,      # up from 2.0
        phrase_length=4.0,
    )


# ---------------------------------------------------------------------------
# Voice range: clear separation for 2 voices
# ---------------------------------------------------------------------------

def _adapt_invention_ranges(
    config: GenerationConfig,
    subject,
) -> GenerationConfig:
    """
    Set voice ranges for a 2-voice invention.

    Voice 0 (RH / treble): subject range + padding above
    Voice 1 (LH / bass):   lower register, clear separation
    """
    subj_lo, subj_hi = subject.pitch_range
    padding = 6

    new_ranges = dict(config.voice_ranges)

    # Voice 0 (treble): from subject low to well above subject high
    new_ranges[0] = (subj_lo - 2, subj_hi + padding + 4)

    # Voice 1 (bass): clearly below, with enough range for independence
    bass_hi = subj_lo + 5   # overlap zone limited
    bass_lo = max(36, subj_lo - 19)  # at least an octave below
    min_span = 18
    if bass_hi - bass_lo < min_span:
        bass_lo = max(36, bass_hi - min_span)
    new_ranges[1] = (bass_lo, bass_hi)

    new_config = GenerationConfig(
        **{**config.__dict__, 'voice_ranges': new_ranges,
           'scale_pcs': config.scale_pcs or key_to_scale_pcs(subject.key_signature)}
    )
    return new_config


# ---------------------------------------------------------------------------
# Energy curve for inventions (simpler than fugue dramaturgy)
# ---------------------------------------------------------------------------

def _invention_energy(
    config: GenerationConfig,
    progress: float,
    section_type: SectionType,
) -> GenerationConfig:
    """
    Gentle energy arc for inventions:
      0.0-0.3: Calm, moderate
      0.3-0.7: Building, slightly wider register
      0.7-0.9: Peak energy
      0.9-1.0: Resolution, narrowing
    """
    dc = copy.copy(config)
    dc.voice_ranges = dict(config.voice_ranges)

    if section_type == SectionType.CODA:
        dc.temperature = 0.25
        dc.phrase_length = 6.0
        return dc

    if progress < 0.3:
        dc.temperature = 0.28
        dc.phrase_length = 4.5
    elif progress < 0.7:
        energy = (progress - 0.3) / 0.4
        dc.temperature = 0.28 + energy * 0.07
        dc.phrase_length = 4.5 - energy * 1.0
        for v in dc.voice_ranges:
            lo, hi = dc.voice_ranges[v]
            expand = int(energy * 2)
            dc.voice_ranges[v] = (max(36, lo - expand), hi + expand)
    elif progress < 0.9:
        dc.temperature = 0.35
        dc.phrase_length = 3.5
        for v in dc.voice_ranges:
            lo, hi = dc.voice_ranges[v]
            dc.voice_ranges[v] = (max(36, lo - 2), hi + 2)
    else:
        dc.temperature = 0.25
        dc.phrase_length = 5.0

    return dc


# ---------------------------------------------------------------------------
# Main invention generator
# ---------------------------------------------------------------------------

def generate_invention(
    plan: FuguePlan,
    config: Optional[GenerationConfig] = None,
) -> dict[int, list[FugueNote]]:
    """
    Generate a complete Two-Part Invention from a plan.

    Simplified pipeline compared to fugue:
      1. Exposition: voice 0 states theme, voice 1 answers
      2. Episodes: sequence-based with tight consonance
      3. Middle entries: single voice states theme, other provides CP
      4. Coda: V→I cadence

    Returns dict mapping voice index → list of FugueNote.
    """
    config = config or _invention_config()

    if not plan.exposition:
        return {}

    # Adapt voice ranges for 2-voice clarity
    config = _adapt_invention_ranges(config, plan.subject)

    # Build harmonic skeleton
    tonic_pc = key_name_to_pc(plan.key_signature)
    is_minor = key_is_minor(plan.key_signature)
    total_dur = sum(s.estimated_duration for s in plan.sections)
    full_skeleton = generate_harmonic_skeleton(
        tonic_pc, is_minor, 0.0, total_dur,
        beats_per_chord=2.0,
        section_type="normal",
    )

    # --- 1. Exposition ---
    voices, countersubject = generate_exposition(
        plan.exposition, plan.subject, config, harmonic_skeleton=full_skeleton,
    )

    # --- 2-7. Remaining sections ---
    total_sections = len(plan.sections)
    for sec_idx, section in enumerate(plan.sections):
        progress = sec_idx / max(1, total_sections - 1) if total_sections > 1 else 0.0
        section_config = _invention_energy(config, progress, section.section_type)

        if section.section_type == SectionType.EXPOSITION:
            continue  # already done

        elif section.section_type == SectionType.EPISODE:
            _generate_invention_episode(
                plan, section, voices, countersubject, section_config, full_skeleton,
            )

        elif section.section_type == SectionType.MIDDLE_ENTRY:
            _generate_invention_entry(
                plan, section, voices, countersubject, section_config, progress,
                full_skeleton,
            )

        elif section.section_type == SectionType.CODA:
            coda_notes = generate_coda(
                plan.subject, plan.num_voices,
                section.start_offset, section.estimated_duration,
                voices, section_config,
            )
            for v, v_notes in coda_notes.items():
                voices.setdefault(v, []).extend(v_notes)

    # Post-process (same as fugue but benefits from tighter config)
    voices = _postprocess_leap_resolution(voices)
    voices = _postprocess_fix_parallels(voices)
    voices = _postprocess_fix_dissonances(voices, scale_pcs=config.scale_pcs)
    voices = _postprocess_climax_placement(voices)

    return voices


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _generate_invention_episode(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    countersubject: list[FugueNote],
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None = None,
) -> None:
    """Generate an episode with clear voice exchange."""
    # Parse modulation target
    mod_target = None
    if section.key_area and "→" in section.key_area:
        parts = section.key_area.split("→")
        if len(parts) == 2:
            from .harmony import key_name_to_pc, key_is_minor
            target_key = parts[1].strip()
            mod_target = (key_name_to_pc(target_key), key_is_minor(target_key))

    tonic_pc = key_name_to_pc(plan.key_signature)
    is_minor = key_is_minor(plan.key_signature)

    ep_skeleton = generate_harmonic_skeleton(
        tonic_pc, is_minor,
        section.start_offset, section.estimated_duration,
        beats_per_chord=2.0,
        section_type="episode",
        modulation_target=mod_target,
    )

    # Motif duration for stagger
    head_notes = [n for n in plan.subject.notes if not n.is_rest][:4]
    if head_notes:
        motif_dur = head_notes[-1].offset + head_notes[-1].duration - head_notes[0].offset
    else:
        motif_dur = 2.0
    motif_dur = max(1.0, motif_dur)

    episode_sources = ["subject_head", "subject_head_inverted"]
    seq_intervals = [-2, 2]

    for v in range(2):
        stagger = v * motif_dur * 0.5  # lighter stagger for 2 voices
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
            sequence_interval=seq_intervals[v],
            sequence_count=3,
            config=config,
            source=episode_sources[v],
            countersubject=countersubject,
            harmonic_skeleton=ep_skeleton,
        )
        voices.setdefault(v, []).extend(ep_notes)


def _generate_invention_entry(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    countersubject: list[FugueNote],
    config: GenerationConfig,
    progress: float,
    harmonic_skeleton: list[ChordLabel] | None = None,
) -> None:
    """
    Generate a middle entry: one voice states theme, the other provides CP.
    For inventions, only one voice enters at a time.
    """
    trans = _key_to_transposition(section.key_area, plan.key_signature)

    # Place theme in the entry voice
    for entry in section.entries:
        subj_notes = place_subject(
            plan.subject, entry.voice, entry.start_offset,
            transposition=trans, transform=entry.transform,
        )
        subj_notes = _adjust_placement_octave(
            subj_notes, entry.voice, voices, plan.num_voices, config,
        )
        voices.setdefault(entry.voice, []).extend(subj_notes)

    # Build section skeleton
    tonic_pc = key_name_to_pc(section.key_area) if section.key_area else key_name_to_pc(plan.key_signature)
    entry_skeleton = generate_harmonic_skeleton(
        tonic_pc, key_is_minor(section.key_area or plan.key_signature),
        section.start_offset, section.estimated_duration,
        beats_per_chord=2.0, section_type="normal",
    )

    # The other voice gets free counterpoint for the full section duration
    entry_voices = {e.voice for e in section.entries}
    sec_start = section.start_offset
    sec_end = section.start_offset + section.estimated_duration

    for v in range(2):
        if v in entry_voices:
            # Fill gaps before/after theme entry
            subj_notes_in_sec = [
                n for n in voices.get(v, [])
                if n.offset >= sec_start - 0.01 and n.offset < sec_end + 0.01
                and n.role in (EntryRole.SUBJECT, EntryRole.ANSWER)
            ]
            if subj_notes_in_sec:
                subj_start = min(n.offset for n in subj_notes_in_sec)
                subj_end = max(n.offset + n.duration for n in subj_notes_in_sec)

                if subj_start - sec_start > 0.5:
                    cp = generate_free_counterpoint(
                        voice=v, start_offset=sec_start,
                        duration=subj_start - sec_start,
                        existing_voices=voices, config=config,
                        progress=progress, harmonic_skeleton=entry_skeleton,
                    )
                    voices.setdefault(v, []).extend(cp)

                if sec_end - subj_end > 0.5:
                    cp = generate_free_counterpoint(
                        voice=v, start_offset=subj_end,
                        duration=sec_end - subj_end,
                        existing_voices=voices, config=config,
                        progress=progress, harmonic_skeleton=entry_skeleton,
                    )
                    voices.setdefault(v, []).extend(cp)
        else:
            # Non-entry voice: try countersubject first, else free CP
            if countersubject and section.entries:
                cp = _place_countersubject(
                    countersubject, voice=v,
                    start_offset=section.entries[0].start_offset,
                    existing_voices=voices, config=config,
                    transposition=trans,
                )
            else:
                cp = generate_free_counterpoint(
                    voice=v, start_offset=sec_start,
                    duration=section.estimated_duration,
                    existing_voices=voices, config=config,
                    progress=progress, harmonic_skeleton=entry_skeleton,
                )
            voices.setdefault(v, []).extend(cp)
