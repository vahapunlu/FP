"""
FugueForge Generator — Orchestrator

Coordinates the full fugue generation pipeline:
  Exposition → Episodes → Middle Entries → Stretto → Coda → Post-processing

Sub-modules handle specific concerns:
  - candidates:   pitch generation & constraint checking
  - counterpoint:  free counterpoint & countersubject placement
  - placement:    subject/answer placement & octave adjustment
  - episodes:     sequence-based episode generation
  - exposition:   exposition generation with CS capture
  - coda:         cadential coda generation
  - postprocess:  post-processing passes
  - voice_utils:  key/pitch helpers, voice range adaptation, output
"""

from __future__ import annotations

from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    FuguePlan,
    SectionType,
    TransformType,
)

# --- Sub-module re-exports for backward compatibility ---
# External modules (search.py, cli.py, pipeline.py, tests) import from
# generator.py — these re-exports keep those imports working.

from .candidates import (  # noqa: F401
    GenerationConfig,
    VoiceState,
    _candidate_pitches,
    _check_hidden_fifth_octave,
    _check_parallel_motion,
    _check_voice_crossing,
    _score_candidate,
)
from .voice_utils import (  # noqa: F401
    _adapt_voice_ranges,
    _key_to_transposition,
    key_to_scale_pcs,
    voices_to_score,
)
from .placement import (  # noqa: F401
    _adjust_placement_octave,
    _get_transposition,
    place_answer,
    place_subject,
)
from .counterpoint import (  # noqa: F401
    _place_countersubject,
    generate_free_counterpoint,
)
from .episodes import generate_episode  # noqa: F401
from .exposition import generate_exposition  # noqa: F401
from .coda import generate_coda  # noqa: F401
from .harmony import (  # noqa: F401
    generate_harmonic_skeleton,
    key_name_to_pc,
    key_is_minor,
    get_chord_at,
    ChordLabel,
)
from .postprocess import (  # noqa: F401
    _postprocess_climax_placement,
    _postprocess_fix_dissonances,
    _postprocess_fix_parallels,
    _postprocess_leap_resolution,
)


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

    # Build harmonic skeleton for the entire piece
    tonic_pc = key_name_to_pc(plan.key_signature)
    is_minor = key_is_minor(plan.key_signature)
    total_dur = sum(s.estimated_duration for s in plan.sections)
    full_skeleton = generate_harmonic_skeleton(
        tonic_pc, is_minor, 0.0, total_dur,
        beats_per_chord=2.0,
        section_type="normal",
    )

    # Generate exposition (also captures countersubject)
    voices, countersubject = generate_exposition(
        plan.exposition, plan.subject, config, harmonic_skeleton=full_skeleton,
    )

    # Process remaining sections
    total_sections = len(plan.sections)
    for sec_idx, section in enumerate(plan.sections):
        progress = (
            sec_idx / max(1, total_sections - 1) if total_sections > 1 else 0.0
        )
        if section.section_type == SectionType.EXPOSITION:
            continue  # already done

        elif section.section_type == SectionType.EPISODE:
            # Episode: use episode-specific harmonic progression
            ep_skeleton = generate_harmonic_skeleton(
                tonic_pc, is_minor,
                section.start_offset,
                section.estimated_duration,
                beats_per_chord=2.0,
                section_type="episode",
                progression_idx=sec_idx,
            )
            _generate_episode_section(
                plan, section, voices, countersubject, config,
                harmonic_skeleton=ep_skeleton,
            )

        elif section.section_type in (SectionType.MIDDLE_ENTRY, SectionType.STRETTO):
            # Entry sections: use normal progression in the section's key
            entry_tonic = key_name_to_pc(section.key_area) if section.key_area else tonic_pc
            entry_skeleton = generate_harmonic_skeleton(
                entry_tonic, is_minor,
                section.start_offset,
                section.estimated_duration,
                beats_per_chord=2.0,
                section_type="normal",
                progression_idx=sec_idx,
            )
            _generate_entry_section(
                plan, section, voices, countersubject, config, progress,
                harmonic_skeleton=entry_skeleton,
            )

        elif section.section_type == SectionType.CODA:
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

    # Post-process: fix common counterpoint issues
    voices = _postprocess_leap_resolution(voices)
    voices = _postprocess_fix_parallels(voices)
    voices = _postprocess_fix_dissonances(voices, scale_pcs=config.scale_pcs)
    voices = _postprocess_climax_placement(voices)

    return voices


# ---------------------------------------------------------------------------
# Section-level helpers
# ---------------------------------------------------------------------------

def _generate_episode_section(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    countersubject: list[FugueNote],
    config: GenerationConfig,
    harmonic_skeleton: list[ChordLabel] | None = None,
) -> None:
    """Generate episodes voice by voice, staggered for imitative texture."""
    motif_dur = sum(
        n.duration for n in plan.subject.notes if not n.is_rest
    ) / max(1, len([n for n in plan.subject.notes if not n.is_rest][:4]))

    episode_sources = [
        "subject_head", "subject_head_inverted",
        "countersubject", "subject_tail",
    ]
    seq_intervals = [-2, 2, -3]

    for v in range(plan.num_voices):
        stagger = v * motif_dur * min(
            4, len([n for n in plan.subject.notes if not n.is_rest][:4])
        )
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
            harmonic_skeleton=harmonic_skeleton,
        )
        voices.setdefault(v, []).extend(ep_notes)


def _generate_entry_section(
    plan: FuguePlan,
    section,
    voices: dict[int, list[FugueNote]],
    countersubject: list[FugueNote],
    config: GenerationConfig,
    progress: float,
    harmonic_skeleton: list[ChordLabel] | None = None,
) -> None:
    """Generate middle entry or stretto section with subject placements."""
    trans = _key_to_transposition(section.key_area, plan.key_signature)

    for entry_i, entry in enumerate(section.entries):
        # In stretto: alternate rectus/inversus for contrapuntal intensity
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
    entry_voices = {e.voice for e in section.entries}
    free_voices = [v for v in range(plan.num_voices) if v not in entry_voices]
    cs_placed = False

    for v in free_voices:
        if not cs_placed and countersubject and section.entries:
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
                harmonic_skeleton=harmonic_skeleton,
            )
        voices.setdefault(v, []).extend(cp)
