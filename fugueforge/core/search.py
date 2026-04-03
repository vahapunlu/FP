"""
FugueForge Search Engine

Monte Carlo Tree Search (MCTS)-inspired planning loop for fugue generation.
Instead of generating one candidate end-to-end, the system:

1. Generates the exposition (multiple candidates)
2. Evaluates each, keeps top-K
3. For each survivor, generates episode candidates
4. Evaluates, prunes
5. Continues section-by-section to the coda
6. Returns the best complete fugue

This dramatically increases quality by exploring the search space
at structural boundaries rather than hoping a single pass works.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional

from .representation import (
    EntryRole,
    FugueNote,
    FuguePlan,
    SectionType,
    Subject,
    TransformType,
)
from .generator import (
    GenerationConfig,
    generate_exposition,
    generate_episode,
    generate_free_counterpoint,
    generate_coda,
    place_subject,
    voices_to_score,
    _adapt_voice_ranges,
    _key_to_transposition,
    _adjust_placement_octave,
    _place_countersubject,
    _postprocess_leap_resolution,
    _postprocess_climax_placement,
    _postprocess_fix_parallels,
    _postprocess_fix_dissonances,
)
from .judge import FugueJudge, JudgeScore
from .rules import check_counterpoint


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Parameters for the MCTS-style search."""
    candidates_per_section: int = 6     # generate N candidates per section
    survivors_per_section: int = 3      # keep top K after evaluation
    total_candidates: int = 20          # total attempts for full fugue
    final_top_k: int = 5               # keep K best full fugues for final ranking
    gen_config: GenerationConfig = field(default_factory=GenerationConfig)


# ---------------------------------------------------------------------------
# Section-level candidate
# ---------------------------------------------------------------------------

@dataclass
class FugueCandidate:
    """A partial or complete fugue being built incrementally."""
    voices: dict[int, list[FugueNote]]
    countersubject: list[FugueNote] = field(default_factory=list)
    sections_completed: int = 0
    running_score: float = 0.0


# ---------------------------------------------------------------------------
# MCTS-style search
# ---------------------------------------------------------------------------

class FugueSearch:
    """
    Section-by-section beam search for fugue generation.
    Generates multiple candidates at each structural boundary,
    evaluates them, and keeps only the best for the next section.
    """

    def __init__(
        self,
        plan: FuguePlan,
        search_config: Optional[SearchConfig] = None,
        judge: Optional[FugueJudge] = None,
    ):
        self.plan = plan
        self.config = search_config or SearchConfig()
        self.judge = judge or FugueJudge()
        self.gen_config = _adapt_voice_ranges(
            self.config.gen_config, plan.subject, plan.num_voices,
        )

    def search(self) -> tuple[dict[int, list[FugueNote]], JudgeScore]:
        """
        Run the section-by-section search and return the best fugue.
        """
        if not self.plan.exposition:
            return {}, JudgeScore()

        # Phase 1: Generate exposition candidates
        exposition_candidates = self._generate_exposition_candidates()

        # Phase 2: Extend section by section
        current_candidates = exposition_candidates
        sections = [s for s in self.plan.sections if s.section_type != SectionType.EXPOSITION]

        for sec_idx, section in enumerate(sections):
            progress = sec_idx / max(1, len(sections) - 1) if len(sections) > 1 else 0.0
            next_candidates: list[FugueCandidate] = []

            for cand in current_candidates:
                for _ in range(self.config.candidates_per_section):
                    new_cand = self._extend_candidate(cand, section, progress=progress)
                    next_candidates.append(new_cand)

            # Evaluate and keep top K
            scored = []
            for nc in next_candidates:
                js = self.judge.evaluate(nc.voices, self.plan.subject, self.plan)
                nc.running_score = js.total
                scored.append((js.total, nc))

            scored.sort(reverse=True, key=lambda x: x[0])
            current_candidates = [
                nc for _, nc in scored[:self.config.survivors_per_section]
            ]

        # Phase 3: Post-process and final evaluation of all surviving candidates
        best_voices = current_candidates[0].voices if current_candidates else {}
        best_js = JudgeScore()
        best_total = -1.0

        for cand in current_candidates:
            cand.voices = _postprocess_leap_resolution(cand.voices)
            cand.voices = _postprocess_fix_parallels(cand.voices)
            cand.voices = _postprocess_fix_dissonances(cand.voices)
            cand.voices = _postprocess_climax_placement(cand.voices)
            js = self.judge.evaluate(cand.voices, self.plan.subject, self.plan)
            if js.total > best_total:
                best_total = js.total
                best_voices = cand.voices
                best_js = js

        return best_voices, best_js

    def _generate_exposition_candidates(self) -> list[FugueCandidate]:
        """Generate multiple exposition variants."""
        candidates: list[FugueCandidate] = []

        for _ in range(self.config.candidates_per_section):
            voices, cs = generate_exposition(
                self.plan.exposition, self.plan.subject, self.gen_config,
            )
            candidates.append(FugueCandidate(
                voices=voices,
                countersubject=cs,
                sections_completed=1,
            ))

        # Evaluate and keep top K
        scored = []
        for c in candidates:
            js = self.judge.evaluate(c.voices, self.plan.subject, self.plan)
            c.running_score = js.total
            scored.append((js.total, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:self.config.survivors_per_section]]

    def _extend_candidate(
        self,
        candidate: FugueCandidate,
        section,
        progress: float = 0.0,
    ) -> FugueCandidate:
        """Extend a candidate with one more section."""
        voices = {v: list(notes) for v, notes in candidate.voices.items()}

        if section.section_type == SectionType.EPISODE:
            motif_notes = [
                n for n in self.plan.subject.notes if not n.is_rest
            ][:4]
            motif_dur = sum(n.duration for n in motif_notes) if motif_notes else 2.0

            # Vary episode source material across voices (Burrows thesis)
            episode_sources = ["subject_head", "subject_head_inverted", "countersubject", "subject_tail"]
            seq_intervals = [-2, 2, -3]

            for v in range(self.plan.num_voices):
                stagger = v * motif_dur * len(motif_notes)
                ep_start = section.start_offset + stagger
                ep_dur = section.estimated_duration - stagger
                if ep_dur <= 0:
                    continue

                ep_notes = generate_episode(
                    subject=self.plan.subject,
                    voice=v,
                    start_offset=ep_start,
                    duration=ep_dur,
                    existing_voices=voices,
                    sequence_interval=seq_intervals[v % len(seq_intervals)],
                    sequence_count=3,
                    config=self.gen_config,
                    source=episode_sources[v % len(episode_sources)],
                    countersubject=candidate.countersubject,
                )
                voices.setdefault(v, []).extend(ep_notes)

        elif section.section_type in (SectionType.MIDDLE_ENTRY, SectionType.STRETTO):
            # Use inversion for variety (per Burrows/BWV 547: rectus + inversus)
            trans = _key_to_transposition(
                section.key_area, self.plan.key_signature,
            )
            for entry_i, entry in enumerate(section.entries):
                use_inversion = (
                    section.section_type == SectionType.STRETTO and entry_i % 2 == 1
                ) or (
                    section.section_type == SectionType.MIDDLE_ENTRY
                    and entry_i > 0 and entry_i % 2 == 1
                )
                xform = TransformType.INVERSION if use_inversion else TransformType.NONE
                subj_notes = place_subject(
                    self.plan.subject,
                    entry.voice,
                    entry.start_offset,
                    transposition=trans,
                    transform=xform,
                )
                subj_notes = _adjust_placement_octave(
                    subj_notes, entry.voice, voices,
                    self.plan.num_voices, self.gen_config,
                )
                voices.setdefault(entry.voice, []).extend(subj_notes)

            # Fill non-subject voices: first gets CS, rest get free CP
            entry_voices = {e.voice for e in section.entries}
            free_voices = [v for v in range(self.plan.num_voices) if v not in entry_voices]
            cs_placed = False

            for v in free_voices:
                if not cs_placed and candidate.countersubject and section.entries:
                    trans = _key_to_transposition(
                        section.key_area, self.plan.key_signature,
                    )
                    cp = _place_countersubject(
                        candidate.countersubject,
                        voice=v,
                        start_offset=section.entries[0].start_offset,
                        existing_voices=voices,
                        config=self.gen_config,
                        transposition=trans,
                    )
                    cs_placed = True
                else:
                    cp = generate_free_counterpoint(
                        voice=v,
                        start_offset=section.start_offset,
                        duration=section.estimated_duration,
                        existing_voices=voices,
                        config=self.gen_config,
                        progress=progress,
                    )
                voices.setdefault(v, []).extend(cp)

        elif section.section_type == SectionType.CODA:
            coda_notes = generate_coda(
                self.plan.subject,
                self.plan.num_voices,
                section.start_offset,
                section.estimated_duration,
                voices,
                self.gen_config,
            )
            for v, v_notes in coda_notes.items():
                voices.setdefault(v, []).extend(v_notes)

        return FugueCandidate(
            voices=voices,
            countersubject=candidate.countersubject,
            sections_completed=candidate.sections_completed + 1,
            running_score=candidate.running_score,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def search_best_fugue(
    plan: FuguePlan,
    search_config: Optional[SearchConfig] = None,
) -> tuple[dict[int, list[FugueNote]], JudgeScore]:
    """
    Generate the best fugue using section-by-section beam search.
    Returns (voices, judge_score).
    """
    searcher = FugueSearch(plan, search_config)
    return searcher.search()
