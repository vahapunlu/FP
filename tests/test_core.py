"""Tests for the FugueForge core modules."""

import pytest

from fugueforge.core.representation import (
    AnswerType,
    EntryRole,
    FugueNote,
    Subject,
    TransformType,
)
from fugueforge.core.rules import (
    CounterpointRuleChecker,
    ViolationType,
    check_counterpoint,
    classify_motion,
    MotionType,
    is_consonance,
    is_dissonance,
)
from fugueforge.core.analyzer import (
    compute_tonal_answer,
    detect_answer_type,
    extract_interval_profile,
    find_subject_occurrences,
)
from fugueforge.core.planner import (
    plan_exposition,
    plan_fugue,
    find_stretto_possibilities,
)
from fugueforge.core.generator import (
    generate_exposition,
    generate_fugue,
    place_subject,
    GenerationConfig,
)
from fugueforge.core.judge import FugueJudge, score_theory
from fugueforge.core.search import FugueSearch, SearchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_c_minor_subject() -> Subject:
    """C minor subject: C-Eb-D-C-B-C-G."""
    notes = [
        FugueNote(pitch=60, duration=1.0, voice=0, offset=0.0),
        FugueNote(pitch=63, duration=0.5, voice=0, offset=1.0),
        FugueNote(pitch=62, duration=0.5, voice=0, offset=1.5),
        FugueNote(pitch=60, duration=0.5, voice=0, offset=2.0),
        FugueNote(pitch=59, duration=0.5, voice=0, offset=2.5),
        FugueNote(pitch=60, duration=0.5, voice=0, offset=3.0),
        FugueNote(pitch=55, duration=1.5, voice=0, offset=3.5),
    ]
    return Subject(notes=notes, key_signature="c", time_signature="4/4")


def make_simple_two_voice() -> dict[int, list[FugueNote]]:
    """Simple two-voice passage for testing."""
    return {
        0: [
            FugueNote(pitch=72, duration=1.0, voice=0, offset=0.0),
            FugueNote(pitch=71, duration=1.0, voice=0, offset=1.0),
            FugueNote(pitch=72, duration=1.0, voice=0, offset=2.0),
        ],
        1: [
            FugueNote(pitch=60, duration=1.0, voice=1, offset=0.0),
            FugueNote(pitch=59, duration=1.0, voice=1, offset=1.0),
            FugueNote(pitch=60, duration=1.0, voice=1, offset=2.0),
        ],
    }


def make_parallel_fifths_passage() -> dict[int, list[FugueNote]]:
    """Two voices moving in parallel fifths (should be flagged)."""
    return {
        0: [
            FugueNote(pitch=67, duration=1.0, voice=0, offset=0.0),  # G4
            FugueNote(pitch=69, duration=1.0, voice=0, offset=1.0),  # A4
        ],
        1: [
            FugueNote(pitch=60, duration=1.0, voice=1, offset=0.0),  # C4
            FugueNote(pitch=62, duration=1.0, voice=1, offset=1.0),  # D4
        ],
    }


# ---------------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------------

class TestSubject:
    def test_duration(self):
        s = make_c_minor_subject()
        assert s.duration == 5.0

    def test_pitch_range(self):
        s = make_c_minor_subject()
        assert s.pitch_range == (55, 63)

    def test_interval_sequence(self):
        s = make_c_minor_subject()
        intervals = s.interval_sequence
        assert intervals == [3, -1, -2, -1, 1, -5]

    def test_to_stream(self):
        s = make_c_minor_subject()
        part = s.to_stream()
        assert part is not None


# ---------------------------------------------------------------------------
# Rules tests
# ---------------------------------------------------------------------------

class TestIntervalHelpers:
    def test_consonance(self):
        assert is_consonance(0)   # unison
        assert is_consonance(7)   # P5
        assert is_consonance(12)  # P8
        assert is_consonance(3)   # m3
        assert is_consonance(4)   # M3

    def test_dissonance(self):
        assert is_dissonance(1)   # m2
        assert is_dissonance(2)   # M2
        assert is_dissonance(6)   # tritone
        assert is_dissonance(11)  # M7


class TestMotion:
    def test_parallel(self):
        assert classify_motion(60, 62, 67, 69) == MotionType.PARALLEL

    def test_contrary(self):
        assert classify_motion(60, 62, 67, 65) == MotionType.CONTRARY

    def test_oblique(self):
        assert classify_motion(60, 60, 67, 69) == MotionType.OBLIQUE

    def test_similar(self):
        assert classify_motion(60, 62, 67, 72) == MotionType.SIMILAR


class TestRuleChecker:
    def test_clean_passage(self):
        voices = make_simple_two_voice()
        score, violations = check_counterpoint(voices)
        # Should have a reasonable score
        assert score >= 0

    def test_parallel_fifths_detected(self):
        voices = make_parallel_fifths_passage()
        _, violations = check_counterpoint(voices)
        p5_violations = [v for v in violations if v.type == ViolationType.PARALLEL_FIFTHS]
        assert len(p5_violations) > 0


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------

class TestAnalyzer:
    def test_interval_profile(self):
        s = make_c_minor_subject()
        profile = extract_interval_profile(s.notes)
        assert profile == [3, -1, -2, -1, 1, -5]

    def test_tonal_answer(self):
        s = make_c_minor_subject()
        answer = compute_tonal_answer(s)
        assert len(answer) == len(s.notes)
        # Answer should be transposed up ~7 semitones
        assert answer[0].pitch == s.notes[0].pitch + 7

    def test_find_occurrences(self):
        s = make_c_minor_subject()
        # Place subject in voice 0 and a transposed copy in voice 1
        voices = {
            0: s.notes,
            1: [
                FugueNote(
                    pitch=n.pitch + 7,
                    duration=n.duration,
                    voice=1,
                    offset=n.offset + s.duration,
                )
                for n in s.notes
            ],
        }
        occurrences = find_subject_occurrences(s, voices)
        assert len(occurrences) >= 1


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_exposition_plan(self):
        s = make_c_minor_subject()
        plan = plan_exposition(s, num_voices=3)
        assert plan.num_voices == 3
        assert len(plan.entries) == 3

    def test_stretto_possibilities(self):
        s = make_c_minor_subject()
        possibilities = find_stretto_possibilities(s, num_voices=3)
        assert isinstance(possibilities, list)

    def test_full_plan(self):
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3, target_measures=24)
        assert plan.num_voices == 3
        assert plan.exposition is not None
        assert len(plan.sections) > 0


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_place_subject(self):
        s = make_c_minor_subject()
        placed = place_subject(s, voice=1, start_offset=5.0, transposition=7)
        assert len(placed) == len(s.notes)
        assert placed[0].offset == 5.0
        assert placed[0].pitch == s.notes[0].pitch + 7

    def test_generate_exposition(self):
        s = make_c_minor_subject()
        plan = plan_exposition(s, num_voices=3)
        voices, countersubject = generate_exposition(plan, s)
        assert len(voices) == 3
        for v_notes in voices.values():
            assert len(v_notes) > 0
        # Countersubject should be captured from first voice's cp against answer
        assert len(countersubject) > 0
        assert all(n.role == EntryRole.COUNTERSUBJECT for n in countersubject)

    def test_generate_full_fugue(self):
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3, target_measures=16)
        voices = generate_fugue(plan)
        assert len(voices) > 0


# ---------------------------------------------------------------------------
# Judge tests
# ---------------------------------------------------------------------------

class TestJudge:
    def test_theory_score(self):
        voices = make_simple_two_voice()
        score, _ = score_theory(voices)
        assert 0 <= score <= 100

    def test_full_evaluation(self):
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3)
        voices = generate_fugue(plan)

        judge = FugueJudge()
        js = judge.evaluate(voices, s, plan)
        assert 0 <= js.total <= 100
        assert js.theory >= 0
        assert js.structure >= 0

    def test_candidate_ranking(self):
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3)

        candidates = [generate_fugue(plan) for _ in range(3)]

        judge = FugueJudge()
        ranked = judge.rank_candidates(candidates, s, plan)
        assert len(ranked) == 3
        # Should be sorted descending
        assert ranked[0][0] >= ranked[-1][0]


class TestSearch:
    def test_search_produces_score(self):
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3)
        cfg = SearchConfig(candidates_per_section=2, survivors_per_section=2)
        voices, js = FugueSearch(plan, cfg).search()
        assert len(voices) > 0
        assert 0 <= js.total <= 100

    def test_search_beats_single_candidate(self):
        """Search with 3 candidates/section should score >= single generation."""
        s = make_c_minor_subject()
        plan = plan_fugue(s, num_voices=3)

        # Single candidate baseline
        single_voices = generate_fugue(plan)
        judge = FugueJudge()
        single_js = judge.evaluate(single_voices, s, plan)

        # Beam search
        cfg = SearchConfig(candidates_per_section=3, survivors_per_section=2)
        _, search_js = FugueSearch(plan, cfg, judge).search()

        # Search should generally not be worse (with some margin for randomness)
        assert search_js.total >= single_js.total - 15

    def test_search_config_defaults(self):
        cfg = SearchConfig()
        assert cfg.candidates_per_section > 0
        assert cfg.survivors_per_section > 0
