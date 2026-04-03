"""Tests for the FugueForge corpus module."""

import pytest

from fugueforge.corpus.gold import (
    GOLD_SUBJECTS,
    get_gold_fugue,
    list_gold_fugues,
    get_subjects_by_voices,
    GoldFugue,
)
from fugueforge.corpus.pipeline import evaluate_single, CorpusReport
from fugueforge.core.representation import AnswerType
from fugueforge.core.search import SearchConfig


class TestGold:
    def test_gold_subjects_exist(self):
        assert len(GOLD_SUBJECTS) >= 10

    def test_get_gold_fugue(self):
        g = get_gold_fugue(847)
        assert g is not None
        assert g.bwv == 847
        assert g.num_voices == 3
        assert g.answer_type == AnswerType.TONAL

    def test_to_subject(self):
        g = get_gold_fugue(847)
        subject = g.to_subject()
        assert len(subject.notes) > 0
        assert subject.duration > 0
        assert subject.key_signature == "c"

    def test_list_gold_sorted(self):
        fugues = list_gold_fugues()
        bwvs = [f.bwv for f in fugues]
        assert bwvs == sorted(bwvs)

    def test_voices_filter(self):
        three_voice = get_subjects_by_voices(3)
        assert all(g.num_voices == 3 for g in three_voice)
        assert len(three_voice) >= 4

    def test_all_subjects_valid(self):
        """Every gold subject should produce a valid Subject object."""
        for bwv, gold in GOLD_SUBJECTS.items():
            subject = gold.to_subject()
            assert len(subject.notes) > 0, f"BWV {bwv} has empty subject"
            assert subject.duration > 0, f"BWV {bwv} has zero duration"
            pitches = [n.pitch for n in subject.notes if not n.is_rest]
            assert len(pitches) >= 3, f"BWV {bwv} has too few pitches"


class TestPipeline:
    def test_evaluate_single_3voice(self):
        """Evaluate a 3-voice fugue from gold corpus."""
        gold = get_gold_fugue(847)  # C minor, 3 voices
        cfg = SearchConfig(candidates_per_section=2, survivors_per_section=2)
        result = evaluate_single(gold, num_gen_candidates=2, search_config=cfg)
        assert result.bwv == 847
        assert 0 <= result.gen_total <= 100
        assert 0 <= result.search_total <= 100

    def test_evaluate_single_4voice(self):
        """Evaluate a 4-voice fugue."""
        gold = get_gold_fugue(846)  # C major, 4 voices
        cfg = SearchConfig(candidates_per_section=2, survivors_per_section=2)
        result = evaluate_single(gold, num_gen_candidates=2, search_config=cfg)
        assert result.bwv == 846
        assert result.num_voices == 4
        assert result.search_total > 0

    def test_corpus_report(self):
        report = CorpusReport()
        assert report.num_fugues == 0
        assert report.avg_gen_total == 0.0
