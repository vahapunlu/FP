"""
FugueForge Corpus — Gold standard fugue annotations and corpus tools.
"""

from .gold import GOLD_SUBJECTS, GoldFugue, get_gold_fugue, list_gold_fugues
from .loader import load_fugue_from_file, CorpusEntry
from .pipeline import evaluate_corpus, CorpusReport
