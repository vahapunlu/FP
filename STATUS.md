# FugueForge — Project Status & Roadmap

## What Is FugueForge?

A neurosymbolic fugue composition engine that generates multi-voice Baroque fugues
from a given musical subject. Input: a melody. Output: a structurally valid,
stylistically convincing fugue in the manner of J.S. Bach.

---

## Current Architecture

```
Subject (melody)
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Analyzer │────▶│ Planner  │────▶│Generator │
│          │     │          │     │          │
│ interval │     │ expo/epi │     │ constrai-│
│ profiles,│     │ middle/  │     │ nt-aware │
│ tonal    │     │ stretto/ │     │ note-by- │
│ answer   │     │ coda     │     │ note     │
└──────────┘     └──────────┘     └──────────┘
                                       │
                                       ▼
                                 ┌──────────┐
                                 │  Search  │
                                 │          │
                                 │ MCTS beam│
                                 │ section  │
                                 │ by sect. │
                                 └──────────┘
                                       │
                                       ▼
                                 ┌──────────┐
                                 │  Judge   │
                                 │          │
                                 │ Theory   │ 30%
                                 │ Structure│ 30%
                                 │ Style    │ 20%
                                 │ Aesthetic│ 20%
                                 └──────────┘
```

### Module Inventory (19 files, 5,912 lines)

| Module | Purpose | Lines |
|--------|---------|-------|
| `core/representation.py` | Data types: FugueNote, Subject, FuguePlan, enums | ~150 |
| `core/rules.py` | Counterpoint rule checker (parallels, crossing, dissonance, spacing, hidden intervals) | ~350 |
| `core/analyzer.py` | Subject analysis, tonal answer, occurrence finder (real + tonal + inverted), section detection, cadence detection | ~450 |
| `core/planner.py` | Macro structure planning: exposition → episodes → middle entries → stretto → coda | ~300 |
| `core/generator.py` | Constraint-aware note-by-note generation, episode/coda generators, post-processors | ~1500 |
| `core/search.py` | MCTS-inspired beam search: section-by-section candidate generation and pruning | ~300 |
| `core/judge.py` | 4-axis scorer: theory, structure, style, aesthetic | ~400 |
| `corpus/gold.py` | 13 Bach WTC fugue subjects with full annotations | ~400 |
| `corpus/pipeline.py` | Evaluation pipeline: single-fugue and full-corpus benchmarking | ~200 |
| `corpus/loader.py` | MIDI/MusicXML loader utilities | ~100 |
| `cli.py` | Command-line interface | ~100 |
| `tests/test_core.py` | 27 unit tests for core modules | ~300 |
| `tests/test_corpus.py` | 9 tests for corpus and pipeline | ~150 |

---

## Current Benchmark (v0.1)

```
BWV    Voices   Search   Theory   Struct   Style   Aesth
────────────────────────────────────────────────────────
846    4         93.8     84.9    100.0    91.7   100.0
847    3         94.9     93.4    100.0    84.3   100.0
848    3         95.3     96.5    100.0    81.8   100.0
849    5         92.8     85.3    100.0    91.3    95.0
850    4         93.8     87.4    100.0    87.8   100.0
851    3         97.2     95.6    100.0    92.5   100.0
853    3         97.8     96.2    100.0    96.6    98.0
855    2         95.2     91.0    100.0    90.0   100.0
858    3         97.3     95.2    100.0    91.9   100.0
860    3         97.4     97.2    100.0    91.0   100.0
861    4         94.6     86.7    100.0    92.8   100.0
865    4         92.8     86.4    100.0    84.5   100.0
869    4         96.1     92.6    100.0    93.4   100.0
────────────────────────────────────────────────────────
AVG              95.2     91.3    100.0    90.0    99.5
```

**36/36 tests passing.**

---

## Honest Assessment — Known Weaknesses

### Methodological

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| M1 | **Self-evaluation bias**: Judge is written by us, then we optimize for it. Score 95 means "95% compliant with our own rules" — not "good music" | High | Not addressed |
| M2 | **No ablation study**: 9+ fixes applied; individual contribution unknown | Medium | Not addressed |
| M3 | **No baseline comparison**: What does a random generator score? A rule-only (no search) system? | Medium | Not addressed |
| M4 | **Tiny corpus**: 13 subjects — possible overfitting to these specific subjects | Medium | Not addressed |
| M5 | **Stochastic variance**: Each run gives different scores (±1-2 pts). Need 30-run mean ± std | Low | Not addressed |

### Musical

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| U1 | **Never listened**: No MIDI output has been aurally evaluated | Critical | Not addressed |
| U2 | **Flat harmonic rhythm**: Every note treated equally; no structural vs ornamental distinction | High | Not addressed |
| U3 | **No phrasing/breath**: Generator produces continuous stream; no phrase boundaries | High | Not addressed |
| U4 | **Surface-level modulation**: Key changes are transpositions, not real harmonic journeys | High | Not addressed |
| U5 | **Mechanical episodes**: Sequences are literal copy-paste transpositions; no variation | Medium | Not addressed |
| U6 | **No dramaturgy**: All sections at same energy level; no tension arc | Medium | Not addressed |

### Technical Ceiling

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| T1 | **Theory in 4/5-voice fugues**: 80-87 range due to combinatorial explosion of voice pairs | High | Partially addressed |
| T2 | **Style variance**: 81-97 range; some BWVs still have high unresolved leap ratios | Medium | Partially addressed |
| T3 | **Search barely beats gen**: +0.2-0.5 only; search not exploiting its potential | Low | Not addressed |

---

## Roadmap

### Phase 1: Listen & Validate (Next Session)
> Goal: Ground truth check — does this actually sound like music?

- [ ] Generate MIDI for all 13 BWV subjects
- [ ] Listen to each one, take notes on what sounds wrong
- [ ] Identify gaps between judge scores and aural quality
- [ ] Create a "listening report" mapping subjective impressions to code
- [ ] Establish baselines: random generator score, no-search score

### Phase 2: Musical Intelligence (1-2 Sessions)
> Goal: Address the musical weaknesses that listening reveals

- [ ] **Phrasing**: Add phrase boundary detection; insert micro-rests or longer notes at phrase ends
- [ ] **Harmonic rhythm**: Distinguish structural beats (harmonic changes) from passing/neighbor tones
- [ ] **Episode variation**: Add subtle rhythmic/intervallic variation to each sequence repetition
- [ ] **Modulation depth**: Real pivot chord modulations instead of raw transposition
- [ ] **Dramaturgy**: Energy curve — register expansion, rhythmic density, texture changes across sections
- [ ] **Ablation study**: Isolate contribution of each major fix
- [ ] Statistical benchmarking: 30 runs × 13 BWVs, report mean ± std

### Phase 3: ML Integration (2-3 Sessions, RunPod)
> Goal: Break through the rule-based ceiling with learned Bach style

**Layer 1 — Neural Style Scorer**
- [ ] Collect Bach fugue MIDI corpus (~50-100 pieces, public domain)
- [ ] Parse into windowed training examples (4-8 beat segments)
- [ ] Train small transformer/CNN: "How Bach-like is this window?" → 0-1
- [ ] Add as 5th judge axis: `neural_style` weight ~0.15
- [ ] Re-benchmark with neural scorer active

**Layer 2 — Learned Candidate Scoring**
- [ ] Train pitch prediction model: context (prev notes + other voices) → next pitch distribution
- [ ] Integrate as bonus in `_score_candidate()`: `score += neural_prior[pitch] * weight`
- [ ] Generator still constraint-aware but now "knows" what Bach would prefer

**Layer 3 — Reward-Guided Search**
- [ ] Use neural scorer for faster section-level evaluation in MCTS
- [ ] Explore more candidates per section with learned guidance
- [ ] Adaptive beam width based on section difficulty

### Phase 4: Evaluation & Publication (1-2 Sessions)
> Goal: Scientific validation and public release

- [ ] **Blind listening test**: 5-10 musicians rate FugueForge vs Bach vs random (Turing test style)
- [ ] **Expanded corpus test**: 20 unseen subjects to verify generalization
- [ ] **Comparison with existing systems**: Literature review + benchmark comparison
- [ ] **Paper draft**: Architecture, results, ablation, listening study
- [ ] Target venues: ISMIR, AIMC, NeurIPS Creative AI workshop
- [ ] **pip package**: `pip install fugueforge` ready for community use
- [ ] GitHub README, documentation, examples

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13.3 |
| Music library | music21 v9.9.1 |
| Numeric | numpy |
| MIDI output | midiutil |
| PDF parsing | pymupdf (for thesis reading) |
| Testing | pytest |
| ML (planned) | PyTorch, RunPod GPU |
| Version control | Git → github.com/vahapunlu/FP |

---

## Score History

| Milestone | Search Avg | Key Change |
|-----------|-----------|------------|
| MVP pipeline | 88.0 | Initial 8-module architecture |
| CS capture + rules | 90.2 | Countersubject system, constraint improvements |
| Burrows thesis | 90.9 | Subject inversion, thematic episodes, thematic coda |
| Theory/Style attack | **95.2** | Leap resolution, PAC fix, climax placement, post-processors |

---

## File Locations

```
D:\PYT\FP\
├── .gitignore
├── pyproject.toml
├── fugueforge/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── representation.py
│   │   ├── rules.py
│   │   ├── analyzer.py
│   │   ├── planner.py
│   │   ├── generator.py    ← largest module (~1500 lines)
│   │   ├── judge.py
│   │   └── search.py
│   └── corpus/
│       ├── __init__.py
│       ├── gold.py
│       ├── loader.py
│       └── pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py        ← 27 tests
│   └── test_corpus.py      ← 9 tests
└── output/                  ← generated MIDI files (gitignored)
```
