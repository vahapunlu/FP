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

## Current Benchmark (v0.2 — post-refactor)

```
BWV    Voices  Random  Single  Search  Theory  Struct  Style  Aesth
─────────────────────────────────────────────────────────────────────
846    4        41.5    93.8    93.9    84.4   100.0   92.8  100.0
847    3        52.1    94.5    94.4    91.4   100.0   84.7  100.0
848    3        53.0    94.9    95.0    95.2   100.0   82.2  100.0
849    5        35.6    91.7    91.2    81.8   100.0   88.4   95.0
850    4        43.8    93.0    93.7    86.7   100.0   88.7  100.0
851    3        49.6    96.5    97.0    96.0   100.0   91.2  100.0
853    3        52.1    95.9    96.7    95.5   100.0   92.1   98.0
855    2        63.0    95.3    95.8    90.6   100.0   92.9  100.0
858    3        56.2    97.4    96.4    94.9   100.0   89.7  100.0
860    3        49.3    96.8    97.0    95.8   100.0   91.5  100.0
861    4        43.6    94.5    94.9    89.2   100.0   90.7  100.0
865    4        42.5    92.1    93.4    89.3   100.0   83.3  100.0
869    4        42.8    95.9    96.4    92.3   100.0   93.8  100.0
─────────────────────────────────────────────────────────────────────
AVG             48.1    94.8    95.1    91.0   100.0   89.4   99.5
```

**Baseline comparison**: Random=48.1, Single-pass=94.8, Search=95.1 (Δ +47.0 vs random, +0.3 vs single)

**36/36 tests passing.** MIDI files generated for all 13 BWVs.

---

## Honest Assessment — Known Weaknesses

### Methodological

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| M1 | **Self-evaluation bias**: Judge is written by us, then we optimize for it. Score 95 means "95% compliant with our own rules" — not "good music" | High | Not addressed |
| M2 | **No ablation study**: 9+ fixes applied; individual contribution unknown | Medium | Not addressed |
| M3 | **No baseline comparison**: What does a random generator score? A rule-only (no search) system? | Medium | **Addressed** — Random=48.1, Single=94.8, Search=95.1 |
| M4 | **Tiny corpus**: 13 subjects — possible overfitting to these specific subjects | Medium | Not addressed |
| M5 | **Stochastic variance**: Each run gives different scores (±1-2 pts). Need 30-run mean ± std | Low | Not addressed |

### Musical

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| U1 | **Never listened**: No MIDI output has been aurally evaluated | Critical | MIDI generated — awaiting listening |
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

### Phase 1: Listen & Validate (Current)
> Goal: Ground truth check — does this actually sound like music?

- [x] Generate MIDI for all 13 BWV subjects → `output/midi/bwv*.mid`
- [x] Establish baselines: random=48.1, single-pass=94.8, search=95.1
- [ ] Listen to each one, take notes on what sounds wrong
- [ ] Identify gaps between judge scores and aural quality
- [ ] Create a "listening report" mapping subjective impressions to code

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
| Refactor + baselines | **95.1** | generator.py split into 8 modules; baselines established |

---

## File Locations

```
fugueforge/
├── __init__.py
├── cli.py
├── core/
│   ├── __init__.py
│   ├── representation.py    (~280 lines)  Data types, enums, conversions
│   ├── rules.py             (~480 lines)  Counterpoint rule checker
│   ├── analyzer.py          (~540 lines)  Subject/section/cadence detection
│   ├── planner.py           (~470 lines)  Macro structure planning
│   ├── candidates.py        (~290 lines)  Pitch generation & constraint scoring
│   ├── counterpoint.py      (~330 lines)  Free CP & countersubject placement
│   ├── placement.py         (~180 lines)  Subject/answer placement
│   ├── episodes.py          (~190 lines)  Episode generation
│   ├── exposition.py        (~115 lines)  Exposition with CS capture
│   ├── coda.py              (~165 lines)  Cadential coda
│   ├── postprocess.py       (~350 lines)  Post-processing passes
│   ├── voice_utils.py       (~120 lines)  Key helpers, range adaptation
│   ├── generator.py         (~230 lines)  Orchestrator + re-exports
│   ├── search.py            (~310 lines)  MCTS beam search
│   └── judge.py             (~400 lines)  4-axis scorer
├── corpus/
│   ├── __init__.py
│   ├── gold.py              13 BWV subjects
│   ├── loader.py            MIDI/MusicXML loader
│   └── pipeline.py          Evaluation pipeline
├── scripts/
│   └── generate_midi.py     Phase 1 MIDI generation + baselines
├── tests/
│   ├── test_core.py         27 tests
│   └── test_corpus.py       9 tests
└── output/midi/             Generated MIDI files (gitignored)
```
