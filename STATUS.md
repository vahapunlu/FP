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
                                  ┌────┴────┐
                                  │ Harmony │
                                  │ Skeleton│
                                  │ pivot   │
                                  │ modul.  │
                                  └────┬────┘
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

### Module Inventory (22 files, ~7,200 lines)

| Module | Purpose | Lines |
|--------|---------|-------|
| `core/representation.py` | Data types: FugueNote, Subject, FuguePlan, enums | ~280 |
| `core/rules.py` | Counterpoint rule checker (parallels, crossing, dissonance, spacing, hidden intervals) | ~480 |
| `core/analyzer.py` | Subject analysis, tonal answer, occurrence finder, section/cadence detection | ~540 |
| `core/planner.py` | Macro structure planning: exposition → episodes → middle entries → stretto → coda | ~470 |
| `core/candidates.py` | Pitch gen & scoring (Phase 2: melodic quality, conjunct motion) | ~360 |
| `core/counterpoint.py` | Free CP & CS placement (Phase 2: phrasing, direction, contour) | ~410 |
| `core/placement.py` | Subject/answer placement & octave adjustment | ~180 |
| `core/episodes.py` | Episode generation (Phase 2: rhythmic/intervallic variation) | ~260 |
| `core/exposition.py` | Exposition with CS capture, overlap prevention | ~125 |
| `core/coda.py` | Cadential coda with scaled V→I, proper voicing | ~170 |
| `core/harmony.py` | Harmonic skeleton, chord progressions, pivot chord modulation | ~300 |
| `core/postprocess.py` | Post-processing passes (parallels, dissonance, leaps, climax) | ~350 |
| `core/voice_utils.py` | Key helpers, voice range adaptation, rest-filling, score output | ~170 |
| `core/generator.py` | Orchestrator + dramaturgy energy curve + re-exports | ~360 |
| `core/search.py` | MCTS-inspired beam search | ~310 |
| `core/judge.py` | 4-axis scorer: theory, structure, style, aesthetic | ~400 |
| `corpus/gold.py` | 13 Bach WTC fugue subjects with full annotations | ~515 |
| `corpus/pipeline.py` | Evaluation pipeline | ~200 |
| `scripts/generate_midi.py` | MIDI generation + baselines | ~220 |
| `scripts/ablation_study.py` | Ablation study (M2) | ~135 |
| `scripts/stat_benchmark.py` | Statistical benchmarking (M5) | ~60 |

---

## Current Benchmark (v0.6 — Phase 2 complete)

```
BWV    Voices  Random  Single  Search  Theory  Struct  Style  Aesth
─────────────────────────────────────────────────────────────────────
846    4        42.1    93.6    90.6    75.5   100.0   89.7  100.0
847    3        49.4    94.5    92.2    86.4   100.0   81.4  100.0
848    3        53.1    93.7    93.1    90.9   100.0   79.4  100.0
849    5        34.0    90.4    90.2    77.2   100.0   87.2   98.0
850    4        41.6    89.1    91.4    83.1   100.0   82.5  100.0
851    3        51.1    94.4    94.2    89.0   100.0   87.7  100.0
853    3        48.3    95.3    95.3    93.3   100.0   88.8   98.0
855    2        61.1    96.3    95.5    94.5   100.0   85.9  100.0
858    3        52.5    94.9    96.4    94.1   100.0   91.0  100.0
860    3        51.8    94.6    95.0    90.8   100.0   89.0  100.0
861    4        39.8    93.3    93.0    83.5   100.0   89.8  100.0
865    4        41.1    90.0    90.0    79.6   100.0   80.4  100.0
869    4        41.2    94.1    94.5    88.7   100.0   89.6  100.0
─────────────────────────────────────────────────────────────────────
AVG             46.7    93.4    93.2    86.7   100.0   86.3   99.7
```

**Statistical benchmark** (5 runs × 13 BWVs): **92.1 ± 2.3** (per-BWV std: 0.4–1.5)

**Baseline comparison**: Random=46.7, Single-pass=93.4, Search=93.2

**36/36 tests passing.** MIDI files generated for all 13 BWVs.

---

## Ablation Study Results

| Condition | AVG | Δ | Impact |
|-----------|-----|---|--------|
| Full system (baseline) | 92.1 | — | BASELINE |
| − Scale awareness | 92.0 | −0.1 | LOW |
| − Harmonic skeleton | 91.9 | −0.2 | LOW |
| − Melodic quality (Phase 2) | 92.3 | +0.2 | LOW |
| − Phrasing & direction | 91.8 | −0.4 | LOW |

**Key insight**: All features show LOW judge-score impact because the judge primarily
measures counterpoint rule compliance (M1 bias). Phase 2 improvements target **aural quality**
which the current judge cannot measure. This confirms the need for Phase 3 neural scoring.

---

## Honest Assessment — Known Weaknesses

### Methodological

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| M1 | **Self-evaluation bias**: Judge measures rule compliance, not musical quality. Ablation confirms: melodic improvements don't change judge score | High | **Confirmed via ablation** — needs Phase 3 neural scorer |
| M2 | **Ablation study**: Feature contributions measured | Medium | **Done** — all features <0.5pt judge impact |
| M3 | **Baseline comparison**: Random vs single-pass vs search | Medium | **Done** — Random=46.7, Single=93.4, Search=93.2 |
| M4 | **Tiny corpus**: 13 subjects — possible overfitting | Medium | Not addressed |
| M5 | **Stochastic variance**: Per-run variance measured | Low | **Done** — 92.1 ± 2.3 (5 runs × 13 BWVs) |

### Musical

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| U1 | **Listening feedback**: 4 rounds of aural evaluation | Critical | **Done** — dissonance, tonality, voice quality, melody |
| U2 | **Harmonic rhythm**: Chord progressions guide strong/weak beats | High | **Done** — harmony.py |
| U3 | **Phrasing/breath**: Phrase boundaries with contour | High | **Done** — phrase_length, arch contour |
| U4 | **Modulation depth**: Pivot chord modulations | High | **Done** — _find_pivot_chord() |
| U5 | **Episode variation**: Rhythmic & intervallic variation | Medium | **Done** — _vary_rhythm(), _vary_interval() |
| U6 | **Dramaturgy**: Energy curve across sections | Medium | **Done** — _apply_dramaturgy() |

### Technical Ceiling

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| T1 | **Theory in 4/5-voice fugues**: 80-87 range due to combinatorial explosion of voice pairs | High | Partially addressed |
| T2 | **Style variance**: 81-97 range; some BWVs still have high unresolved leap ratios | Medium | Partially addressed |
| T3 | **Search barely beats gen**: +0.2-0.5 only; search not exploiting its potential | Low | Not addressed |

---

## Roadmap

### Phase 1: Listen & Validate ✅ COMPLETE
> Goal: Ground truth check — does this actually sound like music?

- [x] Generate MIDI for all 13 BWV subjects → `output/midi/bwv*.mid`
- [x] Establish baselines: random=46.7, single-pass=93.4, search=93.2
- [x] Listen to each one, take notes on what sounds wrong
- [x] Identify gaps between judge scores and aural quality
- [x] Listening feedback: 4 rounds (dissonance → tonality → harmony → melody)

### Phase 2: Musical Intelligence ✅ COMPLETE
> Goal: Address the musical weaknesses that listening reveals

- [x] **Phrasing**: Phrase boundaries every ~4 beats with longer notes, arch contour
- [x] **Harmonic rhythm**: harmony.py with chord progressions, chord tone bonuses
- [x] **Episode variation**: Rhythmic variation (agogic accents, dotted rhythms), intervallic variation
- [x] **Modulation depth**: Pivot chord modulations via _find_pivot_chord()
- [x] **Dramaturgy**: 4-phase energy curve (calm→building→climax→resolution)
- [x] **Melodic quality**: Stepwise +8, consecutive leap −10, direction momentum, register gravity
- [x] **Ablation study**: All features measured — confirms M1 judge bias
- [x] **Statistical benchmarking**: 92.1 ± 2.3 (5 runs × 13 BWVs)

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
| Language | Python 3.12.12 |
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
| Scale + harmony | **93.8** | Scale awareness, harmonic skeleton with chord progressions |
| Voice/timing fixes | **93.8** | Voice ranges, episode stagger, rest-filling, coda voicing |
| Phase 2: melody | **93.9** | Stepwise +8, consecutive leap −10, phrasing, contour |
| **Phase 2: complete** | **93.2** | Episode variation, pivot modulation, dramaturgy (stat: 92.1±2.3) |

> Note: Score decrease from 95.1 → 93.2 is expected: Phase 2 prioritizes aural quality
> over judge compliance. Ablation confirms the judge cannot measure melodic improvements.

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
│   ├── candidates.py        (~360 lines)  Pitch gen & scoring (Phase 2: melodic quality)
│   ├── counterpoint.py      (~410 lines)  Free CP & CS (Phase 2: phrasing, contour)
│   ├── placement.py         (~180 lines)  Subject/answer placement
│   ├── episodes.py          (~260 lines)  Episode gen (Phase 2: variation)
│   ├── exposition.py        (~125 lines)  Exposition with CS capture
│   ├── coda.py              (~170 lines)  Cadential coda
│   ├── harmony.py           (~300 lines)  Harmonic skeleton & pivot modulation
│   ├── postprocess.py       (~350 lines)  Post-processing passes
│   ├── voice_utils.py       (~170 lines)  Key helpers, range adaptation, rest-filling
│   ├── generator.py         (~360 lines)  Orchestrator + dramaturgy + re-exports
│   ├── search.py            (~310 lines)  MCTS beam search
│   └── judge.py             (~400 lines)  4-axis scorer
├── corpus/
│   ├── __init__.py
│   ├── gold.py              13 BWV subjects
│   ├── loader.py            MIDI/MusicXML loader
│   └── pipeline.py          Evaluation pipeline
├── scripts/
│   ├── generate_midi.py     MIDI generation + baselines
│   ├── ablation_study.py    Ablation study (M2)
│   └── stat_benchmark.py    Statistical benchmarking (M5)
├── tests/
│   ├── test_core.py         27 tests
│   └── test_corpus.py       9 tests
└── output/midi/             Generated MIDI files (gitignored)
```
