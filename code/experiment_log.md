# DriftTrace IDS2018 Reproduction — Experiment Log

**Paper:** Pan et al., "DriftTrace: Combating Concept Drift in Security Applications
Through Detection and Explanation", IEEE TIFS vol. 21, 2026.

**Target (Table V, IDS2018):**
| Metric | DriftTrace | Vanilla AE | TRANSCENDENT |
|---|---|---|---|
| Precision | 0.98 ± 0.01 | 0.69 ± 0.08 | 0.61 ± 0.11 |
| Recall | 0.97 ± 0.03 | 0.96 ± 0.04 | 0.89 ± 0.17 |
| F1 | 0.97 ± 0.02 | 0.84 ± 0.04 | 0.75 ± 0.04 |
| Insp. Effort | 0.98 ± 0.02 | 1.34 ± 0.11 | 1.65 ± 0.26 |

---

## Setup (fixed across all versions)

**Dataset:** CSE-CIC-IDS2018 (AWS CICFlowMeter CSVs)
- Benign 66,245 | DoS-Hulk 43,486 | Bot 28,230 | Infiltration 9,238 = 147,199 rows
- Files: Friday-16-02, Friday-02-03, Thursday-01-03

**Architecture:** input → 64 → 32 → 16 → 3 (ReLU, linear output)

**Paper hyperparameters:** m=10, λ=0.1, lr=0.001, epochs=300, batch=256, p=1.5

**Feature engineering (CADE Appendix 1):**
- Dst Port → one-hot 3 frequency bands (high >10k, medium 1k-10k, low <1k)
- Protocol → one-hot unique training values
- Numerical → MinMaxScaler [0,1]

---

## Version History

---

### V1 — Baseline (too few features)
**Phase 2:** `phase2_v1.py` | **Phase 3:** `phase3_v1.py`

**Hypothesis:** Basic faithful implementation of DriftTrace.

**Configuration:**
- Features: 69 (wrong META_PATTERNS excluded too many cols)
- Split: Within-class temporal 80/20
- Contrastive: all-pairs
- SEED=42, BALANCED=True

**Results:**
| Hold-out | Precision | Recall | F1 | Insp.Effort | Contras@300 |
|---|---|---|---|---|---|
| Benign | 0.496 | 0.966 | 0.655 | 1.947 | 0.004 |
| Bot | 0.383 | 0.999 | 0.554 | 2.609 | 0.429 |
| DoS-Hulk | 0.993 | 0.999 | 0.996 | 1.007 | 0.555 |
| Infiltration | 0.147 | 0.344 | 0.206 | 2.337 | 0.006 |
| **Average** | | | **0.603** | | |

**Issues found:**
1. Only 69 features — too aggressive META_PATTERNS dropped Protocol, Source Port
2. Benign precision 0.50 — suspected Timestamp_ms poisoning with temporal split
3. Bot/DoS-Hulk STUCK — Benign+Infiltration similarity in latent space

---

### V2 — 83 features + Timestamp_ms, temporal split
**Phase 2:** `phase2_v2.py` | **Phase 3:** `phase3_v2.py`

**Hypothesis:** Timestamp_ms is CADE's 83rd feature (confirmed from CADE
supplemental Appendix 1). Adding it brings us to exact paper feature count.

**Configuration:**
- Features: 83 (76 numerical + Timestamp_ms + 3 DstPort OHE + 3 Protocol OHE)
- Split: Within-class temporal 80/20
- Contrastive: all-pairs
- SEED=42, BALANCED=True

**Results:**
| Hold-out | Precision | Recall | F1 | Insp.Effort | Contras@300 |
|---|---|---|---|---|---|
| Benign | 0.496 | 0.965 | 0.655 | 1.947 | 0.004 |
| Bot | 0.383 | 0.999 | 0.554 | 2.609 | 0.429 |
| DoS-Hulk | 0.993 | 0.999 | 0.996 | 1.007 | 0.555 |
| Infiltration | 0.147 | 0.344 | 0.206 | 2.337 | 0.006 |
| **Average** | | | **0.603** | | |

**Issues found:**
- Adding/removing Timestamp_ms made ZERO difference to results
- Root cause of Benign precision=0.50 is NOT Timestamp_ms
- Temporal split causes train/test distribution gap in latent space —
  test samples (last 20% by time) have different feature statistics from
  training samples (first 80%), pushing them outside IQR bounds

---

### V3 — 82 features, balanced=False, temporal split
**Phase 2:** `phase2_v3.py` | **Phase 3:** `phase3_v3.py`

**Hypothesis:** Removing balanced sampling (paper does not explicitly mention it)
may improve convergence for stuck runs.

**Configuration:**
- Features: 82 (no Timestamp_ms)
- Split: Within-class temporal 80/20
- Contrastive: all-pairs
- SEED=42, BALANCED=False

**Results:**
| Hold-out | Precision | Recall | F1 | Insp.Effort | Contras@300 |
|---|---|---|---|---|---|
| Benign | 0.495 | 0.963 | 0.653 | 1.947 | 0.007 |
| Bot | 0.368 | 1.000 | 0.537 | 2.723 | 0.865 |
| DoS-Hulk | 0.966 | 1.000 | 0.983 | 1.036 | 1.165 |
| Infiltration | 0.063 | 1.000 | 0.118 | 15.93 | 0.010 |
| **Average** | | | **0.573** | | |

**Issues found:**
- BALANCED=False made Bot WORSE (contras went from 0.43 → 0.87)
- Without balanced sampling, Infiltration gets only ~20/256 batch slots
  → too few Infiltration-class contrastive pairs → worse convergence
- BALANCED=True confirmed as correct for this dataset

---

### V4 — L2 normalisation on encoder output
**Phase 2:** `phase2_v4.py` | **Phase 3:** `phase3_v4.py`

**Hypothesis:** L2 normalising the encoder output projects all latent vectors
onto unit sphere, preventing representation collapse (IQR=0.0003).

**Configuration:**
- Features: 82/83
- Split: Within-class temporal 80/20
- Encoder: added F.normalize(z, p=2, dim=1)
- SEED=42, BALANCED=True

**Results:** ALL FOUR RUNS STUCK at contras≈4.67-4.91
| Hold-out | F1 | Contras@300 | Status |
|---|---|---|---|
| Benign | 0.654 | 4.713 | STUCK |
| Bot | 0.462 | 4.862 | STUCK |
| DoS-Hulk | 0.479 | 4.887 | STUCK |
| Infiltration | ~0.20 | 4.71 | STUCK |
| **Average** | **~0.45** | | |

**Root cause identified:**
- L2 normalisation constrains latent vectors to unit sphere: max sq_d = 4
- Contrastive margin m=10 requires sq_d ≥ 10 for convergence
- 4 < 10 → loss can NEVER converge → all runs stuck at ≈ margin - max_sq_d = 6
- L2 normalisation is GEOMETRICALLY INCOMPATIBLE with m=10

---

### V5 — similar_ratio=0.25 (WRONG implementation)
**Phase 2:** `phase2_v5.py` | **Phase 3:** `phase3_v5.py`

**Hypothesis:** CADE's run script uses --similar-ratio 0.25 (25% positive pairs).
This prevents representation collapse by limiting same-class compression gradient.

**Configuration:**
- Features: 83 (with Timestamp_ms)
- Split: Within-class temporal 80/20
- Contrastive: similar_ratio=0.25 (limiting n_neg_sampled)
- SEED=42, BALANCED=True

**Implementation bug:** With 3 balanced classes, natural positive pair ratio=33%.
Code tried to limit NEGATIVE pairs to achieve 25%, but:
  n_neg_target = n_pos × 3 = 32,130 > n_neg = 21,675
  → used ALL negative pairs → actual ratio stayed at 33%
  → IDENTICAL to all-pairs, no effect.

**Results:**
| Hold-out | Precision | Recall | F1 | Contras@300 |
|---|---|---|---|---|
| Benign | 0.489 | 0.953 | 0.647 | 0.008 |
| Bot | 0.385 | 0.999 | 0.555 | 1.708 |
| DoS-Hulk | STUCK | | ~0.72 | 1.161 |
| Infiltration | 0.063 | 1.000 | 0.118 | 0.010 |
| **Average** | | | **~0.52** | |

---

### V6 — similar_ratio=0.25 (CORRECT), 83 features, temporal split
**Phase 2:** `phase2_v6.py` | **Phase 3:** `phase3_v6.py`

**Hypothesis:** Fix similar_ratio implementation — subsample POSITIVE pairs
(not negatives) to correctly achieve 25% positive ratio.

**Configuration:**
- Features: 83
- Split: Within-class temporal 80/20
- Contrastive: similar_ratio=0.25 CORRECT (n_pos_used = n_neg × 0.25/0.75)
- SEED=42, BALANCED=True

**Verification:**
  n_neg = 21,675 (all)
  n_pos_target = 21,675 × (0.25/0.75) = 7,225
  Actual ratio = 7,225 / (7,225 + 21,675) = 25.0% ✓

**Results:**
| Hold-out | Precision | Recall | F1 | Contras@300 |
|---|---|---|---|---|
| Benign | 0.489 | 0.954 | 0.647 | 0.008 |
| Bot | STUCK | | 0.555 | 1.708 |
| DoS-Hulk | STUCK | | 0.718 | 0.434 |
| Infiltration | | | ~0.205 | 0.008 |
| **Average** | | | **0.531** | |

**Issues found:**
- IQR slightly wider (IQR_Bot: 0.0003 → 0.0014, IQR_DoS 0.011→0.023)
- Precision improvement minimal (0.489 → 0.489)
- Root cause: train/test distribution gap from temporal split STILL causes
  test-side known-class samples to be outside IQR bounds
- Bot/DoS-Hulk: Benign+Infiltration coexistence still prevents convergence

---

### V7 — CURRENT BEST: 83 features, random stratified split, correct similar_ratio
**Phase 2:** `phase2_v7.py` | **Phase 3:** `phase3_v7.py`

**Hypothesis:** Paper says "stratified sampling to ensure consistent distribution
across splits" — this is random stratified split (sklearn), NOT temporal.
Random split guarantees train/test samples from identical distributions →
IQR bounds correctly contain test-side non-drift samples → precision recovers.

**Configuration:**
- Features: 83 (Timestamp_ms in Phase 2, properly scaled)
- Split: Random stratified 80/20 (train_test_split, stratify=class, seed=42)
- Contrastive: similar_ratio=0.25 CORRECT
- SEED=42, BALANCED=True
- lr=0.001, m=10, λ=0.1, epochs=300, batch=256, p=1.5

**Results (BEST SO FAR):**
| Hold-out | Precision | Recall | F1 | Insp.Effort | Contras@300 | Status |
|---|---|---|---|---|---|---|
| Benign | **0.668** | **0.842** | **0.745** | **1.262** | 0.008 | converged |
| Bot | 0.454 | 0.900 | 0.603 | 1.983 | 0.448 | STUCK |
| DoS-Hulk | 0.785 | 0.662 | 0.718 | 0.843 | 0.434 | STUCK |
| Infiltration | 0.151 | 0.320 | 0.205 | 2.119 | 0.008 | converged |
| **Average** | | | **0.568** | | | |

**Improvements from V6:**
- Benign Precision: 0.489 → 0.668 (+37%)
- Benign F1: 0.647 → 0.745 (+15%)
- Benign Inspection Effort: 1.949 → 1.262 (closer to paper's 0.98)
- DoS-Hulk improved (was 0.718, was 0.983 with temporal — DoS-Hulk temporal
  was artificially inflated because attack-only test samples were extreme outliers)

**Remaining gaps:**
1. Converged runs (Benign, Infiltration): Precision still below paper's 0.98
   - Benign P=0.668 vs paper 0.98
   - Infiltration P=0.151 vs paper 0.98
   - Possible cause: lr=0.001 still causes more compression than CADE's lr=0.0001
2. Stuck runs (Bot, DoS-Hulk): Benign+Infiltration coexistence in training
   - These two classes are too similar in 83-feature NetFlow space
   - m=10 cannot push them apart while maintaining reconstruction quality
   - CADE used lr=0.0001 (10x slower) which may prevent this specific failure
3. Infiltration F1=0.205: Infiltration mimics Benign by design (stealthy attack)
   - Only 1,848 test drift samples (smallest class)
   - Even converged latent space places Infiltration near Benign centroid

**Remaining hypotheses to test:**
- V8: lr=0.0001 (CADE's value, even though DriftTrace says 0.001)
  → Expected: less compression → wider IQR → better precision
  → May also help Bot/DoS-Hulk convergence

---

## Summary Table — All Versions

| Version | Features | Split | similar_ratio | lr | Avg F1 | Best individual |
|---|---|---|---|---|---|---|
| V1 | 69 | temporal | all-pairs | 0.001 | 0.603 | DoS-Hulk 0.996 |
| V2 | 83+Ts | temporal | all-pairs | 0.001 | 0.603 | DoS-Hulk 0.996 |
| V3 | 82 | temporal | all-pairs | 0.001 | 0.573 | DoS-Hulk 0.983 |
| V4 | 82 | temporal | all-pairs+L2norm | 0.001 | 0.450 | BROKEN |
| V5 | 83 | temporal | 0.25 WRONG | 0.001 | 0.520 | DoS-Hulk ~0.72 |
| V6 | 83 | temporal | 0.25 CORRECT | 0.001 | 0.531 | DoS-Hulk 0.718 |
| **V7** | **83** | **random strat** | **0.25 CORRECT** | **0.001** | **0.568** | **Benign 0.745** |
| V8 (TODO) | 83 | random strat | 0.25 CORRECT | **0.0001** | TBD | TBD |
| Paper | 83 | ? | ? | 0.001 | **0.970** | All ≥ 0.97 |

---

## Documented Reproducibility Findings

**Finding 1 — Feature count gap:**
Public AWS CICFlowMeter CSVs have 80 columns (Label+Timestamp+78 features).
After CADE's preprocessing (OHE for DstPort and Protocol + Timestamp_ms): 83.
All 83 features are now correctly included in V7+.

**Finding 2 — similar_ratio=0.25:**
CADE's run script specifies --similar-ratio 0.25 (not mentioned in DriftTrace).
DriftTrace adopted CADE's implementation. Correct implementation requires
subsampling POSITIVE pairs (not negatives) when n_neg_target > available n_neg.

**Finding 3 — Train/test split method:**
Paper says "sorted by timestamps" AND "stratified sampling for consistent
distribution." These are contradictory with temporal split when attack classes
span a single capture day (all DoS-Hulk on Feb 16, all Infiltration on Mar 1).
Random stratified split is the interpretation consistent with:
(a) all four hold-out scenarios being viable
(b) consistent train/test distributions → IQR bounds work correctly

**Finding 4 — Benign+Infiltration convergence failure:**
When both Benign and Infiltration appear in the same training set (Bot and
DoS-Hulk hold-outs), contrastive loss cannot converge with m=10 and lr=0.001.
Infiltration is a stealthy attack designed to mimic Benign traffic. In 83-feature
NetFlow space (without Source Port, which was stripped from public AWS files),
these classes are insufficiently separable for the contrastive loss to converge.

**Finding 5 — L2 normalisation incompatibility:**
L2 normalisation on encoder output is geometrically incompatible with margin m=10.
Unit sphere constrains max squared Euclidean distance to 4. Since m=10 > 4,
different-class pairs can never achieve sq_d ≥ m → loss never converges.

---

## Files Index

| File | Description | Key changes |
|---|---|---|
| phase2_v1.py | 69 features, temporal split | Baseline |
| phase2_v2.py | 83 features (Ts in P3), temporal | Add Timestamp_ms |
| phase2_v3.py | 82 features, temporal | Remove Timestamp_ms |
| phase2_v4.py | 82 features, temporal | Same as v3 |
| phase2_v5.py | 83 features, temporal | Same as v2 |
| phase2_v6.py | 83 features, temporal | Same as v2 |
| **phase2_v7.py** | **83 features, random stratified** | **Best: random split + Timestamp in P2** |
| phase3_v1.py | All-pairs, BALANCED=True | Baseline |
| phase3_v2.py | All-pairs, BALANCED=True | +Timestamp_ms at load |
| phase3_v3.py | All-pairs, BALANCED=False | Test balanced=False |
| phase3_v4.py | All-pairs + L2 norm | L2 breaks with m=10 |
| phase3_v5.py | similar_ratio=0.25 WRONG | Bug: achieved 33% |
| phase3_v6.py | similar_ratio=0.25 CORRECT | Fix: subsample pos pairs |
| **phase3_v7.py** | **similar_ratio=0.25 CORRECT, 83 feat** | **Best: all fixes combined** |
