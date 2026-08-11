# DriftTrace Reproduction Study — IDS2018

**Author:** Nadeem Yousaf  
**Affiliation:** NIMBUS Research Centre, Munster Technological University (MTU), Cork, Ireland  
**Repository:** [github.com/MNadeemYousaf](https://github.com/MNadeemYousaf)

---

## Overview

This repository contains a systematic reproduction study of the paper:

> Pan, Y., Zhao, L., Leng, T., Luo, Z., Cai, L., Yu, A., & Meng, D. (2026).
> **DriftTrace: Combating Concept Drift in Security Applications Through Detection and Explanation.**
> *IEEE Transactions on Information Forensics and Security*, vol. 21, pp. 1957–1972.
> DOI: 10.1109/TIFS.2026.3659398

The study evaluates whether the reported results for the IDS2018 dataset
(F1 = 0.97 ± 0.02, boundary crossing = 83.65%) can be independently reproduced
using the methodology described in the paper and the publicly available
CSE-CIC-IDS2018 dataset.

This reproduction work is conducted as part of a journal research project on
**Human-Centred Explanations of AI-based Intrusion Detection Systems for SOC
under Concept Drift**, building on DriftTrace as the base detection framework.

---

## Repository Structure

```
drifttrace_reproduction/
├── README.md                        ← this file
├── EXPERIMENT_REPORT.md             ← full experiment log (all 13 versions)
├── REPRODUCIBILITY_ANALYSIS.md      ← analysis of why exact reproduction failed
├── JOURNAL_SECTION.md               ← DriftTrace critique for journal paper
├── code/
│   ├── phase2_v1.py                 ← baseline: 69 features, temporal split
│   ├── phase2_v2.py → v6.py        ← feature engineering iterations
│   ├── phase2_v7.py                 ← random stratified split (explored)
│   ├── phase2_v10.py                ← FINAL Phase 2: 83 features, temporal 80/20
│   ├── phase3_v1.py → v10.py       ← detection model iterations
│   ├── phase34_v11.py               ← V11: exact paper specification
│   ├── phase34_v12.py               ← V12: similar_ratio=0.10 explored
│   └── experiment_log.md            ← chronological experiment notes
└── results/
    └── results_summary.md           ← all version results in one table
```

---

## Dataset

**CSE-CIC-IDS2018** — publicly available via AWS S3:
```
aws s3 sync --no-sign-request s3://cse-cic-ids2018/ .
```

Files used:
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv`
- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`

Classes: Benign (66,245) | DoS-Hulk (43,486) | Bot (28,230) | Infiltration (9,238)  
Total: 147,199 rows — matching paper Table IV exactly.

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install torch pandas numpy scikit-learn pyarrow matplotlib
```

### Step 2 — Configure paths
Edit `DATA_DIR` and `OUT_DIR` in `phase2_v10.py` and `phase34_v11.py`
to point to your dataset location and output directory.

### Step 3 — Feature engineering and split
```bash
python code/phase2_v10.py
```

### Step 4 — Detection + Explanation (best reproduction version)
```bash
python code/phase34_v11.py
```

For the version closest to the paper's reported numbers, use V10:
```bash
# Edit phase34_v11.py: set SIMILAR_RATIO = 0.25
python code/phase34_v11.py
```

---

## Key Results Summary

### Detection (Table V equivalent)

| Method | Precision | Recall | F1 | Insp. Effort |
|---|---|---|---|---|
| **Paper (DriftTrace)** | **0.98 ± 0.01** | **0.97 ± 0.03** | **0.97 ± 0.02** | **0.98 ± 0.02** |
| Our best (V10, avg) | 0.78 | 0.96 | 0.84 | 1.38 |
| Vanilla AE (paper) | 0.69 ± 0.08 | 0.96 ± 0.04 | 0.84 ± 0.04 | 1.34 ± 0.11 |
| TRANSCENDENT (paper) | 0.61 ± 0.11 | 0.89 ± 0.17 | 0.75 ± 0.04 | 1.65 ± 0.26 |

Our reproduction (V10) surpasses both paper baselines and achieves
DoS-Hulk F1 = 0.960, matching the paper's claim for that class.

### Explanation (Table VIII / Table VI equivalent)

| Metric | Our best | Paper (Infiltration) |
|---|---|---|
| Fidelity d_final | 0.020 | 0.0325 |
| Boundary crossing | 40.5% | 83.65% |

---

## Documented Implementation Details

The following table maps every paper-stated parameter to our implementation:

| Parameter | Paper states | Our implementation |
|---|---|---|
| Architecture | 83→64→32→16→3 | ✓ identical |
| Distance metric | Regular L2 (Eq. 2) | ✓ `(diff².sum + ε).sqrt()` |
| Reconstruction loss | MSE (mean) | ✓ `F.mse_loss` |
| Optimizer | Adam, lr=0.001 | ✓ identical |
| Epochs | 300 | ✓ identical |
| Batch size | 256 | ✓ identical |
| Margin m | 10 | ✓ identical |
| Lambda λ | 0.1 | ✓ identical |
| IQR coefficient p | 1.5 | ✓ identical |
| Features | 83 (CADE Appendix 1) | ✓ identical |
| Balanced batches | Yes | ✓ WeightedRandomSampler |
| Hold-out protocol | 0/100 unseen | ✓ zero drift in training |
| Algorithm 1 | Exact pseudocode | ✓ line-by-line |
| Algorithm 2 | Exact pseudocode | ✓ line-by-line |

**Undocumented parameter (CADE inheritance):**
`similar_ratio` — controls positive-pair proportion in contrastive loss.
Not stated in DriftTrace. CADE uses 0.25. We tested 0.05, 0.10, 0.25, all-pairs.

---

## Contact

Nadeem Yousaf  
NIMBUS Research Centre  
Munster Technological University, Cork, Ireland  
Email: [institutional email]

Corresponding author of DriftTrace:  
Dr. Lixin Zhao — zhaolixin@iie.ac.cn
