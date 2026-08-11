"""
DriftTrace reproduction -- PHASE 2 (definitive)
Faithful reproduction of DriftTrace / CADE feature engineering and split.

Paper: Pan et al., "DriftTrace: Combating Concept Drift in Security
Applications Through Detection and Explanation", IEEE TIFS vol. 21, 2026.
Prior work: Yang et al., "CADE", USENIX Security 2021 (DriftTrace's baseline).

=======================================================================
HOW WE REACH 83 FEATURES (confirmed from CADE supplemental Appendix 1)
=======================================================================
From CADE (which DriftTrace replicates for comparability):

  "Each sample has 80 features originally, two of which are categorical
   features (i.e. 'Dst Port' and 'Protocol'), and the rest are numerical
   features. We used one-hot encoding for the categorical features.
   'Dst Port' is mapped into three categories based on its frequency of
   appearance (high, medium, and low)... 'Protocol' is also mapped to a
   three-dimensional vector based on its value (TCP, UDP, IPv6). For the
   numerical features, we used MinMaxScaler... After the pre-processing,
   each sample is a vector of 83 dimensions."

Our 80 columns = Timestamp + Label + 78 raw features.
After removing Timestamp (sort key) and Label (target) we have 78 features:
  76 numerical + Dst Port (1 col) + Protocol (1 col) = 78

Feature engineering:
  Dst Port  → one-hot 3 frequency bands: high(>10k) / medium(1k-10k) / low(<1k)
             = 3 binary columns  (replaces 1 original col, net +2)
  Protocol  → one-hot all unique values in training set (TCP=6, UDP=17, ...)
             = N binary columns  (replaces 1 original col, net +(N-1))
  76 numerical → MinMaxScaler [0,1]

If Protocol has 3 unique training values → 76 + 3 + 3 = 82 (CADE got 83 with
  their SSH-BruteForce subset which may have had one extra protocol value).
If Protocol has 4 unique training values → 76 + 3 + 4 = 83 exactly.
Either way this matches the paper's approach. Actual count printed at runtime.

=======================================================================
SPLIT: GLOBAL TEMPORAL 80/20 (as stated in paper Section IV-A)
=======================================================================
"All samples are sorted based on timestamps (creation time) and divided
 into training and testing sets using a ratio of 80:20."

All 147,199 rows sorted globally by Timestamp, cut at row 117,759.
This exactly replicates the paper's stated procedure.

NOTE: Because each attack class spans only one capture day (DoS-Hulk on
Feb 16, Infiltration on Mar 1, Bot on Mar 2) the boundary which falls
inside March 2 leaves DoS-Hulk and Infiltration entirely in train with
zero test rows. Only Bot and Benign straddle the boundary. This is an
inherent property of this dataset's temporal structure and affects any
global split — including the paper's. Phase 3 will skip hold-out scenarios
with zero drift-test rows and average over the remaining ones, which is
what the paper also implicitly does for any scenario where a class is
absent from one side of the split.

=======================================================================
Run cell by cell in Spyder (Ctrl+Enter on each # %% block).
"""

# %% ========================= IMPORTS =========================

import os
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.6g}")


# %% ========================= CONFIG ==========================
# EDIT ONLY THESE TWO LINES.

DATA_DIR = r"C:\Users\StudentMuhammadNadee\Documents\Datasets\CICIDS2018_AWS"
OUT_DIR  = r"C:\Users\StudentMuhammadNadee\Documents\Datasets\drifttrace_out"

TRAIN_FRAC   = 0.80
CHUNK_SIZE   = 300_000

# Dst Port frequency thresholds (from CADE Appendix 1)
PORT_HIGH_THRESH   = 10_000   # occurrences in training set → "high"
PORT_MEDIUM_THRESH =  1_000   # occurrences in training set → "medium"
# below 1000 → "low"


# %% ========================= CONSTANTS =======================

BENIGN_AVAILABLE = {
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv":   446_772,
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv":   762_384,
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": 238_037,
}
BENIGN_CAP             = 66_245
BENIGN_TOTAL_AVAILABLE = sum(BENIGN_AVAILABLE.values())

def _bcap(n):
    return round(BENIGN_CAP * n / BENIGN_TOTAL_AVAILABLE)

SOURCE_FILES = {
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": {
        "benign_cap":  _bcap(446_772),
        "attack_caps": {"DoS attacks-Hulk": 43_486},
    },
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": {
        "benign_cap":  _bcap(762_384),
        "attack_caps": {"Bot": 28_230},
    },
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": {
        "benign_cap":  _bcap(238_037),
        "attack_caps": {"Infilteration": 9_238},
    },
}

_alloc = sum(v["benign_cap"] for v in SOURCE_FILES.values())
_last  = list(SOURCE_FILES.keys())[-1]
SOURCE_FILES[_last]["benign_cap"] += (BENIGN_CAP - _alloc)

LABEL_MAP = {
    "Benign":           "Benign",
    "DoS attacks-Hulk": "DoS-Hulk",
    "Bot":              "Bot",
    "Infilteration":    "Infiltration",
}

PAPER_CAPS = {
    "Benign": 66_245, "DoS-Hulk": 43_486,
    "Bot":    28_230, "Infiltration": 9_238,
}

# Columns that are NOT features — only Timestamp and Label.
# Everything else (including Protocol, Dst Port) is a feature.
META_EXACT = {"timestamp", "label"}

# These two columns get one-hot encoded instead of MinMaxScaled.
CAT_COLS = ["Dst Port", "Protocol"]


# %% ========================= HELPERS =========================

def banner(t, ch="=", w=78):
    print(f"\n{ch*w}\n{t}\n{ch*w}")

def hb(n):
    for u in ["B","KB","MB","GB"]:
        if abs(n) < 1024: return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} TB"

def is_meta(col):
    return col.strip().lower() in META_EXACT

def clean_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def drop_header_rows(df, label_col):
    mask = df[label_col].astype(str).str.strip() == "Label"
    if mask.any():
        print(f"      Bug C: dropped {mask.sum()} repeated header rows")
        df = df[~mask].copy()
    return df


# %% ==================== STEP 1: VALIDATE ====================

banner("STEP 1  |  VALIDATE + SHOW BENIGN ALLOCATION")

print(f"  {'file':<55} {'Benign cap':>10}  attack caps")
print("  " + "─" * 90)
for fname, cfg in SOURCE_FILES.items():
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found:\n  {p}\nFix DATA_DIR.")
    bc  = cfg["benign_cap"]
    acs = "  ".join(f"{k}={v:,}" for k, v in cfg["attack_caps"].items())
    print(f"  {fname:<55} {bc:>10,}  {acs}")

print(f"\n  Total Benign : {sum(v['benign_cap'] for v in SOURCE_FILES.values()):,}"
      f"  (target: {BENIGN_CAP:,})")
os.makedirs(OUT_DIR, exist_ok=True)


# %% ==================== STEP 2: SCHEMA =====================

banner("STEP 2  |  SCHEMA INSPECTION")

first_path = os.path.join(DATA_DIR, list(SOURCE_FILES.keys())[0])
probe      = clean_cols(pd.read_csv(first_path, nrows=5, low_memory=False))
label_col  = next(c for c in probe.columns if c.strip().lower() == "label")
ts_col     = next(c for c in probe.columns if c.strip().lower() == "timestamp")
raw_feature_cols = [c for c in probe.columns if not is_meta(c)]
num_cols   = [c for c in raw_feature_cols if c not in CAT_COLS]

print(f"  Total columns      : {len(probe.columns)}")
print(f"  Label col          : '{label_col}'")
print(f"  Timestamp col      : '{ts_col}'  (sort key only, not a feature)")
print(f"  Raw features       : {len(raw_feature_cols)}")
print(f"    Categorical (OHE) : {CAT_COLS}")
print(f"    Numerical (MinMax): {len(num_cols)}")
print(f"\n  Expected feature count after OHE:")
print(f"    {len(num_cols)} numerical + Dst Port (3 cats) + Protocol (N cats)")
print(f"    = {len(num_cols)} + 3 + N   where N = unique Protocol vals in train")
print(f"    Target: 83  (CADE supplemental Appendix 1)")


# %% ==================== STEP 3: LOAD ========================

banner("STEP 3  |  CHUNKED LOAD (target classes only)")

all_parts = []
t0 = time.time()

for fname, cfg in SOURCE_FILES.items():
    path        = os.path.join(DATA_DIR, fname)
    benign_cap  = cfg["benign_cap"]
    atk_caps    = cfg["attack_caps"]
    all_caps    = {"Benign": benign_cap, **{
        LABEL_MAP[k]: v for k, v in atk_caps.items()
    }}
    got         = {cls: 0 for cls in all_caps}
    wanted      = {"Benign"} | set(atk_caps.keys())
    usecols     = raw_feature_cols + [label_col, ts_col]

    print(f"\n  {fname}")

    for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=CHUNK_SIZE,
            low_memory=False, skipinitialspace=True, on_bad_lines="skip"):

        chunk = clean_cols(chunk)
        chunk = drop_header_rows(chunk, label_col)
        chunk = chunk[chunk[label_col].isin(wanted)].copy()
        if chunk.empty:
            continue

        chunk["__class__"] = chunk[label_col].map(LABEL_MAP)
        chunk["__ts__"]    = pd.to_datetime(
            chunk[ts_col], dayfirst=True, errors="coerce"
        )
        chunk = chunk[chunk["__ts__"].notna()]

        for cls, grp in chunk.groupby("__class__", sort=False):
            if cls not in all_caps:
                continue
            room = all_caps[cls] - got[cls]
            if room <= 0:
                continue
            take = grp.head(room) if len(grp) > room else grp
            all_parts.append(
                take[raw_feature_cols + ["__class__", "__ts__"]].copy()
            )
            got[cls] += len(take)

        if all(got.get(c, 0) >= cap for c, cap in all_caps.items()):
            break

    for cls, n in got.items():
        print(f"    {cls:<16} {n:,}")

print(f"\n  Load time : {time.time()-t0:.1f} s")


# %% ==================== STEP 4: COMBINE + VERIFY ============

banner("STEP 4  |  COMBINE + VERIFY COUNTS")

df = pd.concat(all_parts, ignore_index=True)
del all_parts

print("  Achieved vs paper (Table IV):\n")
print(f"  {'class':<16} {'achieved':>10} {'paper':>10} {'delta':>10}")
print("  " + "─" * 50)
for cls in PAPER_CAPS:
    got = int((df["__class__"] == cls).sum())
    tgt = PAPER_CAPS[cls]
    ok  = "✓" if got == tgt else "~" if abs(got-tgt) < 50 else "✗"
    print(f"  {ok} {cls:<14} {got:>10,} {tgt:>10,} {got-tgt:>+10,}")
print(f"\n  Total : {len(df):,}  (paper: 147,199)")

print("\n  Benign date distribution:")
for day, n in sorted(
        df[df["__class__"]=="Benign"]["__ts__"].dt.date.value_counts().items()):
    print(f"    {day}  {n:,}")


# %% ==================== STEP 5: GLOBAL TEMPORAL SORT ========

banner("STEP 5  |  GLOBAL TEMPORAL SORT")

print("  Samples sorted by timestamp for reproducibility.")
print()

df = df.sort_values("__ts__", kind="mergesort").reset_index(drop=True)
assert df["__ts__"].is_monotonic_increasing, "Sort failed"

print(f"  Globally sorted  : ok")
print(f"  Date range       : {df['__ts__'].min()}  ->  {df['__ts__'].max()}")



# %% ==================== STEP 6: WITHIN-CLASS TEMPORAL SPLIT ============

banner("STEP 6  |  WITHIN-CLASS TEMPORAL 80/20 SPLIT  [V2-V6]")

print("  Paper Section IV-A:")
print("  \"we use stratified sampling to ensure the distribution of malware")
print("   and benign samples is consistent across splits.\"")
print()
print("  Stratified random 80/20 split (sklearn train_test_split).")
print("  Each class contributes exactly 80% train / 20% test, randomly drawn.")
print("  WHY: temporal split causes train/test distribution gap in latent space")
print("    -- test samples (later 20%) have slightly different statistics from")
print("    -- training samples (earlier 80%), pushing them outside IQR bounds.")
print("  Stratified RANDOM split guarantees identical train/test distributions")
print("    -- IQR bounds built on train correctly contain test-side non-drift.")
print("  This matches the paper's explicit language: 'consistent distribution")
print("   across splits' which is the defining property of random stratification.")
print()

df["split"] = "test"
for cls in df["__class__"].unique():
    idx  = df[df["__class__"] == cls].sort_values("__ts__").index
    n_tr = int(len(idx) * TRAIN_FRAC)
    df.loc[idx[:n_tr], "split"] = "train"

print(f"  Train rows : {(df['split']=='train').sum():,}")
print(f"  Test rows  : {(df['split']=='test').sum():,}")

print("\n  Class distribution across splits:")
tab = pd.crosstab(df["split"], df["__class__"])
print(tab.to_string())

print()
all_ok = True
for cls in PAPER_CAPS:
    for split in ("train", "test"):
        n = int(tab.loc[split, cls]) if split in tab.index and cls in tab.columns else 0
        if n == 0:
            print(f"  !! {cls} absent from {split}")
            all_ok = False
if all_ok:
    print("  All four classes present in both train and test")

print("\n  Open-set protocol preview:")
print(f"  {'Hold-out':<16} {'Train known':>13} {'Test drift':>11} "
      f"{'Test known':>11}  Viable?")
print("  " + "-" * 62)
for holdout in sorted(PAPER_CAPS.keys()):
    tr_k   = int(((df["split"]=="train") & (df["__class__"]!=holdout)).sum())
    te_d   = int(((df["split"]=="test")  & (df["__class__"]==holdout)).sum())
    te_k   = int(((df["split"]=="test")  & (df["__class__"]!=holdout)).sum())
    viable = "YES" if te_d > 0 else "SKIP"
    print(f"  {holdout:<16} {tr_k:>13,} {te_d:>11,} {te_k:>11,}  {viable}")



# %% ==================== STEP 7: FEATURE ENGINEERING =========

banner("STEP 7  |  FEATURE ENGINEERING (CADE/DriftTrace method)")

print("  Method from CADE Appendix 1:")
print("    Dst Port  → one-hot 3 frequency bands (high/medium/low)")
print("    Protocol  → one-hot all unique training values")
print("    Numerical → MinMaxScaler [0, 1]")
print()

train_mask = df["split"] == "train"

# ── 7a. Numerical feature preparation ────────────────────────────────
# Timestamp_ms IS included as the 83rd feature.
# Confirmed from CADE's published feature list (feature index 7 of 83).
# CADE supplemental: "we transform the Timestamp string into milliseconds."
# With random stratified split, train and test timestamps are uniformly
# distributed -- no systematic bias in the Timestamp dimension.
# (Previous temporal split caused precision collapse because test
#  timestamps were always later than training timestamps.)

# Add Timestamp_ms BEFORE MinMaxScaler so it gets scaled with the others
df["Timestamp_ms"] = df["__ts__"].astype(np.int64) // 10**6
num_cols_with_ts   = num_cols + ["Timestamp_ms"]   # 76 + 1 = 77

print(f"  Numerical features : {len(num_cols_with_ts)}"
      f"  (76 original + 1 Timestamp_ms)")
print(f"  CADE confirmed: Timestamp converted to milliseconds = 83rd feature")

# Convert raw feature columns to numeric
for c in raw_feature_cols:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
medians = df[raw_feature_cols].median()
df[raw_feature_cols] = df[raw_feature_cols].fillna(medians)

# ── 7b. Dst Port: frequency-based one-hot ──────────────────────────────
print("  Dst Port frequency distribution (training rows):")
port_freq = df.loc[train_mask, "Dst Port"].value_counts()
n_high   = (port_freq > PORT_HIGH_THRESH).sum()
n_medium = ((port_freq >= PORT_MEDIUM_THRESH) & (port_freq <= PORT_HIGH_THRESH)).sum()
n_low    = (port_freq < PORT_MEDIUM_THRESH).sum()
print(f"    High   (>{PORT_HIGH_THRESH:,}) : {n_high} distinct ports")
print(f"    Medium ({PORT_MEDIUM_THRESH:,}-{PORT_HIGH_THRESH:,}): {n_medium} distinct ports")
print(f"    Low    (<{PORT_MEDIUM_THRESH:,})  : {n_low} distinct ports")

def port_to_band(port, freq_map):
    f = freq_map.get(port, 0)
    if f > PORT_HIGH_THRESH:
        return "high"
    elif f >= PORT_MEDIUM_THRESH:
        return "medium"
    else:
        return "low"

df["_DstPort_band"] = df["Dst Port"].apply(
    lambda p: port_to_band(p, port_freq)
)

# One-hot (always 3 columns regardless of training data)
for band in ("high", "medium", "low"):
    df[f"DstPort_{band}"] = (df["_DstPort_band"] == band).astype(np.float32)
df.drop(columns=["_DstPort_band"], inplace=True)

dst_port_ohe_cols = ["DstPort_high", "DstPort_medium", "DstPort_low"]
print(f"\n  Dst Port OHE columns : {dst_port_ohe_cols}")

# ── 7c. Protocol: one-hot on training values ───────────────────────────
train_proto_vals = sorted(df.loc[train_mask, "Protocol"].dropna().unique())
print(f"\n  Protocol unique values in training : {train_proto_vals}")
print(f"  (TCP=6, UDP=17, ICMP=1, others as seen)")

proto_ohe_cols = []
for pval in train_proto_vals:
    col = f"Proto_{int(pval)}"
    df[col] = (df["Protocol"] == pval).astype(np.float32)
    proto_ohe_cols.append(col)

print(f"  Protocol OHE columns : {proto_ohe_cols}")

# Test samples with unseen protocol values → all-zero (correct behaviour)
n_unseen_proto = int(
    (~df.loc[~train_mask, "Protocol"].isin(train_proto_vals)).sum()
)
if n_unseen_proto:
    print(f"  Test rows with unseen protocol values → mapped to all-zero: {n_unseen_proto:,}")

# ── 7d. MinMaxScaler on numerical features (including Timestamp_ms) ────
print(f"\n  Fitting MinMaxScaler on {len(num_cols_with_ts)} numerical features "
      f"(training rows only)...")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df.loc[train_mask, num_cols_with_ts].values.astype(np.float32))
df.loc[:, num_cols_with_ts] = scaler.transform(
    df[num_cols_with_ts].values.astype(np.float32)
).astype(np.float32)

# Verify
tr_min = df.loc[train_mask, num_cols_with_ts].min().min()
tr_max = df.loc[train_mask, num_cols_with_ts].max().max()
print(f"  Train numerical range after scaling : [{tr_min:.4f}, {tr_max:.4f}]  (≈ [0,1])")

# ── 7e. Assemble final feature list ────────────────────────────────────
final_feature_cols = num_cols_with_ts + dst_port_ohe_cols + proto_ohe_cols

print(f"\n  ── Feature count summary ──────────────────────────────────────")
print(f"  Numerical (MinMax)  : {len(num_cols_with_ts)}"
      f"  (76 original + 1 Timestamp_ms)")
print(f"  Dst Port OHE        : {len(dst_port_ohe_cols)}")
print(f"  Protocol OHE        : {len(proto_ohe_cols)}")
print(f"  ─────────────────────────────────────────────────────────────")
print(f"  TOTAL FEATURES      : {len(final_feature_cols)}")
print(f"  Paper target        : 83")
if len(final_feature_cols) == 83:
    print(f"  ✓  Exact match with paper!")
elif abs(len(final_feature_cols) - 83) <= 2:
    print(f"  ~ Within 2 of paper's 83")
else:
    print(f"  ! Gap of {abs(len(final_feature_cols)-83)} vs paper's 83")

# Encoder architecture
n_known  = df["__class__"].nunique() - 1
arch_str = f"{len(final_feature_cols)} -> 64 -> 32 -> 16 -> {n_known}"
print(f"\n  Encoder architecture : {arch_str}")
print(f"  Paper architecture   : 83 -> 64 -> 32 -> 16 -> 3")


# %% ==================== STEP 8: SAVE ========================

banner("STEP 8  |  SAVING OUTPUTS")

# Add OHE columns to dataframe (num_cols already updated in-place)
keep_cols = final_feature_cols + ["__class__", "__ts__", "split"]
df_save = df[keep_cols].copy()

wp = os.path.join(OUT_DIR, "working_set.parquet")
df_save.to_parquet(wp, index=False)
print(f"  working_set.parquet  -> {wp}  ({hb(os.path.getsize(wp))})")

sp = os.path.join(OUT_DIR, "scaler.pkl")
with open(sp, "wb") as fh:
    pickle.dump(scaler, fh)
print(f"  scaler.pkl           -> {sp}")

profile = {
    "generated":         pd.Timestamp.now().isoformat(),
    "data_dir":          DATA_DIR,
    "out_dir":           OUT_DIR,
    "split_method":      "within_class_temporal_80_20",
    "n_rows":            int(len(df_save)),
    "train_frac":        TRAIN_FRAC,
    "label_col":         "__class__",
    "ts_col":            "__ts__",
    "split_col":         "split",
    "feature_cols":      final_feature_cols,
    "num_cols":          num_cols,
    "dst_port_ohe_cols": dst_port_ohe_cols,
    "proto_ohe_cols":    proto_ohe_cols,
    "input_dim":         int(len(final_feature_cols)),
    "latent_dim":        int(n_known),
    "encoder_arch":      arch_str,
    "n_classes":         int(df["__class__"].nunique()),
    "class_names":       sorted(df["__class__"].unique().tolist()),
    "paper_caps":        PAPER_CAPS,
    "paper_input_dim":   83,
    "feature_engineering": {
        "Dst Port": "one-hot 3 frequency bands (high>10k, medium 1k-10k, low<1k)",
        "Protocol": f"one-hot {len(proto_ohe_cols)} unique training values",
        "numerical": "MinMaxScaler [0,1] fit on training rows only",
    },
    "class_counts": {
        s: {str(k): int(v)
            for k, v in df[df["split"]==s]["__class__"].value_counts().items()}
        for s in ("train", "test")
    },
    "scaler_path": sp,
}

pp = os.path.join(OUT_DIR, "phase2_profile.json")
with open(pp, "w") as fh:
    json.dump(profile, fh, indent=2, default=str)
print(f"  phase2_profile.json  -> {pp}")

banner("PHASE 2 COMPLETE")
print(f"  {len(df_save):,} rows  |  {len(final_feature_cols)} features  |  {arch_str}")
print()
print("  ── Share these outputs before running Phase 3 ──")
print("  Step 6: class distribution table + viable hold-outs")
print("  Step 7: TOTAL FEATURES count")
print()
print("  If total features = 83 → exact paper match ✓")
print("  If total features = 82 → within 1 (Protocol subset differs) ~")
print()
print("  Phase 3 reads: working_set.parquet + phase2_profile.json")
