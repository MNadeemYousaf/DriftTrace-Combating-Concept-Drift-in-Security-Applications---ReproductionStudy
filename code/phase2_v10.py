"""
DriftTrace reproduction -- PHASE 2  (V10)
Load, engineer features, apply within-class temporal 80/20 split.

Paper: Pan et al., "DriftTrace: Combating Concept Drift in Security
Applications Through Detection and Explanation", IEEE TIFS vol. 21, 2026.

OPEN-SET PROTOCOL (implemented in Phase 3):
  For each hold-out class:
    - Training set : known classes' TRAIN rows only (0 drift-class rows)
    - Test set     : known classes' TEST rows + ALL drift-class rows (100%)
  This is the true zero-day evaluation -- the model has never seen
  a single sample from the drift class during training.

SPLIT METHOD (this phase):
  Within-class temporal 80/20:
    For each class independently:
      1. Sort rows by Timestamp (ascending)
      2. First 80% -> train   (earlier capture, what the model learns from)
      3. Last  20% -> test    (later capture, used for evaluation)
  This satisfies both paper requirements:
    - "sorted based on timestamps" (temporal ordering per class)
    - "stratified sampling" (each class contributes 80/20 proportionally)

FEATURES (83, matching CADE Appendix 1):
  76 numerical (MinMaxScaler [0,1])
  + 1 Timestamp_ms (converted to milliseconds, MinMaxScaled)
  + 3 Dst Port OHE (high >10k / medium 1k-10k / low <1k)
  + 3 Protocol OHE (unique training values)
  = 83 total

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


# %% ========================= CONFIG ==========================
# EDIT ONLY THESE TWO LINES.

DATA_DIR = r"C:\Users\StudentMuhammadNadee\Documents\Datasets\CICIDS2018_AWS"
OUT_DIR  = r"C:\Users\StudentMuhammadNadee\Documents\Datasets\drifttrace_out"

TRAIN_FRAC   = 0.80
CHUNK_SIZE   = 300_000
PORT_HIGH    = 10_000
PORT_MED     =  1_000


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

# Only true identifiers excluded — everything else is a feature
META_EXACT = {"flow id", "source ip", "src ip", "destination ip",
              "dst ip", "timestamp", "label"}
CAT_COLS   = ["Dst Port", "Protocol"]


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

banner("STEP 1  |  VALIDATE")

for fname in SOURCE_FILES:
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found:\n  {p}\nFix DATA_DIR.")
    print(f"  ✓  {fname}  ({hb(os.path.getsize(p))})")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"\n  OUT_DIR : {OUT_DIR}")


# %% ==================== STEP 2: SCHEMA =====================

banner("STEP 2  |  SCHEMA + FEATURE PLAN")

first_path   = os.path.join(DATA_DIR, list(SOURCE_FILES.keys())[0])
probe        = clean_cols(pd.read_csv(first_path, nrows=5, low_memory=False))
label_col    = next(c for c in probe.columns if c.strip().lower() == "label")
ts_col       = next(c for c in probe.columns if c.strip().lower() == "timestamp")
raw_feat     = [c for c in probe.columns if not is_meta(c)]
num_cols     = [c for c in raw_feat if c not in CAT_COLS]

print(f"  Label col   : '{label_col}'")
print(f"  Timestamp   : '{ts_col}'")
print(f"  Raw features: {len(raw_feat)}")
print(f"    Numerical : {len(num_cols)}")
print(f"    Categorical (OHE): {CAT_COLS}")
print(f"\n  Feature plan:")
print(f"    {len(num_cols)} numerical + 1 Timestamp_ms = {len(num_cols)+1} numerical total")
print(f"    + 3 Dst Port OHE + 3 Protocol OHE ≈ 83 total")


# %% ==================== STEP 3: LOAD ========================

banner("STEP 3  |  CHUNKED LOAD")

all_parts = []
t0 = time.time()

for fname, cfg in SOURCE_FILES.items():
    path        = os.path.join(DATA_DIR, fname)
    benign_cap  = cfg["benign_cap"]
    atk_caps    = cfg["attack_caps"]
    all_caps    = {"Benign": benign_cap,
                   **{LABEL_MAP[k]: v for k, v in atk_caps.items()}}
    got         = {cls: 0 for cls in all_caps}
    wanted      = {"Benign"} | set(atk_caps.keys())
    usecols     = raw_feat + [label_col, ts_col]

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
            chunk[ts_col], dayfirst=True, errors="coerce")
        chunk = chunk[chunk["__ts__"].notna()]

        for cls, grp in chunk.groupby("__class__", sort=False):
            if cls not in all_caps:
                continue
            room = all_caps[cls] - got[cls]
            if room <= 0:
                continue
            take = grp.head(room) if len(grp) > room else grp
            all_parts.append(
                take[raw_feat + ["__class__", "__ts__"]].copy())
            got[cls] += len(take)

        if all(got.get(c, 0) >= cap for c, cap in all_caps.items()):
            break

    for cls, n in got.items():
        print(f"    {cls:<16} {n:,}")

print(f"\n  Load time : {time.time()-t0:.1f} s")


# %% ==================== STEP 4: COMBINE + VERIFY ============

banner("STEP 4  |  COMBINE + VERIFY")

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


# %% ==================== STEP 5: WITHIN-CLASS TEMPORAL SPLIT =

banner("STEP 5  |  WITHIN-CLASS TEMPORAL 80/20 SPLIT")

print("  For each class separately:")
print("    1. Sort rows by Timestamp (ascending)")
print("    2. First 80% -> train  (earlier capture)")
print("    3. Last  20% -> test   (later capture)")
print("  All 4 classes get the split -- Phase 3 handles which is hold-out.")
print()

df["split"] = "test"
for cls in df["__class__"].unique():
    idx  = df[df["__class__"] == cls].sort_values("__ts__").index
    n_tr = int(len(idx) * TRAIN_FRAC)
    df.loc[idx[:n_tr], "split"] = "train"

tab = pd.crosstab(df["split"], df["__class__"])
print("  Class distribution (all classes, before hold-out selection):")
print(tab.to_string())

print("\n  Temporal order check (train earlier than test within each class):")
for cls in PAPER_CAPS:
    sub    = df[df["__class__"] == cls]
    tr_max = sub[sub["split"] == "train"]["__ts__"].max()
    te_min = sub[sub["split"] == "test"]["__ts__"].min()
    ok     = "✓" if tr_max <= te_min else "✗"
    print(f"    {ok}  {cls:<16} train_max={tr_max}  test_min={te_min}")

print("\n  Open-set preview (0/100 hold-out -- Phase 3 logic):")
print(f"  {'Hold-out':<16} {'Train known':>13} {'Test known':>11} "
      f"{'Test drift':>11} {'Total test':>11}")
print("  " + "─" * 68)
for holdout in sorted(PAPER_CAPS.keys()):
    tr_k  = int(((df["split"]=="train") & (df["__class__"]!=holdout)).sum())
    te_k  = int(((df["split"]=="test")  & (df["__class__"]!=holdout)).sum())
    te_d  = int((df["__class__"] == holdout).sum())   # ALL drift rows
    print(f"  {holdout:<16} {tr_k:>13,} {te_k:>11,} {te_d:>11,} {te_k+te_d:>11,}")

print("\n  Note: 'Test drift' is ALL rows of the hold-out class (0/100 split).")
print("  The model sees ZERO drift-class rows during training.")


# %% ==================== STEP 6: FEATURE ENGINEERING =========

banner("STEP 6  |  FEATURE ENGINEERING (CADE Appendix 1, 83 features)")

train_mask = df["split"] == "train"

# ── Fix infinities / object types ─────────────────────────────────────
for c in raw_feat:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
medians = df[raw_feat].median()
df[raw_feat] = df[raw_feat].fillna(medians)

# ── Timestamp_ms (83rd feature, confirmed CADE Appendix 1) ────────────
df["Timestamp_ms"] = df["__ts__"].astype(np.int64) // 10**6
num_cols_all = num_cols + ["Timestamp_ms"]
print(f"  Numerical features : {len(num_cols_all)}"
      f"  ({len(num_cols)} original + 1 Timestamp_ms)")

# ── Dst Port OHE (frequency bands, fit on TRAIN only) ─────────────────
port_freq = df.loc[train_mask, "Dst Port"].value_counts()

def port_band(p):
    f = port_freq.get(p, 0)
    return "high" if f > PORT_HIGH else ("medium" if f >= PORT_MED else "low")

df["_DstPort_band"] = df["Dst Port"].apply(port_band)
for band in ("high", "medium", "low"):
    df[f"DstPort_{band}"] = (df["_DstPort_band"] == band).astype(np.float32)
df.drop(columns=["_DstPort_band"], inplace=True)
dst_ohe = ["DstPort_high", "DstPort_medium", "DstPort_low"]

# ── Protocol OHE (unique training values only) ─────────────────────────
train_protos = sorted(df.loc[train_mask, "Protocol"].dropna().unique())
proto_ohe    = []
for p in train_protos:
    col = f"Proto_{int(p)}"
    df[col] = (df["Protocol"] == p).astype(np.float32)
    proto_ohe.append(col)
print(f"  Protocol values in train : {train_protos}")
print(f"  Protocol OHE columns     : {proto_ohe}")

# ── MinMaxScaler on numerical (fit on ALL train rows) ──────────────────
print(f"\n  Fitting MinMaxScaler on {len(num_cols_all)} numerical features"
      f" (train rows only)...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df.loc[train_mask, num_cols_all].values.astype(np.float32))
df.loc[:, num_cols_all] = scaler.transform(
    df[num_cols_all].values.astype(np.float32)).astype(np.float32)

# ── Final feature list ─────────────────────────────────────────────────
final_features = num_cols_all + dst_ohe + proto_ohe

print(f"\n  ── Feature count ──────────────────────────────────────────────")
print(f"  Numerical (MinMax) : {len(num_cols_all)}  (76 + Timestamp_ms)")
print(f"  Dst Port OHE       : {len(dst_ohe)}")
print(f"  Protocol OHE       : {len(proto_ohe)}")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  TOTAL FEATURES     : {len(final_features)}")
print(f"  Paper target       : 83")
if len(final_features) == 83:
    print(f"  ✓  Exact match!")

n_known  = df["__class__"].nunique() - 1
arch_str = f"{len(final_features)} -> 64 -> 32 -> 16 -> {n_known}"
print(f"\n  Encoder architecture : {arch_str}")
print(f"  Paper architecture   : 83 -> 64 -> 32 -> 16 -> 3")


# %% ==================== STEP 7: SAVE ========================

banner("STEP 7  |  SAVING OUTPUTS")

keep = final_features + ["__class__", "__ts__", "split"]
wp   = os.path.join(OUT_DIR, "working_set.parquet")
df[keep].to_parquet(wp, index=False)
print(f"  working_set.parquet  -> {wp}  ({hb(os.path.getsize(wp))})")

sp = os.path.join(OUT_DIR, "scaler.pkl")
with open(sp, "wb") as fh:
    pickle.dump(scaler, fh)
print(f"  scaler.pkl           -> {sp}")

profile = {
    "generated":       pd.Timestamp.now().isoformat(),
    "version":         "v10",
    "split_method":    "within_class_temporal_80_20",
    "holdout_method":  "0_100_in_phase3",
    "n_rows":          int(len(df)),
    "train_frac":      TRAIN_FRAC,
    "label_col":       "__class__",
    "ts_col":          "__ts__",
    "split_col":       "split",
    "feature_cols":    final_features,
    "input_dim":       int(len(final_features)),
    "latent_dim":      int(n_known),
    "encoder_arch":    arch_str,
    "n_classes":       int(df["__class__"].nunique()),
    "class_names":     sorted(df["__class__"].unique().tolist()),
    "paper_caps":      PAPER_CAPS,
    "class_counts":    {
        s: {str(k): int(v)
            for k, v in df[df["split"]==s]["__class__"].value_counts().items()}
        for s in ("train", "test")
    },
}
pp = os.path.join(OUT_DIR, "phase2_profile.json")
with open(pp, "w") as fh:
    json.dump(profile, fh, indent=2, default=str)
print(f"  phase2_profile.json  -> {pp}")

banner("PHASE 2 v10 COMPLETE")
print(f"  {len(df):,} rows  |  {len(final_features)} features  |  {arch_str}")
print()
print("  Phase 3 v10 reads:")
print("    working_set.parquet  -- all 4 classes, within-class temporal split")
print("    phase2_profile.json  -- feature list, class names, encoder arch")
print()
print("  OPEN-SET PROTOCOL (implemented in Phase 3):")
print("  For each hold-out class: training sees ZERO drift-class rows.")
print("  Test set = known classes' test rows + ALL drift-class rows.")
