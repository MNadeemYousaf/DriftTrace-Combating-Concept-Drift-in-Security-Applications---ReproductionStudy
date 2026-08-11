"""
DriftTrace reproduction -- PHASE 3  (V10)
Contrastive autoencoder + IQR-based drift detection.

Paper: Pan et al., "DriftTrace: Combating Concept Drift in Security
Applications Through Detection and Explanation", IEEE TIFS vol. 21, 2026.

KEY CHANGES FROM PREVIOUS VERSIONS
====================================
1. OPEN-SET PROTOCOL -- 0/100 hold-out (zero-day evaluation):
   - Training set : known classes' TRAIN rows ONLY (0 drift-class rows)
   - Test set     : known classes' TEST rows + ALL drift-class rows (100%)
   The model has never seen a single drift-class sample during training.
   This is the true zero-day evaluation consistent with the paper.

2. CONTRASTIVE LOSS -- regular Euclidean distance (not squared):
   DriftTrace Equation 2 uses ||z_i - z_j||₂ (L2 norm, subscript 2).
   The paper says "Euclidean distance" explicitly. Squared distance (d²)
   was used in V1-V8 by mistake. Correct formula:
     L = y_ij · d  +  (1-y_ij) · max(0, m - d)
   where d = ||z_i - z_j||₂ (regular Euclidean distance).

3. NUMERICALLY STABLE IMPLEMENTATION:
   - Epsilon INSIDE sqrt: (d² + ε).sqrt()  prevents NaN gradient at d=0
   - Mean-based loss (not sum): bounded regardless of pair count
   - Gradient clipping: max_norm=1.0 safety net

4. SIMILAR-RATIO=0.25 CORRECT IMPLEMENTATION:
   Subsample positive pairs (not negatives) to achieve 25% positive ratio.
   With 3 balanced classes: natural positive ratio = 33%.
   To achieve 25%: n_pos_used = n_neg × (0.25/0.75) = n_neg/3.

All other hyperparameters: paper's exact values (m=10, λ=0.1, lr=0.001,
epochs=300, batch=256, p=1.5, balanced=True).

Run cell by cell in Spyder (Ctrl+Enter on each # %% block).
Prerequisite: run phase2_v10.py first.
"""

# %% ========================= IMPORTS =========================

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% ========================= CONFIG ==========================
# EDIT ONLY THIS LINE.

OUT_DIR = r"C:\Users\StudentMuhammadNadee\Documents\Datasets\drifttrace_out"

# ── Paper hyperparameters (Section IV-A) ──────────────────────────────
MARGIN        = 10.0   # m   (Eq. 2)
LAMBDA        = 0.1    # λ   (Eq. 3)
LR            = 0.001  # Adam learning rate
EPOCHS        = 300
BATCH_SIZE    = 256
IQR_P         = 1.5    # p   (Algorithm 1)
SIMILAR_RATIO = 0.25   # positive-pair ratio (CADE --similar-ratio 0.25)
BALANCED      = True   # class-balanced mini-batch (paper explicit)

DEVICE = torch.device("cpu")


# %% ========================= HELPERS =========================

def banner(t, ch="=", w=78):
    print(f"\n{ch*w}\n{t}\n{ch*w}")


# %% ==================== STEP 0: LOAD =========================

banner("STEP 0  |  LOAD PHASE 2 OUTPUT")

wp = os.path.join(OUT_DIR, "working_set.parquet")
pp = os.path.join(OUT_DIR, "phase2_profile.json")

for path in (wp, pp):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\nRun phase2_v10.py first.")

df = pd.read_parquet(wp)
with open(pp) as fh:
    profile = json.load(fh)

FEATURE_COLS = profile["feature_cols"]
CLASS_NAMES  = sorted(profile["class_names"])
INPUT_DIM    = len(FEATURE_COLS)
LATENT_DIM   = len(CLASS_NAMES) - 1   # 3

print(f"  Rows         : {len(df):,}")
print(f"  Features     : {INPUT_DIM}  (paper: 83)")
print(f"  Classes      : {CLASS_NAMES}")
print(f"  Latent dim   : {LATENT_DIM}")
print(f"  Split method : {profile.get('split_method','')}")
print(f"  Hold-out     : {profile.get('holdout_method','')}")

print("\n  Base split distribution (before hold-out selection):")
tab = pd.crosstab(df["split"], df["__class__"])
print(tab.to_string())

print(f"\n  Hyperparameters:")
print(f"    m={MARGIN}  λ={LAMBDA}  lr={LR}  epochs={EPOCHS}"
      f"  batch={BATCH_SIZE}  p={IQR_P}")
print(f"    similar_ratio={SIMILAR_RATIO}  balanced={BALANCED}")
print(f"\n  Encoder: {INPUT_DIM} -> 64 -> 32 -> 16 -> {LATENT_DIM}")
print(f"  Paper  : 83 -> 64 -> 32 -> 16 -> 3")


# %% ==================== STEP 1: MODEL ========================

banner("STEP 1  |  CONTRASTIVE AUTOENCODER")

print(f"  Distance: REGULAR L2 norm (Euclidean distance)")
print(f"            ||z_i - z_j||₂  (paper Eq. 2, subscript 2 = L2 norm)")
print(f"  NOT squared distance (d² was wrong in V1-V8)")


class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(64, 32, 16)):
        super().__init__()
        enc_dims = [input_dim] + list(hidden) + [latent_dim]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)

        dec_dims = [latent_dim] + list(reversed(hidden)) + [input_dim]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def reconstruction_loss(x, x_hat):
    """Eq. 1: MSE(x, x_hat)"""
    return F.mse_loss(x_hat, x)


def contrastive_loss(z, labels, margin=MARGIN, similar_ratio=SIMILAR_RATIO):
    """
    Eq. 2: L = y_ij · ||z_i-z_j||₂ + (1-y_ij) · max(0, m - ||z_i-z_j||₂)

    Uses REGULAR Euclidean distance (L2 norm), not squared.
    Paper notation: ||z_i-z_j||₂ (subscript 2 = L2 norm, not superscript 2).

    Numerical stability:
      - Epsilon inside sqrt: (d² + ε).sqrt() prevents NaN gradient at d=0
      - Mean-based loss: bounded regardless of number of pairs

    similar_ratio=0.25 (CADE --similar-ratio 0.25):
      Subsample POSITIVE pairs to achieve 25% positive ratio.
      Natural positive ratio with 3 balanced classes = 33%.
      n_pos_used = n_neg × (0.25/0.75) = n_neg/3  ← correct implementation
    """
    n = z.size(0)
    if n < 2:
        return torch.tensor(0.0, device=z.device)

    rows, cols = torch.triu_indices(n, n, offset=1, device=z.device)
    diff = z[rows] - z[cols]

    # Regular L2 distance with epsilon inside sqrt (numerically stable)
    d    = (diff.pow(2).sum(dim=1) + 1e-8).sqrt()
    same = (labels[rows] == labels[cols])

    pos_idx = same.nonzero(as_tuple=True)[0]
    neg_idx = (~same).nonzero(as_tuple=True)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=z.device)

    # Subsample positive pairs to achieve similar_ratio
    # n_pos_used / n_neg_used = similar_ratio / (1 - similar_ratio) = 1/3
    n_pos_target = max(1, int(n_neg * similar_ratio / (1.0 - similar_ratio)))
    n_pos_used   = min(n_pos, n_pos_target)

    perm_pos    = torch.randperm(n_pos, device=z.device)[:n_pos_used]
    sampled_pos = pos_idx[perm_pos]

    # Mean-based loss (not sum) to keep gradients bounded
    l_pos_mean = d[sampled_pos].mean()
    l_neg_mean = torch.clamp(margin - d[neg_idx], min=0.0).mean()

    # Weighted combination preserving the ratio
    total = n_pos_used + n_neg
    loss  = (l_pos_mean * n_pos_used + l_neg_mean * n_neg) / total
    return loss


# %% ==================== STEP 2: TRAINING =====================

banner("STEP 2  |  TRAINING FUNCTION")


def make_loader(X, y, batch_size, balanced=True):
    dataset = TensorDataset(X, y)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, drop_last=False)
    counts  = torch.bincount(y)
    weights = 1.0 / counts[y].float()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    return DataLoader(dataset, batch_size=batch_size,
                     sampler=sampler, drop_last=False)


def train_model(X_tr, y_tr, model, optimizer,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                lam=LAMBDA, margin=MARGIN, similar_ratio=SIMILAR_RATIO,
                balanced=BALANCED, print_every=50):
    loader  = make_loader(X_tr, y_tr, batch_size, balanced=balanced)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_tot = ep_rec = ep_con = 0.0
        n_bat  = 0

        for xb, yb in loader:
            xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
            z, x_hat = model(xb)
            l_rec    = reconstruction_loss(xb, x_hat)
            l_con    = contrastive_loss(z, yb, margin=margin,
                                        similar_ratio=similar_ratio)
            loss     = l_rec + lam * l_con

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents explosion at init with L2 distance
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_tot += loss.item()
            ep_rec += l_rec.item()
            ep_con += l_con.item()
            n_bat  += 1

        avg = ep_tot / n_bat
        history.append((epoch, avg, ep_rec/n_bat, ep_con/n_bat))
        if epoch % print_every == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"total={avg:.5f}  "
                  f"recon={ep_rec/n_bat:.5f}  "
                  f"contras={ep_con/n_bat:.5f}")

    return history


print("  Training function ready.")
print(f"  Distance: regular L2 norm (not squared)")
print(f"  Loss: mean-based (bounded, stable)")
print(f"  Gradient clipping: max_norm=1.0")


# %% ==================== STEP 3: IQR DETECTION ================

banner("STEP 3  |  IQR DETECTION (Algorithm 1)")


def compute_centroids_iqr(model, X_tr, y_tr, class_ids, p=IQR_P):
    """Algorithm 1: centroids and IQR bounds from training data."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(X_tr.to(DEVICE)).cpu().numpy()
    y_np   = y_tr.numpy()
    params = {}
    for cls_id in class_ids:
        mask = y_np == cls_id
        Zc   = Z[mask]
        c    = Zc.mean(axis=0)
        d    = np.linalg.norm(Zc - c, axis=1)   # regular L2 distance
        Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
        IQR  = Q3 - Q1
        params[cls_id] = {
            "centroid": c,
            "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "upper": Q3 + p * IQR,
            "lower": Q1 - p * IQR,
        }
        print(f"    class {cls_id}  n={mask.sum():,}  "
              f"IQR={IQR:.4f}  upper={Q3 + p*IQR:.4f}")
    return params


def detect_drift(model, X_te, params, p=IQR_P):
    """Algorithm 1: drift scoring for all test samples."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(X_te.to(DEVICE)).cpu().numpy()
    cls_ids = sorted(params.keys())
    N, K    = Z.shape[0], len(cls_ids)
    dists   = np.zeros((N, K))
    A_mat   = np.zeros((N, K))
    for j, cid in enumerate(cls_ids):
        pr    = params[cid]
        denom = p * pr["IQR"] if pr["IQR"] > 1e-10 else 1e-6
        d     = np.linalg.norm(Z - pr["centroid"], axis=1)
        dists[:, j] = d
        A_mat[:, j] = np.maximum(0,
            np.maximum((d - pr["upper"]) / denom,
                       (pr["lower"] - d) / denom))
    return A_mat.min(axis=1), np.array([cls_ids[i]
           for i in dists.argmin(axis=1)]), dists


print("  Detection functions ready.")


# %% ==================== STEP 4: METRICS ======================

banner("STEP 4  |  METRICS (Section IV-B)")


def compute_metrics(drift_scores, y_true_is_drift):
    n_drift = int(y_true_is_drift.sum())
    if n_drift == 0:
        return {"precision": 0., "recall": 0., "f1": 0.,
                "inspection_effort": 0., "n_drift_test": 0}
    order    = np.argsort(-drift_scores)
    y_sorted = y_true_is_drift[order]
    best_f1 = best_p = best_r = 0.0
    best_k  = 0
    tp      = 0
    for k in range(1, len(drift_scores) + 1):
        if y_sorted[k - 1]:
            tp += 1
        p_k = tp / k
        r_k = tp / n_drift
        f_k = (2 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
        if f_k > best_f1:
            best_f1, best_p, best_r, best_k = f_k, p_k, r_k, k
    return {
        "precision":         round(best_p, 4),
        "recall":            round(best_r, 4),
        "f1":                round(best_f1, 4),
        "inspection_effort": round(best_k / n_drift, 4),
        "n_drift_test":      n_drift,
        "best_k":            best_k,
    }


print("  Metrics function ready.")


# %% ==================== STEP 5: EXPERIMENT ===================

banner("STEP 5  |  OPEN-SET EXPERIMENT  (4 hold-out scenarios)")

print("  OPEN-SET PROTOCOL:")
print("  For each hold-out class:")
print("    Train: known classes' TRAIN rows only  (0 hold-out rows in train)")
print("    Test : known classes' TEST rows + ALL hold-out rows (0/100)")
print()

X_all     = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
y_str_all = df["__class__"].values
split_all = df["split"].values

cls_to_int = {c: i for i, c in enumerate(CLASS_NAMES)}
int_to_cls = {i: c for c, i in cls_to_int.items()}
y_int_all  = torch.tensor([cls_to_int[c] for c in y_str_all], dtype=torch.long)

train_mask = split_all == "train"
test_mask  = split_all == "test"

all_results = {}

for holdout in CLASS_NAMES:

    banner(f"  HOLD-OUT: {holdout}  (zero-day -- 0 training rows)", ch="-", w=70)

    hid       = cls_to_int[holdout]
    known_ids = [i for i in cls_to_int.values() if i != hid]

    # ── A. Build train / test tensors ─────────────────────────────────
    # Train: known classes' TRAIN rows (0/100: zero hold-out rows)
    tr_mask = train_mask & (y_str_all != holdout)
    # Test: known classes' TEST rows + ALL hold-out rows
    te_known_mask = test_mask  & (y_str_all != holdout)
    te_drift_mask = (y_str_all == holdout)           # ALL rows regardless of split
    te_mask       = te_known_mask | te_drift_mask

    X_tr    = X_all[tr_mask]
    y_tr    = y_int_all[tr_mask]
    X_te    = X_all[te_mask]
    y_te    = y_str_all[te_mask]

    n_drift = int((y_te == holdout).sum())
    n_known_te = int((y_te != holdout).sum())

    tr_counts = {int_to_cls[i]: int((y_tr == i).sum()) for i in known_ids}
    total_tr  = sum(tr_counts.values())

    print(f"\n  Known classes   : {[int_to_cls[i] for i in known_ids]}")
    print(f"  Train (80% of known): "
          + "  ".join(f"{c}={n:,}({n/total_tr*100:.0f}%)"
                      for c, n in tr_counts.items()))
    print(f"  Train total     : {len(X_tr):,}")
    print(f"  Test known (20%): {n_known_te:,}")
    print(f"  Test drift (ALL): {n_drift:,}  (100% of {holdout})")
    print(f"  Test total      : {len(X_te):,}")

    if n_drift == 0:
        print("  SKIP -- no drift samples")
        all_results[holdout] = {"skipped": True}
        continue

    # ── B. Train ──────────────────────────────────────────────────────
    print(f"\n  Training {EPOCHS} epochs  "
          f"(m={MARGIN}, λ={LAMBDA}, lr={LR}, "
          f"batch={BATCH_SIZE}, similar_ratio={SIMILAR_RATIO}, "
          f"balanced={BALANCED})...")

    model = ContrastiveAutoencoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    t0      = time.time()
    history = train_model(X_tr, y_tr, model, opt)
    t_sec   = time.time() - t0

    final_con = history[-1][3]
    converged = final_con < 0.5   # L2 distance scale is different from d²
    print(f"  Training time   : {t_sec:.0f} s")
    print(f"  contras@300     : {final_con:.5f}  "
          f"({'converged' if converged else 'STUCK'})")

    # ── C. Centroids + IQR ────────────────────────────────────────────
    print(f"\n  Class centroids and IQR bounds (p={IQR_P}):")
    det_params = compute_centroids_iqr(model, X_tr, y_tr, known_ids)

    # ── D. Detect ─────────────────────────────────────────────────────
    print(f"\n  Running Algorithm 1 on {len(X_te):,} test samples...")
    drift_scores, pred_cls, dists = detect_drift(model, X_te, det_params)
    y_true_drift = (y_te == holdout).astype(int)

    # ── E. Metrics ────────────────────────────────────────────────────
    m = compute_metrics(drift_scores, y_true_drift)
    print(f"\n  Results:")
    print(f"    Precision        : {m['precision']:.4f}")
    print(f"    Recall           : {m['recall']:.4f}")
    print(f"    F1               : {m['f1']:.4f}")
    print(f"    Inspection Effort: {m['inspection_effort']:.4f}"
          f"  (reviewed {m['best_k']:,} to reach peak F1)")
    print(f"    Drift / Test     : {m['n_drift_test']:,} / {len(X_te):,}"
          f"  ({m['n_drift_test']/len(X_te)*100:.1f}%)")

    # ── F. Save ───────────────────────────────────────────────────────
    mp = os.path.join(OUT_DIR, f"model_v10_{holdout.replace('-','_')}.pt")
    torch.save({
        "model_state":      model.state_dict(),
        "detection_params": det_params,
        "history":          history,
        "holdout_cls":      holdout,
        "known_ids":        known_ids,
        "cls_to_int":       cls_to_int,
        "feature_cols":     FEATURE_COLS,
        "input_dim":        INPUT_DIM,
        "latent_dim":       LATENT_DIM,
    }, mp)

    all_results[holdout] = {
        "skipped":           False,
        "n_train":           int(len(X_tr)),
        "n_test_known":      n_known_te,
        "n_test_drift":      n_drift,
        "n_test_total":      int(len(X_te)),
        "drift_pct":         round(n_drift/len(X_te)*100, 1),
        "precision":         m["precision"],
        "recall":            m["recall"],
        "f1":                m["f1"],
        "inspection_effort": m["inspection_effort"],
        "train_time_s":      round(t_sec, 1),
        "contras_final":     round(final_con, 5),
        "converged":         converged,
    }


# %% ==================== STEP 6: RESULTS TABLE ================

banner("STEP 6  |  RESULTS TABLE  (paper Table V format)")

print(f"\n  Config v10: m={MARGIN} | p={IQR_P} | "
      f"balanced={BALANCED} | input_dim={INPUT_DIM}")
print(f"  Hold-out: 0/100 (all drift-class rows in test, zero in train)")
print(f"  Distance: regular L2 norm (Euclidean, not squared)")

print(f"\n  {'Hold-out':<16} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'Insp.Effort':>13} {'Drift%':>7} {'Contras':>10}")
print("  " + "─" * 80)

f1s = []
for cls in CLASS_NAMES:
    r = all_results.get(cls, {})
    if r.get("skipped"):
        print(f"  {cls:<16}   SKIPPED")
        continue
    f1s.append(r["f1"])
    print(f"  {cls:<16} {r['precision']:>10.4f} {r['recall']:>8.4f} "
          f"{r['f1']:>8.4f} {r['inspection_effort']:>13.4f} "
          f"{r['drift_pct']:>6.1f}% {r['contras_final']:>10.5f}")

if f1s:
    avg = sum(f1s) / len(f1s)
    print("  " + "─" * 80)
    print(f"  {'Average':<16} {'':>27} {avg:>8.4f}"
          f"              (paper: 0.97 ± 0.02)")

print(f"""
  Paper Table V (IDS2018):
    DriftTrace     P=0.98±0.01  R=0.97±0.03  F1=0.97±0.02  IE=0.98±0.02
    Vanilla AE     P=0.69±0.08  R=0.96±0.04  F1=0.84±0.04  IE=1.34±0.11
    TRANSCENDENT   P=0.61±0.11  R=0.89±0.17  F1=0.75±0.04  IE=1.65±0.26
""")


# %% ==================== STEP 7: SAVE =========================

banner("STEP 7  |  SAVING")

rp = os.path.join(OUT_DIR, "phase3_v10_results.json")
with open(rp, "w") as fh:
    json.dump(all_results, fh, indent=2, default=str)
print(f"  phase3_v10_results.json -> {rp}")

banner("PHASE 3 v10 COMPLETE")
n_done  = sum(1 for r in all_results.values() if not r.get("skipped"))
n_conv  = sum(1 for r in all_results.values()
              if not r.get("skipped") and r.get("converged"))
print(f"  Runs completed : {n_done}  |  Converged : {n_conv}/{n_done}")
if f1s:
    print(f"  Average F1     : {avg:.4f}  (paper: 0.97)")
print()
print("  Share the Step 6 results table.")
print("  Key diagnostic: IQR bounds after Benign hold-out training.")
print("  With regular L2 distance, expect IQR >> 0.002 (previous).")
