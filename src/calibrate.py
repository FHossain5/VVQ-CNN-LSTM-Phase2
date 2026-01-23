import os
import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import timm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ================= CONFIG (MATCH TRAINING) =================

class CFG:
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model_name  = "efficientnet_b3"
    img_size    = 300
    in_chans    = 3

    best_ckpt_path = "/content/drive/MyDrive/..."
    split_dir      = "/content/drive/MyDrive/...s"

    batch_size  = 32
    num_workers = 0          # main-process loading: stable for Drive
    use_tta     = True       # orig + hflip
    trimmed_p   = 0.10       # 10% trimmed mean (video-level)

print(f"[Device] {CFG.device}")
print(f"[Checkpoint] {CFG.best_ckpt_path}")

# ================= 1) LOAD SPLITS =================

train_csv = os.path.join(CFG.split_dir, "train.csv")
val_csv   = os.path.join(CFG.split_dir, "val.csv")
test_csv  = os.path.join(CFG.split_dir, "test.csv")

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# Ensure required cols
for name, df in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
    for col in ["frame_path", "pmos", "video_id"]:
        if col not in df.columns:
            raise RuntimeError(f"{name} missing required column: {col}")

print("[Data] rows:",
      "train=", len(train_df),
      "val=", len(val_df),
      "test=", len(test_df))

# Combined for any meta-analysis (optional; not strictly needed for CI)
df_all = pd.concat([
    train_df.assign(split="train"),
    val_df.assign(split="val"),
    test_df.assign(split="test"),
], ignore_index=True)

# ================= 2) ROBUST IMAGE LOADER =================

def safe_open_image(path: str, retries: int = 8, delay: float = 0.4):
    """Robust loader for Google Drive with retries & truncated tolerance."""
    last_err = None
    for _ in range(retries):
        try:
            if not os.path.exists(path):
                last_err = FileNotFoundError(path)
                time.sleep(delay)
                continue
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err

# ================= 3) DATASET & DATALOADERS =================

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

eval_tf = T.Compose([
    T.Resize(int(CFG.img_size * 1.1)),
    T.CenterCrop(CFG.img_size),
    T.ToTensor(),
    T.Normalize(mean, std),
])

class IQADatasetEval(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = safe_open_image(row["frame_path"])
        img = self.tf(img)
        mos = float(row["pmos"])
        vid = str(row["video_id"])
        return img, torch.tensor([mos], dtype=torch.float32), vid

def make_loader(df, name):
    loader = DataLoader(
        IQADatasetEval(df, eval_tf),
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    print(f"[Loader] {name}: {len(loader)} batches")
    return loader

val_loader  = make_loader(val_df,  "val")
test_loader = make_loader(test_df, "test")

# ================= 4) MODEL: LOAD TRAINED EFFNET-B3 =================

def load_effnet_b3_regressor(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model = timm.create_model(
        CFG.model_name,
        pretrained=False,
        in_chans=CFG.in_chans,
        num_classes=1,
    ).to(CFG.device)

    ckpt = torch.load(path, map_location=CFG.device)
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] Non-strict load. Check if arch matches training.")
        print("Missing keys (first 10):", missing[:10])
        print("Unexpected keys (first 10):", unexpected[:10])

    model.eval()
    return model

model = load_effnet_b3_regressor(CFG.best_ckpt_path)

def predict_logits(m, xb):
    out = m(xb)
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    return out

# ================= 5) METRICS & HELPERS =================

def rmse_torch(y, p):
    return float(torch.sqrt(torch.mean((p - y) ** 2)).item())

def mae_torch(y, p):
    return float(torch.mean(torch.abs(p - y)).item())

def r2_torch(y, p):
    y = y.cpu().numpy()
    p = p.cpu().numpy()
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def plcc_torch(y, p):
    y = y.cpu().numpy()
    p = p.cpu().numpy()
    return float(pearsonr(y, p)[0]) if len(y) > 1 else 0.0

def srcc_torch(y, p):
    y = y.cpu().numpy()
    p = p.cpu().numpy()
    return float(spearmanr(y, p)[0]) if len(y) > 1 else 0.0

def trimmed_mean(values, p=0.10):
    v = np.sort(np.asarray(values, float))
    n = len(v)
    k = int(n * p)
    if n <= 2 * k:
        return float(v.mean())
    return float(v[k:n-k].mean())

# ================= 6) EVAL (TTA + TRIMMED VIDEO) =================

@torch.no_grad()
def tta_predict_batch(m, xb):
    p1 = predict_logits(m, xb)
    xb_flip = torch.flip(xb, dims=[3])
    p2 = predict_logits(m, xb_flip)
    return (p1 + p2) / 2.0

@torch.no_grad()
def evaluate(m, loader, name, use_tta=True):
    m.eval()
    all_y, all_p, all_vids = [], [], []

    print(f"[Eval:{name}] Starting over {len(loader)} batches...")
    for i, (xb, yb, vids) in enumerate(loader):
        xb = xb.to(CFG.device, non_blocking=True)
        yb = yb.to(CFG.device, non_blocking=True)

        if use_tta:
            preds = tta_predict_batch(m, xb)
        else:
            preds = predict_logits(m, xb)

        all_y.append(yb.squeeze(1).cpu())
        all_p.append(preds.cpu())
        all_vids.extend(list(vids))

        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(f"[Eval:{name}] {i+1}/{len(loader)} batches done")

    print(f"[Eval:{name}] Done.")
    y = torch.cat(all_y)
    p = torch.cat(all_p)

    frame_metrics = {
        "PLCC": plcc_torch(y, p),
        "SRCC": srcc_torch(y, p),
        "RMSE": rmse_torch(y, p),
        "MAE":  mae_torch(y, p),
        "R2":   r2_torch(y, p),
    }

    df_pred = pd.DataFrame({
        "video_id": all_vids,
        "y_true": y.numpy(),
        "y_pred": p.numpy(),
    })

    # video-level using mean(gt) + trimmed-mean(pred)
    video_rows = []
    for vid, g in df_pred.groupby("video_id"):
        gt = g["y_true"].mean()
        pr = trimmed_mean(g["y_pred"].values, p=CFG.trimmed_p)
        video_rows.append((vid, gt, pr))
    vdf = pd.DataFrame(video_rows, columns=["video_id", "y_true", "y_pred"])

    vy = torch.tensor(vdf["y_true"].values)
    vp = torch.tensor(vdf["y_pred"].values)

    video_metrics = {
        "PLCC": plcc_torch(vy, vp),
        "SRCC": srcc_torch(vy, vp),
        "RMSE": rmse_torch(vy, vp),
        "MAE":  mae_torch(vy, vp),
        "R2":   r2_torch(vy, vp),
    }

    return frame_metrics, video_metrics, df_pred, vdf

val_frame_m, val_video_m, val_df_pred, val_vdf = evaluate(model, val_loader,  "val",  use_tta=CFG.use_tta)
test_frame_m, test_video_m, test_df_pred, test_vdf = evaluate(model, test_loader, "test", use_tta=CFG.use_tta)

print("\n===== VAL METRICS (TTA + trimmed video) =====")
print("Frame:", val_frame_m)
print("Video:", val_video_m)

print("\n===== TEST METRICS (TTA + trimmed video, UNCALIBRATED) =====")
print("Frame:", test_frame_m)
print("Video:", test_video_m)

# ================= 7) LINEAR CALIBRATION (FITTED ON VAL VIDEO-LEVEL) =================

# y_true = a * y_pred + b  (on validation video-level)
x_val = val_vdf["y_pred"].values
y_val = val_vdf["y_true"].values

a, b = np.polyfit(x_val, y_val, 1)
print(f"\n[Calibration] Fitted on VAL (video-level): y = {a:.6f} * ŷ + {b:.6f}")

# Apply to validation (for sanity) and test
val_vdf["y_pred_cal"] = a * val_vdf["y_pred"].values + b
test_vdf["y_pred_cal"] = a * test_vdf["y_pred"].values + b

def metrics_from_numpy(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    # reuse numpy versions for consistency in CI later
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return {
        "PLCC": float(np.corrcoef(y, p)[0,1]) if len(y) > 1 else 0.0,
        "SRCC": float(spearmanr(y, p)[0]) if len(y) > 1 else 0.0,
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE":  float(np.mean(np.abs(y - p))),
        "R2":   float(1.0 - ss_res / ss_tot),
    }

uncal_test = metrics_from_numpy(test_vdf["y_true"].values,
                                test_vdf["y_pred"].values)
cal_test   = metrics_from_numpy(test_vdf["y_true"].values,
                                test_vdf["y_pred_cal"].values)

print("\n===== TEST VIDEO-LEVEL METRICS: UNCALIBRATED vs CALIBRATED =====")
print("Uncalibrated:", uncal_test)
print("Calibrated  :", cal_test)

# ================= 8) 95% CONFIDENCE INTERVALS via BOOTSTRAPPING =================
# (THIS is step 5 from the client: new; not done earlier.)

def plcc_np(y, p):
    if len(y) < 2:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])

def srcc_np(y, p):
    if len(y) < 2:
        return 0.0
    return float(spearmanr(y, p)[0])

def r2_np(y, p):
    if len(y) < 2:
        return 0.0
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)

def bootstrap_ci_pair(metric_fn, y, p, n=2000, alpha=0.05, seed=42):
    """
    Paired bootstrap:
      - Resample (y_i, p_i) *together* with replacement.
      - Compute metric for each resample.
      - Return (low, high) CI bounds.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    N = len(y)
    scores = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        scores.append(metric_fn(y[idx], p[idx]))
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)

def summarize_with_ci(label, y, p):
    y = np.asarray(y)
    p = np.asarray(p)

    plcc_val = plcc_np(y, p)
    srcc_val = srcc_np(y, p)
    r2_val   = r2_np(y, p)

    plcc_ci = bootstrap_ci_pair(plcc_np, y, p)
    srcc_ci = bootstrap_ci_pair(srcc_np, y, p)
    r2_ci   = bootstrap_ci_pair(r2_np,   y, p)

    print(f"\n[{label}] Test video-level metrics with 95% CI (bootstrap, paired):")
    print(f"PLCC = {plcc_val:.3f}  (95% CI: {plcc_ci[0]:.3f} – {plcc_ci[1]:.3f})")
    print(f"SRCC = {srcc_val:.3f}  (95% CI: {srcc_ci[0]:.3f} – {srcc_ci[1]:.3f})")
    print(f"R²   = {r2_val:.3f}  (95% CI: {r2_ci[0]:.3f} – {r2_ci[1]:.3f})")

y_test = test_vdf["y_true"].values
p_test_uncal = test_vdf["y_pred"].values
p_test_cal   = test_vdf["y_pred_cal"].values

summarize_with_ci("Uncalibrated", y_test, p_test_uncal)
summarize_with_ci("Calibrated",   y_test, p_test_cal)
