
#!pip -q install timm==1.0.9 "scipy>=1.14,<1.15" --upgrade

# 1) Imports & setup
import os, sys, math, time, json, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partial JPEG headers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from scipy.stats import spearmanr
import timm

# OpenCV optional
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Drive
from google.colab import drive
drive.mount('/content/drive')

# Repro
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# 2) Config
@dataclass
class CFG:
    # Paths
    WORK_DIR   : str = "/content/drive/MyDrive/..."
    SPLIT_DIR  : str = f"{WORK_DIR}/splits"
    CKPT_DIR   : str = "/content/drive/MyDrive/..."
    LOG_DIR    : str = "/content/drive/MyDrive/..."

    TRAIN_CSV  : str = "train.csv"
    VAL_CSV    : str = "val.csv"
    TEST_CSV   : str = "test.csv"

    # Columns
    COL_IMG    : str = "frame_path"
    COL_TARGET : str = "pmos"
    COL_VIDEO  : str = "video_id"

    # Data/Transforms
    img_size   : int = 300
    mean       : Tuple[float,float,float] = (0.485, 0.456, 0.406)
    std        : Tuple[float,float,float]  = (0.229, 0.224, 0.225)

    # Training
    epochs     : int = 50
    batch_size : int = 32
    lr         : float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_accum_steps: int = 1
    early_stop_patience: int = 8
    mixed_precision: bool = True

    # Loader strategy (no-copy Drive safe):
    #   "solo"   => num_workers=0 (most reliable, slowest but no timeouts)
    #   "mild"   => num_workers=1
    #   "parallel" => num_workers=2  (may timeout on Drive; use only if stable)
    loader_mode: str = "solo"

    # Model
    model_name : str = "efficientnet_b3"
    pretrained : bool = True
    drop_rate  : float = 0.2
    drop_path_rate: float = 0.2

    # EMA (set 0.999 to enable)
    ema_decay  : float = 0.0

    # Device
    device     : str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = CFG()
os.makedirs(CFG.CKPT_DIR, exist_ok=True)
os.makedirs(CFG.LOG_DIR, exist_ok=True)
print(f"[Device] {CFG.device} | loader_mode={CFG.loader_mode} | batch={CFG.batch_size} | img={CFG.img_size}")

# 3) Drive-safe image open
def safe_open_image(path: str, retries: int = 10, delay: float = 0.4):
    """
    Robust open for Google Drive:
      - Retry PIL a few times with tiny sleeps (Drive often transiently fails)
      - Fallback to OpenCV imdecode if available
    """
    last_err = None
    for _ in range(retries):
        try:
            if not os.path.exists(path):
                last_err = FileNotFoundError(path); time.sleep(delay); continue
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception as e:
            last_err = e; time.sleep(delay)

    if _HAS_CV2:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None: raise OSError("cv2.imdecode returned None")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return Image.fromarray(im)
        except Exception as e:
            last_err = e

    raise last_err

# 4) Dataset
class FrameRegDataset(Dataset):
    """
    CSV must have: frame_path, pmos, video_id
    """
    def __init__(self, csv_path: str, transforms=None,
                 col_img="frame_path", col_target="pmos", col_video="video_id"):
        self.df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in self.df.columns}
        self.c_img = cols[col_img.lower()]
        self.c_tgt = cols[col_target.lower()]
        self.c_vid = cols[col_video.lower()]
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        path = str(row[self.c_img])
        y    = float(row[self.c_tgt])
        vid  = row[self.c_vid]

        im = safe_open_image(path, retries=10, delay=0.4)
        if self.transforms:
            im = self.transforms(im)

        return im, torch.tensor([y], dtype=torch.float32), str(vid)

# 5) Transforms
def build_transforms(img_size, mean, std):
    train_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf

train_tf, eval_tf = build_transforms(CFG.img_size, CFG.mean, CFG.std)

# 6) DataLoaders — main-process mode to avoid Drive timeouts
def _resolve_workers(mode: str):
    if mode == "solo": return 0
    if mode == "mild": return 1
    if mode == "parallel": return 2
    return 0

def make_loader(csv_name: str, transforms, shuffle: bool, batch_size: int):
    use_cuda   = (CFG.device == "cuda" and torch.cuda.is_available())
    workers    = _resolve_workers(CFG.loader_mode)
    pin_memory = True if (use_cuda and workers>=0) else False  # harmless on cpu
    # timeout=0 disables timeout (critical for Drive)
    dl = DataLoader(
        FrameRegDataset(
            csv_path=os.path.join(CFG.SPLIT_DIR, csv_name),
            transforms=transforms,
            col_img=CFG.COL_IMG, col_target=CFG.COL_TARGET, col_video=CFG.COL_VIDEO
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=(2 if workers>0 else None),
        timeout=0,
        drop_last=False,
    )
    return dl

train_dl = make_loader(CFG.TRAIN_CSV, train_tf, True,  CFG.batch_size)
val_dl   = make_loader(CFG.VAL_CSV,   eval_tf, False, CFG.batch_size)
test_dl  = make_loader(CFG.TEST_CSV,  eval_tf, False, CFG.batch_size)
print(f"[Data] train_batches={len(train_dl)} | val_batches={len(val_dl)} | test_batches={len(test_dl)} | workers={_resolve_workers(CFG.loader_mode)}")

# 7) Model, loss, optimizer, AMP, EMA
def build_model():
    return timm.create_model(
        CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=1,
        drop_rate=CFG.drop_rate,
        drop_path_rate=CFG.drop_path_rate,
        in_chans=3
    )

model = build_model().to(CFG.device)
criterion = nn.SmoothL1Loss(beta=0.5).to(CFG.device)
mse_crit  = nn.MSELoss().to(CFG.device)
optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

def cosine_lr(epoch, base_lr, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

scaler = torch.amp.GradScaler('cuda', enabled=(CFG.mixed_precision and CFG.device=='cuda'))

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = build_model().to(CFG.device)
        self.ema.load_state_dict(model.state_dict())
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)

ema = ModelEMA(model, CFG.ema_decay) if CFG.ema_decay > 0 else None

# 8) Metrics
def plcc(y_true, y_pred):
    if len(y_true) < 2: return 0.0
    return float(np.nan_to_num(np.corrcoef(y_true, y_pred)[0,1]))
def srcc(y_true, y_pred):
    if len(y_true) < 2: return 0.0
    s, _ = spearmanr(y_true, y_pred)
    return float(np.nan_to_num(s))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def mae (y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) + 1e-12
    return 1.0 - ss_res/ss_tot

def aggregate_by_video(vids: List[str], preds: np.ndarray, tgts: np.ndarray):
    bucket = {}
    for v, p, t in zip(vids, preds, tgts):
        bucket.setdefault(v, {"p": [], "t": []})
        bucket[v]["p"].append(p); bucket[v]["t"].append(t)
    vpred, vtrue = [], []
    for v in bucket:
        vpred.append(np.mean(bucket[v]["p"]))
        vtrue.append(np.mean(bucket[v]["t"]))
    return np.array(vpred), np.array(vtrue)

# 9) Train / Eval
def train_one_epoch(epoch: int) -> Dict[str, float]:
    model.train(); n=0; loss_sum=0.0; mse_sum=0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, targets, vids) in enumerate(train_dl):
        imgs    = imgs.to(CFG.device, non_blocking=True)
        targets = targets.to(CFG.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(CFG.mixed_precision and CFG.device=='cuda')):
            preds = model(imgs)
            loss  = criterion(preds, targets)
            mse   = mse_crit(preds, targets)

        scaler.scale(loss / CFG.grad_accum_steps).backward()
        if (step + 1) % CFG.grad_accum_steps == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema: ema.update(model)

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        mse_sum  += mse.item()  * bs
        n += bs

    return {"loss": loss_sum / max(1, n), "mse": mse_sum / max(1, n)}

@torch.no_grad()
def evaluate(dloader: DataLoader, use_ema=False) -> Dict[str, float]:
    mdl = (ema.ema if (ema and use_ema) else model)
    mdl.eval()
    preds_all, tgts_all, vids_all = [], [], []

    for imgs, targets, vids in dloader:
        imgs = imgs.to(CFG.device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(CFG.mixed_precision and CFG.device=='cuda')):
            out = mdl(imgs).squeeze(1).detach().cpu().numpy()
        tg  = targets.squeeze(1).detach().cpu().numpy()
        preds_all.append(out); tgts_all.append(tg); vids_all.extend(vids)

    y_pred = np.concatenate(preds_all); y_true = np.concatenate(tgts_all)
    fr = dict(
        frame_PLCC=plcc(y_true, y_pred),
        frame_SRCC=srcc(y_true, y_pred),
        frame_RMSE=rmse(y_true, y_pred),
        frame_MAE =mae (y_true, y_pred),
        frame_R2  =r2_score(y_true, y_pred)
    )
    v_pred, v_true = aggregate_by_video(vids_all, y_pred, y_true)
    vd = dict(
        video_PLCC=plcc(v_true, v_pred),
        video_SRCC=srcc(v_true, v_pred),
        video_RMSE=rmse(v_true, v_pred),
        video_MAE =mae (v_true, v_pred),
        video_R2  =r2_score(v_true, v_pred)
    )
    return {**fr, **vd}

# 10) Orchestrator
def save_ckpt(epoch, model, optimizer, best_metric, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": CFG.__dict__,
    }, path)

def train_loop():
    best_rmse = float("inf")
    best_path = os.path.join(CFG.CKPT_DIR, "best_rmse.pt")
    last_path = os.path.join(CFG.CKPT_DIR, "last.pt")
    no_improve = 0
    history = []

    for epoch in range(CFG.epochs):
        lr_now = (CFG.lr if CFG.warmup_epochs==0 else
                  (CFG.lr * (epoch+1)/CFG.warmup_epochs if epoch < CFG.warmup_epochs
                   else CFG.lr * 0.5 * (1 + math.cos(math.pi * (epoch-CFG.warmup_epochs)/max(1,(CFG.epochs-CFG.warmup_epochs))))))
        for g in optimizer.param_groups: g["lr"] = lr_now

        t0 = time.time()
        tr = train_one_epoch(epoch)
        va = evaluate(val_dl, use_ema=False)

        metric = va["video_RMSE"]
        save_ckpt(epoch, model, optimizer, best_rmse, last_path)
        if metric < best_rmse:
            best_rmse = metric; no_improve = 0
            save_ckpt(epoch, model, optimizer, best_rmse, best_path)
        else:
            no_improve += 1

        log_row = {
            "epoch": epoch, "lr": lr_now,
            **{f"train_{k}": v for k,v in tr.items()},
            **{f"val_{k}": v for k,v in va.items()},
            "time_sec": round(time.time()-t0, 2)
        }
        history.append(log_row)
        print(json.dumps(log_row, indent=2))

        if no_improve >= CFG.early_stop_patience:
            print(f"Early stop at epoch {epoch}. Best video_RMSE={best_rmse:.4f}")
            break

    # logs
    hist_path = os.path.join(CFG.LOG_DIR, "train_log.jsonl")
    with open(hist_path, "w") as f:
        for row in history: f.write(json.dumps(row) + "\n")
    print("[Logs]", hist_path)
    print("[Best]", best_path)

    # final test
    ckpt = torch.load(best_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    te = evaluate(test_dl, use_ema=False)
    print("\n=== TEST (best by val video_RMSE) ===")
    for k, v in te.items():
        print(f"{k}: {v:.6f}")

# 11) Train
train_loop()
