"""
GeoVision AI - inference engine
================================
Used by app.py (Streamlit website), notebooks and the command line.

Tasks
  1. Land-cover classification    Hybrid ResNet50 + Swin-T          (10 EuroSAT classes)
  2. Multi-disaster segmentation  SegFormer-style, 7 classes        (flood, wildfire, landslide, ...)
  3. Flood segmentation           SegFormer-style, binary
  4. Change detection             ChangeFormer-style, before/after
  +  Grad-CAM explanation for the classifier

Command line
  python geovision_pipeline.py scene.jpg --out results/
  python geovision_pipeline.py before.png --after after.png --tiled
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

# ------------------------------------------------------------------ constants
TILE = 256            # the segmentation / change models work on 256x256 inputs
CLS_SIZE = 128        # the classifier was trained on 128x128 inputs
MAX_SIDE = 1536       # larger images are down-scaled to keep inference fast

DEFAULT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
DEFAULT_MEAN = [0.3439, 0.3800, 0.4075]
DEFAULT_STD = [0.2024, 0.1367, 0.1154]

DISASTER_NAMES = {0: "Background", 1: "Flood", 2: "Wildfire", 3: "Landslide",
                  4: "Earthquake Damage", 5: "Drought", 6: "Cyclone/Storm Damage"}
DISASTER_COLORS = {1: (30, 100, 255), 2: (255, 80, 0), 3: (150, 100, 50),
                   4: (190, 190, 190), 5: (240, 200, 60), 6: (170, 60, 200)}

CKPT_FILES = {
    "classifier": "checkpoints/hybrid_best.pt",
    "flood": "segmentation/segformer_best.pt",
    "disaster": "multi_disaster/multidisaster_segformer_best.pt",
    "change": "change_detection/changeformer_best.pt",
}
TASK_LABELS = {
    "classifier": "Land-cover classification",
    "disaster": "Multi-disaster detection",
    "flood": "Flood segmentation",
    "change": "Change detection",
}


# ---------------------------------------------------------------- architectures
class HybridCNNTransformer(nn.Module):
    """ResNet50 (local features) + Swin-T (global context) -> fusion MLP -> classes."""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_dim = 2048
        self.transformer = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.transformer_features = nn.Sequential(*list(self.transformer.children())[:-1])
        self.transformer_dim = 768
        fusion_dim = 512
        self.cnn_projection = nn.Sequential(nn.Linear(self.cnn_dim, fusion_dim), nn.ReLU(), nn.Dropout(0.5))
        self.transformer_projection = nn.Sequential(nn.Linear(self.transformer_dim, fusion_dim), nn.ReLU(), nn.Dropout(0.5))
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x):
        c = self.cnn_features(x)
        c = self.cnn_projection(c.view(c.size(0), -1))
        t = self.transformer_projection(self.transformer_features(x))
        return self.classifier(self.fusion_mlp(torch.cat([c, t], dim=1)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn)[0]
        return x + self.mlp(self.norm2(x))


# Keep the training-time class names so that checkpoints load without renaming.
SegFormerBlock = ChangeFormerBlock = _Block


class SegFormer(nn.Module):
    """Binary flood segmentation (1 output channel)."""

    def __init__(self, in_channels=3, out_channels=1, img_size=256):
        super().__init__()
        self.patch_size = 16
        n = (img_size // 16) ** 2
        self.patch_embed = nn.Conv2d(in_channels, 256, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, 256))
        self.encoder_blocks = nn.ModuleList([SegFormerBlock(256, 8) for _ in range(4)])
        self.decode1 = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))
        self.decode2 = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))
        self.head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, out_channels, 1, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        xf = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.encoder_blocks:
            xf = blk(xf)
        s = H // 16
        xs = xf.transpose(1, 2).reshape(B, 256, s, s).permute(0, 2, 3, 1).reshape(B * s * s, 256)
        xd = self.decode1(xs).reshape(B, s, s, 128).permute(0, 3, 1, 2)
        xd = F.interpolate(xd, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(xd)


class MultiDisasterSegFormer(nn.Module):
    """7-class disaster segmentation (per-pixel class logits)."""

    def __init__(self, in_channels=3, num_classes=7, img_size=256):
        super().__init__()
        self.patch_size = 16
        n = (img_size // 16) ** 2
        self.patch_embed = nn.Conv2d(in_channels, 256, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, 256))
        self.encoder_blocks = nn.ModuleList([SegFormerBlock(256, 8) for _ in range(4)])
        self.decode1 = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))
        self.head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, num_classes, 1, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        xf = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.encoder_blocks:
            xf = blk(xf)
        s = H // 16
        xs = xf.transpose(1, 2).reshape(B, 256, s, s).permute(0, 2, 3, 1).reshape(B * s * s, 256)
        xd = self.decode1(xs).reshape(B, s, s, 128).permute(0, 3, 1, 2)
        xd = F.interpolate(xd, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(xd)


class ChangeFormer(nn.Module):
    """Takes before+after stacked on the channel axis (6 channels) -> change logits."""

    def __init__(self, in_channels=3, out_channels=1, img_size=256):
        super().__init__()
        self.patch_size = 16
        n = (img_size // 16) ** 2
        self.patch_embed_A = nn.Conv2d(in_channels, 256, kernel_size=16, stride=16)
        self.patch_embed_B = nn.Conv2d(in_channels, 256, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, n * 2, 256))
        self.encoder_blocks = nn.ModuleList([ChangeFormerBlock(256, 8) for _ in range(3)])
        self.change_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, out_channels, 3, 1, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        a = self.patch_embed_A(x[:, :3]).flatten(2).transpose(1, 2)
        b = self.patch_embed_B(x[:, 3:]).flatten(2).transpose(1, 2)
        p = torch.cat([a, b], dim=1) + self.pos_embed.expand(B, -1, -1)
        for blk in self.encoder_blocks:
            p = blk(p)
        N = p.size(1) // 2
        feat = self.change_head(p[:, :N] - p[:, N:])
        g = H // 16
        out = self.decoder(feat.transpose(1, 2).reshape(-1, 64, g, g))
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# ----------------------------------------------------------------- file helpers
def find_root(explicit=None) -> Path:
    """Folder that contains checkpoints/, segmentation/, change_detection/, multi_disaster/."""
    candidates = [explicit, os.environ.get("GEOVISION_ROOT"), "/kaggle/working/GeoVision-AI",
                  Path(__file__).resolve().parent, Path.cwd()]
    for c in candidates:
        if c and any((Path(c) / rel).exists() for rel in CKPT_FILES.values()):
            return Path(c)
    return Path(explicit or os.environ.get("GEOVISION_ROOT") or Path(__file__).resolve().parent)


def ensure_checkpoints(root: Path) -> None:
    """Download missing checkpoints from the Hugging Face Hub when GEOVISION_HF_REPO is set."""
    repo = os.environ.get("GEOVISION_HF_REPO")
    if not repo:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    for rel in CKPT_FILES.values():
        target = Path(root) / rel
        if not target.exists():
            try:
                hf_hub_download(repo_id=repo, filename=Path(rel).name, local_dir=str(target.parent))
            except Exception as exc:  # noqa: BLE001
                print(f"[geovision] could not download {rel}: {exc}")


def load_pil(src) -> Image.Image:
    """Open an image (path / bytes / file-like) as 8-bit RGB. Handles 16-bit and float TIFFs."""
    img = Image.open(src)
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        a = np.array(img).astype(np.float32)
        lo, hi = np.percentile(a, (2, 98))
        a = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8))
    return img.convert("RGB")


def to_float_rgb(image, max_side=MAX_SIDE) -> np.ndarray:
    """PIL / numpy (uint8 0-255 or float 0-1) -> float32 HxWx3 in [0,1], capped to max_side."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    if arr.dtype == np.uint8 or arr.max() > 1.5:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        arr = cv2.resize(arr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


# -------------------------------------------------------------- visual helpers
def colorize(mask: np.ndarray) -> np.ndarray:
    """Multi-class mask (ids 0-6) -> RGB. Background stays black."""
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    for c, col in DISASTER_COLORS.items():
        rgb[mask == c] = col
    return rgb


def colorize_binary(mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[np.asarray(mask).astype(bool)] = color
    return rgb


def overlay(arr: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if base.shape[:2] != color_mask.shape[:2]:
        base = cv2.resize(base, (color_mask.shape[1], color_mask.shape[0]))
    out = base.copy()
    m = color_mask.any(axis=2)
    out[m] = (alpha * color_mask[m] + (1 - alpha) * base[m]).astype(np.uint8)
    return out


def heatmap_overlay(arr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def severity_label(affected_pct: float) -> str:
    if affected_pct < 1:
        return "None"
    if affected_pct < 10:
        return "Low"
    if affected_pct < 30:
        return "Moderate"
    return "Severe"


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------ synthetic scenes
# Same generators that produced the training data, so the website can offer
# "try a sample" scenes (with ground truth) even when no dataset is on disk.
def _jpeg(arr: np.ndarray, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


_PATCH_COLORS = {1: (25, 25, 112), 2: (200, 60, 20), 3: (110, 90, 70),
                 4: (130, 130, 130), 5: (180, 150, 100), 6: (90, 80, 60)}


def make_disaster_scene(kind: int, size: int = 256, seed=None):
    """kind 0 = clean scene, 1..6 = disaster class. Returns (PIL image, class-id mask)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[: size // 3] = (135, 206, 235)
    img[size // 3:] = (34, 139, 34)
    mask = np.zeros((size, size), np.uint8)
    if kind in _PATCH_COLORS:
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(size // 3, 2 * size // 3))
            x = int(rng.integers(0, size))
            s = int(rng.integers(30, 80))
            img[y:y + s, x:x + s] = _PATCH_COLORS[kind]
            mask[y:y + s, x:x + s] = kind
    return _jpeg(img), mask


def make_flood_scene(size: int = 256, seed=None):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    img[: size // 3] = (135, 206, 235)
    img[size // 3: 2 * size // 3] = (34, 139, 34)
    mask = np.zeros((size, size), bool)
    for _ in range(int(rng.integers(1, 4))):
        y = int(rng.integers(size // 3, 2 * size // 3))
        x = int(rng.integers(0, size))
        s = int(rng.integers(30, 80))
        img[y:y + s, x:x + s] = (25, 25, 112)
        mask[y:y + s, x:x + s] = True
    return _jpeg(img), mask


def make_change_pair(size: int = 256, seed=None):
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size, 3), np.uint8)
    a[: size // 3] = (135, 206, 235)
    a[size // 3:] = (100, 150, 100)
    for _ in range(3):
        y = int(rng.integers(size // 3, 2 * size // 3))
        x = int(rng.integers(0, size))
        s = int(rng.integers(30, 60))
        a[y:y + s, x:x + s] = (150, 150, 150)
    b = a.copy()
    mask = np.zeros((size, size), bool)
    for _ in range(int(rng.integers(1, 4))):
        y = int(rng.integers(size // 3, 2 * size // 3))
        x = int(rng.integers(0, size))
        s = int(rng.integers(30, 80))
        b[y:y + s, x:x + s] = (120, 120, 120)
        mask[y:y + s, x:x + s] = True
    return _jpeg(a), _jpeg(b), mask


# --------------------------------------------------------------------- pipeline
class GeoVisionPipeline:
    def __init__(self, root=None, device=None, load=("classifier", "flood", "disaster", "change")):
        self.root = find_root(root)
        ensure_checkpoints(self.root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = list(DEFAULT_CLASSES)
        self.mean = np.array(DEFAULT_MEAN, np.float32)
        self.std = np.array(DEFAULT_STD, np.float32)
        cfg_path = self.root / "configs" / "preprocessing_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                self.classes = cfg.get("classes", self.classes)
                self.mean = np.array(cfg.get("train_mean", DEFAULT_MEAN), np.float32)
                self.std = np.array(cfg.get("train_std", DEFAULT_STD), np.float32)
            except Exception:  # noqa: BLE001
                pass
        self.models, self.errors, self.ckpt_info = {}, {}, {}
        for task in load:
            self._load(task)

    # -- loading
    def _builder(self, task):
        return {
            "classifier": lambda: HybridCNNTransformer(len(self.classes), pretrained=False),
            "flood": lambda: SegFormer(3, 1, TILE),
            "disaster": lambda: MultiDisasterSegFormer(3, len(DISASTER_NAMES), TILE),
            "change": lambda: ChangeFormer(3, 1, TILE),
        }[task]

    def _load(self, task):
        path = self.root / CKPT_FILES[task]
        if not path.exists():
            self.errors[task] = f"checkpoint not found: {CKPT_FILES[task]}"
            return
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = self._builder(task)()
            model.load_state_dict(ckpt["model_state_dict"])
            self.models[task] = model.to(self.device).eval()
            if task == "classifier" and ckpt.get("classes"):
                self.classes = list(ckpt["classes"])
            self.ckpt_info[task] = {
                "file": CKPT_FILES[task],
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "epoch": ckpt.get("epoch"),
                "val_metric": next((ckpt[k] for k in ("val_acc", "val_iou", "val_miou") if k in ckpt), None),
            }
        except Exception as exc:  # noqa: BLE001
            self.errors[task] = f"could not load: {exc}"

    def available(self, task) -> bool:
        return task in self.models

    @property
    def n_loaded(self):
        return len(self.models)

    # -- core inference (whole image resized, or tiled for large images)
    def _logits(self, model, arr, tiled=False, arr_b=None, batch=8):
        H, W = arr.shape[:2]
        if arr_b is not None and arr_b.shape[:2] != (H, W):
            arr_b = cv2.resize(arr_b, (W, H), interpolation=cv2.INTER_LINEAR)
        stack = arr if arr_b is None else np.concatenate([arr, arr_b], axis=2)
        with torch.no_grad():
            if not tiled:
                x = cv2.resize(stack, (TILE, TILE), interpolation=cv2.INTER_LINEAR)
                t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                out = F.interpolate(model(t), size=(H, W), mode="bilinear", align_corners=False)
                return out[0].cpu().numpy()
            ph, pw = (-H) % TILE, (-W) % TILE
            padded = np.pad(stack, ((0, ph), (0, pw), (0, 0)), mode="edge")
            Hp, Wp = padded.shape[:2]
            coords = [(y, x) for y in range(0, Hp, TILE) for x in range(0, Wp, TILE)]
            full = None
            for i in range(0, len(coords), batch):
                chunk = coords[i:i + batch]
                tiles = np.stack([padded[y:y + TILE, x:x + TILE] for y, x in chunk])
                t = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float().to(self.device)
                o = model(t).cpu().numpy()
                if full is None:
                    full = np.zeros((o.shape[1], Hp, Wp), np.float32)
                for j, (y, x) in enumerate(chunk):
                    full[:, y:y + TILE, x:x + TILE] = o[j]
            return full[:, :H, :W]

    # -- tasks
    def classify(self, image, top_k=3):
        arr = to_float_rgb(image)
        x = cv2.resize(arr, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x - self.mean) / self.std
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.models["classifier"](t), dim=1)[0].cpu().numpy()
        order = np.argsort(-probs)[:top_k]
        return {"label": self.classes[int(order[0])], "confidence": float(probs[order[0]]),
                "top": [(self.classes[int(i)], float(probs[i])) for i in order]}

    def explain_classification(self, image):
        """Grad-CAM heat-map (HxW in [0,1]) for the predicted land-cover class."""
        model = self.models["classifier"]
        arr = to_float_rgb(image)
        x = cv2.resize(arr, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x - self.mean) / self.std
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        store = {}
        layer = model.cnn.layer4[-1].conv3
        handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            with torch.enable_grad():
                logits = model(t)
                idx = int(logits.argmax(dim=1))
                grads = torch.autograd.grad(logits[0, idx], store["a"])[0]
        finally:
            handle.remove()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True))[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

    def detect_disaster(self, image, tiled=False):
        arr = to_float_rgb(image)
        logits = self._logits(self.models["disaster"], arr, tiled)
        mask = logits.argmax(axis=0).astype(np.uint8)
        pcts = {name: float((mask == c).mean() * 100) for c, name in DISASTER_NAMES.items()}
        hazards = {k: v for k, v in pcts.items() if k != "Background"}
        top, top_pct = max(hazards.items(), key=lambda kv: kv[1])
        affected = float(sum(hazards.values()))
        return {"mask": mask, "percentages": pcts, "affected_pct": affected,
                "severity": severity_label(affected),
                "dominant": top if top_pct >= 1.0 else "No disaster detected"}

    def segment_flood(self, image, threshold=0.5, tiled=False):
        arr = to_float_rgb(image)
        logits = self._logits(self.models["flood"], arr, tiled)
        mask = (1 / (1 + np.exp(-logits[0])) > threshold)
        pct = float(mask.mean() * 100)
        return {"mask": mask, "percentage": pct, "severity": severity_label(pct)}

    def detect_change(self, before, after, threshold=0.5, tiled=False):
        a, b = to_float_rgb(before), to_float_rgb(after)
        logits = self._logits(self.models["change"], a, tiled, arr_b=b)
        mask = (1 / (1 + np.exp(-logits[0])) > threshold)
        return {"mask": mask, "percentage": float(mask.mean() * 100)}

    # -- everything at once
    def analyze(self, before, after=None, tasks=None, tiled=False, flood_threshold=0.5,
                change_threshold=0.5, explain=True):
        tasks = list(tasks or ["classifier", "disaster", "flood", "change"])
        arr = to_float_rgb(before)
        res = {"image": arr, "image_after": None, "size": arr.shape[:2], "tiled": tiled}
        if after is not None:
            b = to_float_rgb(after)
            if b.shape[:2] != arr.shape[:2]:
                b = cv2.resize(b, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
            res["image_after"] = b
        if "classifier" in tasks and self.available("classifier"):
            res["classification"] = self.classify(arr)
            if explain:
                res["classification"]["cam"] = self.explain_classification(arr)
        if "disaster" in tasks and self.available("disaster"):
            res["disaster"] = self.detect_disaster(arr, tiled)
            if res["image_after"] is not None:
                after_d = self.detect_disaster(res["image_after"], tiled)
                res["disaster_after"] = after_d
                res["delta"] = {k: after_d["percentages"][k] - res["disaster"]["percentages"][k]
                                for k in DISASTER_NAMES.values() if k != "Background"}
        if "flood" in tasks and self.available("flood"):
            res["flood"] = self.segment_flood(arr, flood_threshold, tiled)
        if "change" in tasks and self.available("change") and res["image_after"] is not None:
            res["change"] = self.detect_change(arr, res["image_after"], change_threshold, tiled)
        return res


def build_report(res: dict, meta: dict | None = None) -> dict:
    """JSON-serialisable summary of an analyze() result (no arrays)."""
    rep = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "image_size_hw": list(res["size"]), "tiled": bool(res.get("tiled")), **(meta or {})}
    if "classification" in res:
        c = res["classification"]
        rep["land_cover"] = {"label": c["label"], "confidence": round(c["confidence"], 4),
                             "top": [{"class": n, "probability": round(p, 4)} for n, p in c["top"]]}
    if "disaster" in res:
        d = res["disaster"]
        rep["disaster"] = {"dominant": d["dominant"], "severity": d["severity"],
                           "affected_percent": round(d["affected_pct"], 2),
                           "percent_by_class": {k: round(v, 2) for k, v in d["percentages"].items()}}
    if "flood" in res:
        rep["flood"] = {"flooded_percent": round(res["flood"]["percentage"], 2), "severity": res["flood"]["severity"]}
    if "change" in res:
        rep["change"] = {"changed_percent": round(res["change"]["percentage"], 2)}
    if "delta" in res:
        rep["disaster_change_after_vs_before"] = {k: round(v, 2) for k, v in res["delta"].items()}
    return rep


# -------------------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description="GeoVision AI - analyse satellite images")
    ap.add_argument("images", nargs="+", help="image file(s) or a folder")
    ap.add_argument("--after", help="second image for change detection (single before image only)")
    ap.add_argument("--out", default="geovision_results")
    ap.add_argument("--root", help="folder that contains the checkpoints")
    ap.add_argument("--tiled", action="store_true", help="process large images in 256x256 tiles")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.images):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}) if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = GeoVisionPipeline(args.root)
    print(f"models loaded: {sorted(pipe.models)} | problems: {pipe.errors or 'none'}")
    after = load_pil(args.after) if args.after else None
    for f in files:
        res = pipe.analyze(load_pil(f), after if len(files) == 1 else None, tiled=args.tiled)
        rep = build_report(res, {"file": f.name})
        (out / f"{f.stem}_report.json").write_text(json.dumps(rep, indent=2))
        if "disaster" in res:
            png = overlay(res["image"], colorize(res["disaster"]["mask"]))
            (out / f"{f.stem}_overlay.png").write_bytes(to_png_bytes(png))
        print(f"{f.name}: " + json.dumps({k: v for k, v in rep.items() if k in ("land_cover", "disaster", "flood", "change")})[:300])


if __name__ == "__main__":
    _cli()
