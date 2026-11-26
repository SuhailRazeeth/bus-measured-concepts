"""Shared helpers: config, image I/O (DICOM/PGM/PNG/JPG), ROI cropping, mask-geometry concepts."""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".dcm"}


# ----------------------------------------------------------------------------- config
def load_cfg(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else HERE / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["work_root"] = Path(cfg["work_root"])
    cfg["work_root"].mkdir(parents=True, exist_ok=True)
    if cfg.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(cfg["hf_home"]))
    return cfg


def ds_dir(cfg: dict, name: str) -> Path:
    return cfg["data_root"] / cfg["datasets"][name]


def manifest_path(cfg: dict) -> Path:
    return cfg["work_root"] / "manifest.csv"


def roi_manifest_path(cfg: dict) -> Path:
    return cfg["work_root"] / "roi_manifest.csv"


# ----------------------------------------------------------------------------- image I/O
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def read_gray(path: str | os.PathLike) -> np.ndarray:
    """Read any supported image as a 2-D uint8 array."""
    path = str(path)
    if path.lower().endswith(".dcm"):
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr = _to_uint8(arr)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
        return arr
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cannot read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    return arr


def read_mask(path: str | os.PathLike) -> np.ndarray:
    m = read_gray(path) if str(path).lower().endswith(".dcm") else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cannot read mask {path}")
    return (m > 127).astype(np.uint8) if m.max() > 1 else m.astype(np.uint8)


def is_binary_image(path: str | os.PathLike) -> bool:
    try:
        a = read_gray(path)
    except Exception:
        return False
    return len(np.unique(a[:: max(1, a.shape[0] // 256), :: max(1, a.shape[1] // 256)])) <= 3


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ----------------------------------------------------------------------------- ROI
def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_crop(img: np.ndarray, box, context: float, min_box: int, out_size: int, mask: np.ndarray | None = None):
    """Square crop around box (x0,y0,x1,y1) with context, zero-padded, resized to out_size."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, min_box) * (1 + 2 * context)
    half = side / 2
    X0, Y0, X1, Y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    H, W = img.shape[:2]
    pad = max(0, -X0, -Y0, X1 - W, Y1 - H)

    def _cut(a, interp):
        if pad:
            a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        c = a[Y0 + pad : Y1 + pad, X0 + pad : X1 + pad]
        return cv2.resize(c, (out_size, out_size), interpolation=interp)

    roi = _cut(img, cv2.INTER_AREA)
    roi_m = _cut(mask.astype(np.uint8) * 255, cv2.INTER_NEAREST) if mask is not None else None
    return roi, roi_m


# ----------------------------------------------------------------------------- mask geometry concepts
GEOM_COLS = ["g_solidity", "g_circularity", "g_ellipse_misfit", "g_radial_cv", "g_spiculation", "g_taller_than_wide"]


def mask_geometry(mask: np.ndarray) -> dict:
    """Shape/margin descriptors computed from a binary lesion mask (largest component).

    Higher values = more suspicious, for all except g_solidity / g_circularity (lower = more irregular).
      g_solidity        area / convex-hull area                    (shape regularity)
      g_circularity     4*pi*A / P^2                               (shape regularity)
      g_ellipse_misfit  1 - IoU(mask, fitted ellipse)              (BI-RADS 'oval' vs 'irregular')
      g_radial_cv       std/mean of centroid-to-boundary distance  (lobulation)
      g_spiculation     high-frequency energy fraction of radial profile (spiculated / angular margin)
      g_taller_than_wide  bbox height / width                      (US 'not parallel' orientation)
    """
    out = {k: np.nan for k in GEOM_COLS}
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return out
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    if area < 20 or len(c) < 10:
        return out
    hull = cv2.convexHull(c)
    out["g_solidity"] = area / max(cv2.contourArea(hull), 1e-6)
    out["g_circularity"] = 4 * np.pi * area / max(perim**2, 1e-6)
    if len(c) >= 5:
        ell = cv2.fitEllipse(c)
        em = np.zeros_like(m)
        cv2.ellipse(em, ell, 1, -1)
        inter, union = np.logical_and(m, em).sum(), np.logical_or(m, em).sum()
        out["g_ellipse_misfit"] = 1 - inter / max(union, 1)
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    pts = c[:, 0, :].astype(np.float32)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    bins = np.linspace(-np.pi, np.pi, 129)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, 127)
    prof = np.array([rad[idx == i].max() if np.any(idx == i) else np.nan for i in range(128)])
    if np.isnan(prof).all():
        return out
    good = ~np.isnan(prof)
    prof = np.interp(np.arange(128), np.flatnonzero(good), prof[good], period=128)
    out["g_radial_cv"] = float(prof.std() / max(prof.mean(), 1e-6))
    spec = np.abs(np.fft.rfft(prof - prof.mean())) ** 2
    out["g_spiculation"] = float(spec[8:].sum() / max(spec[1:].sum(), 1e-9))
    x, y, w, h = cv2.boundingRect(c)
    out["g_taller_than_wide"] = h / max(w, 1)
    return out


# ----------------------------------------------------------------------------- intensity (echo) concepts
INT_COLS = ["i_hypoecho", "i_heterogeneity", "i_post_shadow", "i_margin_blur", "i_halo"]


def _largest(mask: np.ndarray) -> np.ndarray | None:
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    return (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _disk(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def intensity_concepts(img: np.ndarray, mask: np.ndarray) -> dict:
    """BI-RADS ultrasound echo descriptors measured from the ROI image and a lesion mask (no labels needed).

    All oriented so that HIGHER = MORE SUSPICIOUS. Intensities are divided by the ROI's robust contrast
    (p90 - p10 of non-padding pixels), so scanner gain matters less; 08 also z-scores them per dataset.
      i_hypoecho       median(surrounding tissue ring) - median(lesion)          (echo pattern: hypo/anechoic)
                       NOTE: on real data this runs the OTHER way (AUC 0.27-0.50): the darkest lesions are mostly
                       anechoic cysts (benign). Read it as "cyst-like darkness"; the CBM head learns the sign.
      i_heterogeneity  std of lesion interior (lightly smoothed)                 (heterogeneous / complex echo)
      i_post_shadow    mean(beside lesion, same depth) - mean(directly below)    (posterior shadowing; <0 = enhancement)
      i_margin_blur    -mean gradient magnitude on the lesion boundary           (indistinct vs circumscribed margin)
      i_halo           mean(thin band just outside) - mean(tissue further out)   (echogenic halo)
    Pixels that are exactly 0 outside the lesion are treated as crop padding and ignored.
    """
    out = {k: np.nan for k in INT_COLS}
    m = _largest(mask)
    if m is None or m.sum() < 20:
        return out
    m = m.astype(bool)
    im = img.astype(np.float32)
    valid = im > 0
    v = im[valid | m]
    s = float(np.percentile(v, 90) - np.percentile(v, 10)) + 1e-6
    area = m.sum()
    d = max(2, int(round(0.12 * np.sqrt(area / np.pi))))  # band width ~6% of the equivalent diameter
    mu8 = m.astype(np.uint8)
    dil = {k: cv2.dilate(mu8, _disk(k * d)).astype(bool) for k in (1, 2, 4)}
    core = cv2.erode(mu8, _disk(d)).astype(bool)
    core = core if core.sum() >= 20 else m
    ring = dil[4] & ~dil[2] & valid      # surrounding tissue
    band = dil[1] & ~m & valid           # just outside the margin

    def ok(r):
        return r.sum() >= 20

    if ok(ring):
        out["i_hypoecho"] = (np.median(im[ring]) - np.median(im[m])) / s
        if ok(band):
            out["i_halo"] = (im[band].mean() - im[ring].mean()) / s
    sm = cv2.GaussianBlur(im, (5, 5), 1.0)
    out["i_heterogeneity"] = float(sm[core].std()) / s
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    edge = (cv2.dilate(mu8, _disk(1)) - cv2.erode(mu8, _disk(1))).astype(bool) & valid
    if ok(edge):
        out["i_margin_blur"] = -float(np.hypot(gx, gy)[edge].mean()) / s
    # posterior acoustic features: below the lesion vs beside it at the same depth
    ys, xs = np.nonzero(m)
    y1, x0, x1 = ys.max() + 1, xs.min(), xs.max() + 1
    h, w = ys.max() + 1 - ys.min(), x1 - x0
    H, W = m.shape
    rows = slice(y1, min(H, y1 + max(d, h // 2)))
    free = valid & ~dil[1]
    post = np.zeros_like(m)
    post[rows, x0 + w // 6 : x1 - w // 6] = True
    side = np.zeros_like(m)
    side[rows, max(0, x0 - w // 2) : x0] = True
    side[rows, x1 : min(W, x1 + w // 2)] = True
    post, side = post & free, side & free
    if ok(post) and ok(side):
        out["i_post_shadow"] = (im[side].mean() - im[post].mean()) / s
    return out
