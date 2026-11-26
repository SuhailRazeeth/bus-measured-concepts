"""TEST C: replace radiologist masks with automatic masks from a box-prompted SAM / MedSAM (GPU).

For every US lesion that has a mask, a box is taken from the mask's bounding box and randomly enlarged/shifted
(--jitter, default 10%) to imitate a quick box drawn by a clinician. SAM segments the lesion from that box; the same
geometry concepts are computed from the predicted mask. Output: <work_root>/sam/sam_geometry.csv (uid, dice, g_*).
Then run:  python 06_pilot2.py --geom sam

Usage:  CUDA_VISIBLE_DEVICES=0 python 07_sam_masks.py [--model wanglab/medsam-vit-base] [--jitter 0.1]
        (default model facebook/sam-vit-base; downloads once, needs internet the first time)
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import GEOM_COLS, load_cfg, mask_bbox, mask_geometry, roi_manifest_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--datasets", nargs="*", default=["BUS-BRA", "BUSB"])
    ap.add_argument("--limit", type=int, default=0, help="only N lesions per dataset (quick test)")
    ap.add_argument("--name", default="sam", help="output folder name under work_root (e.g. medsam)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    from transformers import SamModel, SamProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SamProcessor.from_pretrained(a.model)
    model = SamModel.from_pretrained(a.model).to(dev).eval()
    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[df.dataset.isin(a.datasets) & df.roi_mask_path.notna()].reset_index(drop=True)
    if a.limit:
        df = df.groupby("dataset").head(a.limit).reset_index(drop=True)
    out_dir = cfg["work_root"] / a.name
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        img = cv2.imread(r.roi_path, cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(r.roi_mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        box = mask_bbox(gt)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        j = a.jitter
        x0 = x0 - rng.uniform(0, j) * w + rng.uniform(-j, j) * w / 2
        x1 = x1 + rng.uniform(0, j) * w + rng.uniform(-j, j) * w / 2
        y0 = y0 - rng.uniform(0, j) * h + rng.uniform(-j, j) * h / 2
        y1 = y1 + rng.uniform(0, j) * h + rng.uniform(-j, j) * h / 2
        H, W = img.shape
        b = [float(np.clip(x0, 0, W - 1)), float(np.clip(y0, 0, H - 1)), float(np.clip(x1, 1, W)), float(np.clip(y1, 1, H))]
        inp = proc(images=Image.fromarray(img).convert("RGB"), input_boxes=[[b]], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp, multimask_output=False)
        pm = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                     inp["reshaped_input_sizes"].cpu())[0][0, 0].numpy().astype(np.uint8)
        dice = 2 * (pm & gt).sum() / max(pm.sum() + gt.sum(), 1)
        p = out_dir / r.dataset / (r.uid.replace("/", "__") + "_sam.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), pm * 255)
        rows.append(dict(uid=r.uid, dataset=r.dataset, label=r.label, dice=dice, sam_mask_path=str(p), **mask_geometry(pm)))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "sam_geometry.csv", index=False)
    print(res.groupby("dataset").dice.describe().round(3))
    # quick check: geometry AUC with SAM masks vs radiologist masks
    from sklearn.metrics import roc_auc_score

    gt_geo = df.set_index("uid")[GEOM_COLS]
    print("\nmalignancy AUC per geometry concept  (radiologist mask | SAM mask)")
    for d, g in res.groupby("dataset"):
        for c in GEOM_COLS:
            s_sam, s_gt = g[c].values, gt_geo.loc[g.uid, c].values
            ok = ~np.isnan(s_sam) & ~np.isnan(s_gt)
            a1, a2 = roc_auc_score(g.label[ok], s_gt[ok]), roc_auc_score(g.label[ok], s_sam[ok])
            print(f"  {d:8s} {c:20s} {max(a1, 1 - a1):.3f} | {max(a2, 1 - a2):.3f}")
    print(f"\nsaved {out_dir / 'sam_geometry.csv'}")
