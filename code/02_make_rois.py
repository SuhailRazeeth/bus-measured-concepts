"""Step 2: crop a square ROI around every lesion and compute mask-geometry (weak) concepts.

Outputs  <work_root>/rois/<dataset>/<id>.png  (+ _mask.png)  and  roi_manifest.csv
(manifest + roi_path, roi_mask_path + g_* geometry columns).

Usage:  python 02_make_rois.py [--workers 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import GEOM_COLS, load_cfg, manifest_path, mask_bbox, mask_geometry, read_gray, read_mask, \
    roi_manifest_path, square_crop


def process(args):
    r, out_root, rc = args
    try:
        img = read_gray(r["image_path"])
        mask, ctx = None, rc["context"]
        if isinstance(r["mask_path"], str) and r["mask_path"]:
            mask = read_mask(r["mask_path"])
            if mask.shape != img.shape:  # e.g. some CBIS-DDSM masks differ in size
                mask = cv2.resize(mask, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
            box = mask_bbox(mask)
            if box is None:
                return None
        elif not (isinstance(r.get("circle"), str) and r["circle"]):  # no mask, no circle: whole image
            box, ctx = (0, 0, img.shape[1], img.shape[0]), 0.0
        else:  # MIAS circle; MIAS y-axis origin is bottom-left
            cx, cy, rad = (float(v) for v in str(r["circle"]).split(","))
            cy = img.shape[0] - cy
            box = (cx - rad, cy - rad, cx + rad, cy + rad)
        roi, roi_m = square_crop(img, box, ctx, rc["min_box_px"], rc["out_size"], mask)
        stem = r["uid"].replace("/", "__")
        d = out_root / r["dataset"]
        d.mkdir(parents=True, exist_ok=True)
        rp = d / f"{stem}.png"
        cv2.imwrite(str(rp), roi)
        res = {"uid": r["uid"], "roi_path": str(rp), "roi_mask_path": ""}
        if roi_m is not None:
            mp = d / f"{stem}_mask.png"
            cv2.imwrite(str(mp), roi_m)
            res["roi_mask_path"] = str(mp)
            res.update(mask_geometry(mask))
        return res
    except Exception as e:
        return {"uid": r["uid"], "error": repr(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    man = pd.read_csv(manifest_path(cfg), dtype={"mask_path": str, "circle": str}, keep_default_na=False,
                      na_values=[""])
    man["mask_path"] = man["mask_path"].fillna("")
    out_root = cfg["work_root"] / "rois"
    jobs = [(r, out_root, cfg["roi"]) for r in man.to_dict("records")]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [x for x in tqdm(ex.map(process, jobs, chunksize=8), total=len(jobs)) if x]
    res = pd.DataFrame(res)
    errs = res[res.get("error", pd.Series(dtype=str)).notna()] if "error" in res else res.iloc[:0]
    if len(errs):
        print(f"{len(errs)} errors, first few:\n", errs.head())
    res = res[res["roi_path"].notna()].drop(columns=["error"], errors="ignore")
    for c in GEOM_COLS:
        if c not in res:
            res[c] = np.nan
    df = man.merge(res, on="uid", how="inner")
    df.to_csv(roi_manifest_path(cfg), index=False)
    print(f"wrote {roi_manifest_path(cfg)}  ({len(df)} ROIs)")
    print(df.groupby("dataset")[GEOM_COLS].median().round(3))
