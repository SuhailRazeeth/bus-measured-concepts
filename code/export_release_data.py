"""Collect the derived data promised in the paper into one folder for the code release (CPU, about 1 min).

Writes, under <out> (default: ../data next to this folder):
  lesions.csv               lesion identifier, dataset, label, patient identifier, scanner (no image paths)
  concepts_medsam.csv       eleven raw concept values measured on the MedSAM masks, and MedSAM Dice
  concepts_radiologist.csv  the same concepts measured on the radiologist masks
  splits.csv                role of every lesion (train / val / test) for every held-out dataset and seed,
                            recomputed exactly as in 08_train_cbm.py (patient level 15% validation split)
  medsam_masks/<dataset>/   box prompted MedSAM masks in the 384 x 384 ROI frame (PNG)

No images and no radiologist masks are copied. Obtain the images from the original sources (see data/README.md).
Usage:  python export_release_data.py [--out ../data] [--no_masks] [--skip_mask_datasets BUSB]
"""
import argparse
import importlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import GEOM_COLS, INT_COLS, load_cfg, roi_manifest_path

s08 = importlib.import_module("08_train_cbm")
CC = GEOM_COLS + INT_COLS
US = ["BUSI", "BUSB", "BUS-BRA", "BrEaST", "BUS-UCLM"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--no_masks", action="store_true")
    ap.add_argument("--skip_mask_datasets", nargs="*", default=[],
                    help="datasets whose license does not allow sharing derived masks")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    W, out = cfg["work_root"], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(roi_manifest_path(cfg))
    keep = [c for c in ("uid", "dataset", "label", "patient_id", "device", "modality") if c in man]
    les = man[keep]
    les = les[les.dataset.isin(US)].drop(columns="modality", errors="ignore")
    les.to_csv(out / "lesions.csv", index=False)

    sam = pd.read_csv(W / "medsam" / "sam_geometry.csv")
    ms = sam[["uid", "dataset", "dice"] + GEOM_COLS].merge(pd.read_csv(W / "intensity" / "medsam.csv")[["uid"] + INT_COLS], on="uid", how="left")
    ms.to_csv(out / "concepts_medsam.csv", index=False)
    rad = man[man.dataset.isin(US)][["uid", "dataset"] + GEOM_COLS].merge(pd.read_csv(W / "intensity" / "gt.csv")[["uid"] + INT_COLS], on="uid", how="left")
    rad = rad.dropna(subset=GEOM_COLS, how="all")
    rad.to_csv(out / "concepts_radiologist.csv", index=False)

    # splits, as in 08_train_cbm.py: ultrasound lesions in manifest order, patient level validation split
    df = man[man.modality == "US"].reset_index(drop=True) if "modality" in man else man[man.dataset.isin(US)].reset_index(drop=True)
    rows = []
    for held in [d for d in cfg["pilot"]["us_targets"] if (df.dataset == d).any()]:
        tr_idx = np.flatnonzero((df.dataset != held).values)
        te_idx = np.flatnonzero((df.dataset == held).values)
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            val = s08.patient_split(df.patient_id.values[tr_idx], cfg["train"]["val_fraction"], rng)
            role = np.empty(len(df), dtype=object)
            role[tr_idx] = np.where(val, "val", "train")
            role[te_idx] = "test"
            rows.append(pd.DataFrame({"uid": df.uid, "heldout": held, "seed": seed, "role": role}))
    pd.concat(rows).to_csv(out / "splits.csv", index=False)

    n = 0
    if not a.no_masks:
        for r in sam.itertuples():
            if r.dataset in a.skip_mask_datasets or not isinstance(r.sam_mask_path, str):
                continue
            src = Path(r.sam_mask_path)
            if src.exists():
                dst = out / "medsam_masks" / r.dataset / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"wrote {out}: {len(les)} lesions, {len(ms)} MedSAM concept rows, {len(rad)} radiologist concept rows, "
          f"{len(rows)} splits, {n} masks")
