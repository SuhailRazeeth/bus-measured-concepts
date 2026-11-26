"""Label-free BI-RADS echo (intensity) concepts from each ultrasound ROI + lesion mask (CPU, ~5-10 min).

Adds 5 concepts to the 6 geometry ones (see common.intensity_concepts; higher = more suspicious for all):
  i_hypoecho  i_heterogeneity  i_post_shadow  i_margin_blur  i_halo
Computed twice: with the radiologist masks (gt; used only to EVALUATE / intervene) and with the automatic MedSAM
masks (the label-free training labels for 08 --intensity).

Output: <work_root>/intensity/<geom>.csv   (uid, dataset, label, i_*)
        <work_root>/intensity/report.txt   (per-dataset malignancy AUC of each concept, MedSAM vs radiologist agreement)
Usage:  python 12_intensity_concepts.py [--geoms gt medsam]
Then:   python -u 08_train_cbm.py --backbone dinov2_b --geom medsam --modes cbm --int_p 0.25 --unfreeze 12 --intensity
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import INT_COLS, intensity_concepts, load_cfg, roi_manifest_path


def auc(y, s):
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if ok.sum() > 10 and len(np.unique(y[ok])) == 2 else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--geoms", nargs="*", default=["gt", "medsam"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    out = cfg["work_root"] / "intensity"
    out.mkdir(exist_ok=True)

    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[(df.modality == "US") & (df.lesion_type == "mass") & df.label.notna()].reset_index(drop=True)
    res = {}
    for geom in a.geoms:
        if geom == "gt":
            d = df[["uid", "dataset", "label", "roi_path", "roi_mask_path"]].rename(columns={"roi_mask_path": "mpath"})
        else:
            sam = pd.read_csv(cfg["work_root"] / geom / "sam_geometry.csv")[["uid", "sam_mask_path"]]
            d = df[["uid", "dataset", "label", "roi_path"]].merge(sam.rename(columns={"sam_mask_path": "mpath"}), on="uid")
        d = d[d.mpath.notna() & (d.mpath.astype(str) != "")].reset_index(drop=True)
        rows = []
        for i, r in enumerate(d.itertuples()):
            img = cv2.imread(r.roi_path, cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(r.mpath, cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            if msk.shape != img.shape:
                msk = cv2.resize(msk, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
            c = intensity_concepts(img, msk > 127 if msk.max() > 1 else msk > 0)
            rows.append(dict(uid=r.uid, dataset=r.dataset, label=r.label, **{k: float(v) for k, v in c.items()}))
            if (i + 1) % 500 == 0:
                print(f"  {geom}: {i + 1}/{len(d)}", flush=True)
        res[geom] = pd.DataFrame(rows)
        res[geom].to_csv(out / f"{geom}.csv", index=False)
        print(f"{geom}: {len(res[geom])} lesions -> {out / f'{geom}.csv'}", flush=True)

    L = ["INTENSITY CONCEPTS: malignancy AUC per dataset (0.5 = useless; > 0.5 = higher value more malignant)", ""]
    for geom, t in res.items():
        tab = pd.DataFrame({ds: {c: auc(g.label.values.astype(int), g[c].values) for c in INT_COLS}
                            for ds, g in t.groupby("dataset")}).round(3)
        tab["n_missing_%"] = (t[INT_COLS].isna().mean() * 100).round(1)
        L += [f"[{geom} masks]", tab.to_string(), ""]
    if "gt" in res and len(res) > 1:
        for geom in [g for g in res if g != "gt"]:
            mm = res["gt"].merge(res[geom], on=["uid", "dataset"], suffixes=("_gt", "_auto"))
            sp = pd.DataFrame({ds: {c: spearmanr(g[f"{c}_gt"], g[f"{c}_auto"], nan_policy="omit").statistic for c in INT_COLS}
                               for ds, g in mm.groupby("dataset")}).round(3)
            L += [f"Spearman agreement {geom} vs radiologist masks", sp.to_string(), ""]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt + "\n")
    print("\n" + txt + f"\nsaved {out}")
