"""Clinical validation of the label-free concepts against radiologist BI-RADS descriptors on BrEaST (CPU, < 1 min).

BrEaST is the only dataset with BI-RADS lexicon annotations. For every descriptor column of the BrEaST clinical table
(shape, orientation, margin, echogenicity, posterior features, halo, ... whatever the table provides) and every
category with >= 10 lesions, this reports the one-vs-rest AUC of each concept, computed three ways:
  medsam     concept measured on the automatic MedSAM mask (the training target; label-free)
  radiol     concept measured on the radiologist mask
  predicted  concept predicted from the image by a trained CBM that never saw BrEaST (held-out BrEaST runs, seed mean)
AUC > 0.5 = higher concept value in lesions with that descriptor. A pre-specified table of EXPECTED pairs
(e.g. irregular shape <-> solidity low, not parallel <-> taller than wide high) gives the headline numbers.

Output: <work_root>/birads_validation/{all_pairs.csv, expected_pairs.csv, report.txt}
Usage:  python 14_birads_validation.py [--run dinov2_b_medsam_int50_noexp_ic_u12]
"""
import argparse
import importlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import GEOM_COLS, INT_COLS, ds_dir, load_cfg, roi_manifest_path

m01 = importlib.import_module("01_build_manifest")
CC = GEOM_COLS + INT_COLS
SKIP = ("classification", "image", "file", "mask", "tumor", "case", "age", "birads", "bi-rads", "verification",
        "diagnosis", "pixel", "symptom", "interpretation")

# descriptor keyword -> category keyword -> (concept, expected direction +1 = higher in that category)
EXPECTED = [
    ("shape", "irregular", "g_solidity", -1), ("shape", "irregular", "g_ellipse_misfit", +1),
    ("shape", "irregular", "g_circularity", -1),
    ("orient", "not", "g_taller_than_wide", +1),
    ("margin", "NONCIRC", "g_spiculation", +1), ("margin", "NONCIRC", "g_radial_cv", +1),
    ("margin", "NONCIRC", "i_margin_blur", +1),
    ("margin", "spicul", "g_spiculation", +1), ("margin", "indistinct", "i_margin_blur", +1),
    ("echo", "heterogen", "i_heterogeneity", +1), ("echo", "complex", "i_heterogeneity", +1),
    ("echo", "anechoic", "i_hypoecho", +1),
    ("posterior", "shadow", "i_post_shadow", +1), ("posterior", "enhanc", "i_post_shadow", -1),
    ("halo", "yes", "i_halo", +1), ("halo", "present", "i_halo", +1),
]


def auc(y, s):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    return roc_auc_score(y, s) if len(s) > 10 and 0 < y.sum() < len(y) else np.nan


def boot_ci(y, s, B=1000, seed=0):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)


SPLIT = r"\s*(?:&|,|;|/|\+)\s*"


def tokens(series):
    """lower-cased token lists per cell (cells may hold several descriptors joined by & , ; /)."""
    vals = [("" if pd.isna(v) else str(v)).strip().lower() for v in series]
    return pd.Series([[t for t in re.split(SPLIT, v) if t and t != "nan"] for v in vals], index=series.index)


def load_table(root):
    for t in list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")) + list(root.rglob("*.csv")):
        try:
            d = m01.read_table(t)
        except Exception:
            continue
        if m01._find_col(d, "classification") is not None:
            return d
    raise SystemExit("BrEaST clinical table not found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--run", default="dinov2_b_medsam_int50_noexp_ic_u12", help="run folder for predicted concepts")
    ap.add_argument("--mode", default="cbm", help="cbm | seq | hyb | shuf: which concept head supplies the predicted concepts")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    W = cfg["work_root"]
    out = W / ("birads_validation" if (a.mode == "cbm" and a.run == "dinov2_b_medsam_int50_noexp_ic_u12") else f"birads_validation_{a.run}_{a.mode}")
    out.mkdir(exist_ok=True)

    meta = load_table(ds_dir(cfg, "BrEaST"))
    c_case = m01._find_col(meta, "case")
    meta["uid"] = "BrEaST/" + meta[c_case].astype(str).str.strip()
    print("BrEaST table columns (name | type | distinct values | example):")
    for c in meta.columns:
        ex = meta[c].dropna().astype(str).head(2).tolist()
        print(f"  {str(c)[:40]:40s} | {str(meta[c].dtype):8s} | {meta[c].nunique():4d} | {ex}")
    desc = [c for c in meta.columns if c != "uid" and not pd.api.types.is_numeric_dtype(meta[c])
            and not any(k in str(c).lower().replace(" ", "_") for k in SKIP) and 2 <= meta[c].nunique() <= 60]
    print("descriptor columns:", desc)

    man = pd.read_csv(roi_manifest_path(cfg))
    man = man[man.dataset == "BrEaST"]
    radiol = man[["uid", "label"] + GEOM_COLS].merge(pd.read_csv(W / "intensity" / "gt.csv")[["uid"] + INT_COLS], on="uid", how="left")
    medsam = pd.read_csv(W / "medsam" / "sam_geometry.csv")[["uid"] + GEOM_COLS] \
        .merge(pd.read_csv(W / "intensity" / "medsam.csv")[["uid"] + INT_COLS], on="uid", how="outer")
    sources = {"radiol": radiol, "medsam": medsam}
    labels = man[["uid", "label"]]
    # predicted concepts from the held-out-BrEaST CBM runs (model never trained on BrEaST)
    rd = W / "cbm_runs" / a.run
    P = []
    for d in sorted(rd.glob(f"{a.mode}_s*_BrEaST")):
        if not (d / "preds_test.csv").exists():
            continue
        names = json.load(open(d / "run.json"))["concepts"]
        p = pd.read_csv(d / "preds_test.csv")
        P.append(p[["uid"]].assign(**{n: p[f"c{j}"] for j, n in enumerate(names) if n in CC}))
    if P:
        pred = pd.concat(P).groupby("uid").mean().reset_index()
        sources["predicted"] = pred
        print(f"predicted concepts: {len(P)} seeds from {rd.name}")
    else:
        print(f"no held-out BrEaST runs found in {rd} (predicted column skipped)")

    rows = []
    for col in desc:
        tk = tokens(meta[col])
        cnt = pd.Series([t for ts in tk for t in set(ts)]).value_counts()
        for cat, n in cnt.items():
            if cat in ("not applicable", "none") or n < 10 or n > len(tk) - 10:
                continue
            ydf = pd.DataFrame({"uid": meta.uid, "y": tk.apply(lambda ts: cat in ts).astype(int)})
            for src, t in sources.items():
                m = ydf.merge(t, on="uid")
                for c in CC:
                    if c in m:
                        rows.append(dict(descriptor=col, category=cat, n=int(m.y.sum()), N=len(m), source=src, concept=c,
                                         auc=auc(m.y.values, m[c].values.astype(float))))
    allp = pd.DataFrame(rows)
    allp.to_csv(out / "all_pairs.csv", index=False)

    exp_rows = []
    for dkey, ckey, concept, sign in EXPECTED:
        cols = [c for c in desc if dkey in str(c).lower()]
        for col in cols:
            tk = tokens(meta[col])
            NA = {"not applicable", "not available", "none"}
            if ckey == "NONCIRC":  # any margin descriptor other than circumscribed (the lexicon's "not circumscribed")
                cats = sorted({t for ts in tk for t in ts if t not in NA | {"circumscribed", "no"}})
            else:
                cats = sorted({t for ts in tk for t in ts if re.search(ckey, t)})
            if not cats:
                continue
            ydf = pd.DataFrame({"uid": meta.uid, "y": tk.apply(lambda ts: any(t in cats for t in ts)).astype(int)})
            ydf = ydf[tk.apply(lambda ts: len(ts) > 0 and not set(ts) <= NA).values]  # drop cases without a descriptor
            if ydf.y.sum() < 10:
                continue
            r = dict(descriptor=col, category="non-circumscribed" if ckey == "NONCIRC" else "/".join(sorted(cats)),
                     concept=concept, expected=("+" if sign > 0 else "-"), n_pos=int(ydf.y.sum()), N=len(ydf))
            for src, t in sources.items():
                m = ydf.merge(t.drop(columns="label", errors="ignore"), on="uid").merge(labels, on="uid", how="left")
                if concept not in m:
                    continue
                s = sign * m[concept].values.astype(float)  # oriented so 0.5 = no relation, > 0.5 = expected direction
                r[src] = auc(m.y.values, s)
                lo, hi = boot_ci(m.y.values, s)
                r[f"{src}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
                # within one pathology class: is the concept tracking the descriptor, or only malignancy?
                for cls, nm in ((0, "inBen"), (1, "inMal")):
                    k = (m.label == cls).values
                    r[f"{src}_{nm}"] = auc(m.y.values[k], s[k]) if k.sum() > 10 and 3 <= m.y.values[k].sum() <= k.sum() - 3 else np.nan
            exp_rows.append(r)
    exp = pd.DataFrame(exp_rows)
    exp.to_csv(out / "expected_pairs.csv", index=False)

    L = ["BI-RADS DESCRIPTOR VALIDATION ON BrEaST (radiologist lexicon labels)", "",
         "Expected pairs: AUC oriented to the clinically expected direction (0.5 = unrelated, > 0.5 = as expected)",
         exp[[c for c in exp.columns if not c.endswith("_ci")]].assign(category=lambda d: d.category.str[:28])
         .round(3).to_string(index=False) if len(exp) else "(no expected pair found; check descriptor names below)",
         "(inBen / inMal = AUC within benign / within malignant lesions only; CIs in expected_pairs.csv)", "",
         "Descriptor columns and categories found:"]
    for col in desc:
        cnt = pd.Series([t for ts in tokens(meta[col]) for t in set(ts)]).value_counts()
        L.append(f"  {col}: " + ", ".join(f"{k} ({v})" for k, v in cnt.items()))
    if len(allp):
        best = allp[allp.source == "medsam"].assign(dev=lambda d: (d.auc - 0.5).abs()).sort_values("dev", ascending=False)
        L += ["", "Strongest concept-descriptor relations on MedSAM concepts (any direction):",
              best.head(25).drop(columns="dev").round(3).to_string(index=False)]

    # what does g_radial_cv measure? compare with elongation of the fitted ellipse and with lobulation around it
    import cv2
    from scipy.stats import spearmanr
    rows = []
    for _, r in man[man.roi_mask_path.notna()].iterrows():
        m = cv2.imread(str(r.roi_mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 10:
            continue
        (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
        a_, b_ = max(ax1, ax2) / 2, min(ax1, ax2) / 2
        pts = c[:, 0, :].astype(float)
        t = np.deg2rad(ang)
        u = (pts[:, 0] - cx) * np.cos(t) + (pts[:, 1] - cy) * np.sin(t)
        v = -(pts[:, 0] - cx) * np.sin(t) + (pts[:, 1] - cy) * np.cos(t)
        rad, th = np.hypot(u, v), np.arctan2(v, u)
        ax_u, ax_v = (ax1 / 2, ax2 / 2)  # fitEllipse: first axis along the rotated x
        r_ell = ax_u * ax_v / np.sqrt((ax_v * np.cos(th)) ** 2 + (ax_u * np.sin(th)) ** 2)
        ratio = rad / np.maximum(r_ell, 1e-6)
        rows.append(dict(uid=r.uid, elongation=1 - b_ / max(a_, 1e-6), lobulation_vs_ellipse=ratio.std() / ratio.mean()))
    el = pd.DataFrame(rows).merge(radiol[["uid", "g_radial_cv"]], on="uid")
    if len(el) > 10:
        L += ["", "What g_radial_cv measures (radiologist masks, BrEaST):",
              f"  Spearman(g_radial_cv, ellipse elongation 1-minor/major) = {spearmanr(el.g_radial_cv, el.elongation).statistic:.3f}",
              f"  Spearman(g_radial_cv, lobulation around fitted ellipse)  = {spearmanr(el.g_radial_cv, el.lobulation_vs_ellipse).statistic:.3f}"]
        mm = el.merge(pd.DataFrame({"uid": meta.uid, "noncirc": tokens(meta["Margin"]).apply(
            lambda ts: int(any(t not in {"circumscribed", "not applicable"} for t in ts)) if ts else np.nan)}), on="uid").dropna()
        L.append(f"  AUC for non-circumscribed margin: radial_cv {auc(mm.noncirc.values, mm.g_radial_cv.values):.3f}, "
                 f"lobulation-around-ellipse {auc(mm.noncirc.values, mm.lobulation_vs_ellipse.values):.3f}, "
                 f"elongation {auc(mm.noncirc.values, mm.elongation.values):.3f}")
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nsaved {out}")
