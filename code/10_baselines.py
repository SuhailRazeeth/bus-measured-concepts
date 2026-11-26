"""Baselines for the paper (CPU, uses the frozen features from 04 + geometry; ~10-20 min).

Same leave-one-dataset-out protocol as 08 (train on all other US datasets, test on the held-out one, no target labels).
For each backbone with saved features:
  probe_lr          logistic regression on frozen features (linear probe)
  posthoc_cbm       post-hoc CBM: ridge features -> MedSAM geometry concepts, then LR on the 6 predicted concepts
  text_cbm          label-free text CBM (BiomedCLIP only): cosine to BI-RADS text prompts -> LR on concept scores
  text_zeroshot     BiomedCLIP "malignant vs benign" prompt, no training at all
Backbone-free references (need a lesion mask/box at TEST time, unlike the image-only models):
  geom_lr_medsam       LR on MedSAM-mask geometry   (segment-then-measure pipeline)
  geom_lr_radiologist  LR on radiologist-mask geometry (upper reference)
If <work_root>/intensity/ exists (12_intensity_concepts.py), the same baselines are repeated with the 11 concepts
(geometry + echo): posthoc_cbm_ic, geomint_lr_medsam, geomint_lr_radiologist.
Full fine-tuning is run with 08:  --modes bb --unfreeze 12

AUC with 95% patient-cluster bootstrap CI.  Output: <work_root>/baselines/{baselines.csv, report.txt}
Usage:  python 10_baselines.py [--backbones biomedclip dinov2_b] [--geom medsam] [--boot 1000]
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from common import GEOM_COLS, INT_COLS, load_cfg, roi_manifest_path

p1 = importlib.import_module("05_pilot_experiments")
p9 = importlib.import_module("09_analyze_cbm")
auc, zs, cluster_boot = p1.auc, p1.zs, p9.cluster_boot
warnings.filterwarnings("ignore")
TEXT_CONCEPTS = ["shape_irregular", "margin_noncirc", "margin_spiculated"]


def fit_lr(X, y, pid, rng, Cs=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """LR with C chosen on a patient-disjoint 15% validation split of the training data, then refit on all."""
    u = np.unique(pid)
    va = np.isin(pid, rng.choice(u, max(1, int(0.15 * len(u))), replace=False))
    best, bC = -1, 1.0
    for C in Cs:
        a = auc(y[va], LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[~va], y[~va]).decision_function(X[va]))
        if a == a and a > best:
            best, bC = a, C
    return LogisticRegression(C=bC, max_iter=5000, class_weight="balanced").fit(X, y)


def zgroup(df, cols):
    """Per-dataset z-scoring (label-free, as in 08)."""
    return df.groupby("dataset")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8)).values


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backbones", nargs="*", default=["biomedclip", "dinov2_b"])
    ap.add_argument("--geom", default="medsam")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    rng = np.random.default_rng(0)

    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[(df.lesion_type == "mass") & df.label.notna() & (df.modality == "US")].reset_index(drop=True)
    us_sets = [d for d in cfg["pilot"]["us_targets"] if (df.dataset == d).any()]
    y_all = df.label.values.astype(int)
    pid_all = df.patient_id.astype(str).values
    G_rad = zgroup(df, GEOM_COLS)
    sam = pd.read_csv(cfg["work_root"] / a.geom / "sam_geometry.csv")[["uid"] + GEOM_COLS]
    dm = df[["uid", "dataset"]].merge(sam, on="uid", how="left")
    G_sam = zgroup(dm, GEOM_COLS)
    idir = cfg["work_root"] / "intensity"
    GI = {}  # 11-concept versions (geometry + intensity)
    if (idir / "gt.csv").exists() and (idir / f"{a.geom}.csv").exists():
        for nm, G, f in [("radiologist", G_rad, "gt.csv"), ("medsam", G_sam, f"{a.geom}.csv")]:
            it = df[["uid", "dataset"]].merge(pd.read_csv(idir / f)[["uid"] + INT_COLS], on="uid", how="left")
            GI[nm] = np.hstack([G, zgroup(it, INT_COLS)])
        print("intensity concepts found: adding 11-concept baselines")

    feats = {}
    for b in a.backbones:
        try:
            feats[b] = p1.load_feats(cfg, b, df)
        except FileNotFoundError:
            print(f"no features for {b}, skipped")

    rows = []
    for held in us_sets:
        te, tr = (df.dataset == held).values, (df.dataset != held).values
        y, pid = y_all[te], pid_all[te]
        scores = {}
        for b, (F, text) in feats.items():
            scores[f"{b}:probe_lr"] = fit_lr(F[tr], y_all[tr], pid_all[tr], rng).decision_function(F[te])
            ok = tr & ~np.isnan(G_sam).any(1)
            rg = Ridge(alpha=1.0).fit(F[ok], G_sam[ok])
            Ctr, Cte = rg.predict(F[tr]), rg.predict(F[te])
            scores[f"{b}:posthoc_cbm"] = fit_lr(Ctr, y_all[tr], pid_all[tr], rng).decision_function(Cte)
            if GI:
                ok = tr & ~np.isnan(GI["medsam"]).any(1)
                rg = Ridge(alpha=1.0).fit(F[ok], GI["medsam"][ok])
                scores[f"{b}:posthoc_cbm_ic"] = fit_lr(rg.predict(F[tr]), y_all[tr], pid_all[tr], rng).decision_function(rg.predict(F[te]))
            if text:
                S = np.stack([F @ (text[f"text_US_{c}_pos"] - text[f"text_US_{c}_neg"]) for c in TEXT_CONCEPTS], 1)
                scores[f"{b}:text_cbm"] = fit_lr(zs(S[tr]), y_all[tr], pid_all[tr], rng).decision_function(zs(S[te], S[tr]))
                if "text_US_malignant_pos" in text:
                    scores[f"{b}:text_zeroshot"] = F[te] @ (text["text_US_malignant_pos"] - text["text_US_malignant_neg"])
        geo_sets = [("geom_lr_medsam", G_sam), ("geom_lr_radiologist", G_rad)]
        geo_sets += [("geomint_lr_medsam", GI["medsam"]), ("geomint_lr_radiologist", GI["radiologist"])] if GI else []
        for name, G in geo_sets:
            ok_tr, ok_te = tr & ~np.isnan(G).any(1), te & ~np.isnan(G).any(1)
            if ok_te.sum() > 10:
                m = fit_lr(G[ok_tr], y_all[ok_tr], pid_all[ok_tr], rng)
                s = np.full(te.sum(), np.nan)
                s[ok_te[te]] = m.decision_function(G[ok_te])
                scores[name] = s
        # bootstrap (rows with NaN scores are dropped inside auc)
        bs = cluster_boot(y, scores, pid, a.boot, rng)
        for name, s in scores.items():
            lo, hi = np.nanpercentile(bs[name], [2.5, 97.5])
            bb, _, meth = name.rpartition(":")
            rows.append(dict(heldout=held, backbone=bb or "-", method=meth, n=int((~np.isnan(s)).sum()),
                             auc=auc(y, s), ci=f"[{lo:.3f}, {hi:.3f}]"))
        print(f"{held}: " + ", ".join(f"{r['backbone']}:{r['method']} {r['auc']:.3f}" for r in rows if r["heldout"] == held), flush=True)

    out = cfg["work_root"] / "baselines"
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "baselines.csv", index=False)
    piv = res.pivot_table(index=["backbone", "method"], columns="heldout", values="auc").round(3)
    txt = "BASELINES (leave-one-dataset-out AUC; CIs in baselines.csv)\n\n" + piv.to_string() + "\n"
    (out / "report.txt").write_text(txt)
    print("\n" + txt + f"\nsaved {out}")
