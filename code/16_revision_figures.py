"""Revision figures (CPU, < 1 min). Run after 09 (all runs) and 15_stats.py.

  fig9_tradeoff.pdf        accuracy (mean AUC without BUSI, from 15) against correctability (mean AUC change when the
                           concepts of every test case are replaced by radiologist-mask values, from 09 section 4)
  fig10_concept_share.pdf  hybrid models: share of the logit variance carried by the named concepts, and the AUC of
                           the concept part, the residual part and the full logit, per held-out dataset (09 section 8)
  fig11_hybrid_cases.pdf   randomly sampled held-out cases for one hybrid model (one network, so the explanation is
                           exactly additive): ROI + MedSAM outline, concept contributions and the image residual
Every figure is saved as .pdf and .png (300 dpi) in <work_root>/figures/, with the plotted numbers as .csv.

Usage:  python 16_revision_figures.py [--case_run biomedclip_medsam_noexp_ic_u12_rp01] [--case_held BUS-BRA]
"""
import argparse
import importlib
import json

import cv2
import numpy as np
import pandas as pd

f13 = importlib.import_module("13_figures")  # shared style, palette, concept names and save()
plt, save, NICE, SERIES, INK, INK2, POS, NEG = f13.plt, f13.save, f13.NICE, f13.SERIES, f13.INK, f13.INK2, f13.POS, f13.NEG
from common import load_cfg, roi_manifest_path  # noqa: E402

DS4 = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST"]
SHORT = {"Black box": "BB", "Multitask black box": "MT", "Joint CBM p=0.25": "CBM p=.25", "Joint CBM p=0.5": "CBM p=.5",
         "Joint CBM p=0.75": "CBM p=.75", "Joint CBM p=1.0": "CBM p=1", "Joint CBM, 6 geometry concepts": "CBM geo",
         "Shuffled-concept CBM": "Shuffled", "Sequential CBM": "Sequential", "Hybrid CBM, penalty 0.1": "Hybrid .1",
         "Hybrid CBM, penalty 1.0": "Hybrid 1"}
LABEL_POS = {  # (backbone, model) -> (offset, ha, va): fixed placements where neighbouring labels would collide
    ("DINOv2", "Joint CBM p=0.25"): ((5, -2), "left", "top"),
    ("DINOv2", "Joint CBM p=0.5"): ((5, -3), "left", "top"),
    ("BiomedCLIP", "Joint CBM p=0.5"): ((-5, -3), "right", "top"),
    ("BiomedCLIP", "Joint CBM, 6 geometry concepts"): ((5, -3), "left", "top"),
}
FAMILY = {"Joint": SERIES[0], "Shuffled": SERIES[4], "Sequential": SERIES[2], "Hybrid": SERIES[1]}


def correction_gain(CR, root, mode):
    """mean over DS4 of AUC(all cases corrected with radiologist values) - AUC(no correction)."""
    f = CR / root / "analysis" / "4_concept_intervention.csv"
    if not f.exists():
        return np.nan
    t = pd.read_csv(f)
    if "mode" in t.columns:
        t = t[t["mode"] == mode]
    t = t[(t.source == "radiologist") & t.heldout.isin(DS4)]
    g = t.pivot_table(index="heldout", columns="fraction", values="auc")
    return float((g[1.0] - g[0.0]).mean()) if 1.0 in g and 0.0 in g else np.nan


def fig9(cfg, out):
    mt = cfg["work_root"] / "stats" / "main_table.csv"
    if not mt.exists():
        print("  fig9 skipped (run 15_stats.py first)")
        return
    m = pd.read_csv(mt)
    s15 = importlib.import_module("15_stats")
    CR = cfg["work_root"] / "cbm_runs"
    rows = []
    for bk, bname in s15.BACKBONES.items():
        for name, (root, mode) in s15.configs(bk).items():
            r = m[(m.backbone == bname) & (m.model == name)]
            if r.empty:
                continue
            rows.append(dict(backbone=bname, model=name, auc=float(r.mean_noBUSI.iloc[0]),
                             gain=correction_gain(CR, root, mode) if mode in ("cbm", "shuf", "seq", "hyb") else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(out / "fig9_tradeoff.csv", index=False)
    bks = list(t.backbone.unique())
    fig, axs = plt.subplots(1, len(bks), figsize=(3.5 * len(bks), 3.1), sharey=True, squeeze=False)
    for ax, b in zip(axs[0], bks):
        d = t[t.backbone == b]
        refs = [(ref, ls, d[d.model == ref].auc) for ref, ls in (("Black box", "--"), ("Multitask black box", ":"))]
        refs = [(ref, ls, v.iloc[0]) for ref, ls, v in refs if len(v)]
        for ref, ls, v in refs:  # reference lines are named in the figure legend, not in the plot
            ax.axhline(v, color=INK2, lw=0.9, ls=ls)
        ax.axvline(0, color=INK2, lw=0.8)
        pts = d.dropna(subset=["gain"]).sort_values("auc", ascending=False)
        XMIN = -0.13  # points further left (the shuffled control) are drawn at the edge with their value
        placed = []
        for _, r in pts.iterrows():
            fam = next(k for k in FAMILY if r.model.startswith(k))
            x = max(r.gain, XMIN)
            clipped = r.gain < XMIN
            ax.scatter(x, r.auc, s=36, color=FAMILY[fam], edgecolor="white", linewidth=0.8, zorder=3,
                       marker="<" if clipped else "o")
            lab = SHORT[r.model] + (f" ({r.gain:+.2f})" if clipped else "")
            # put the label below the point when a label was already placed just above it
            # a label placed just above-left of this point would collide: put this one to the left of the point
            close = any(abs(x - px) < 0.025 and 0 <= py - r.auc < 0.006 for px, py in placed)
            xy, ha, va = ((-5, -3), "right", "top") if close else ((5, 3), "left", "bottom")
            if (b, r.model) in LABEL_POS:
                xy, ha, va = LABEL_POS[(b, r.model)]
            ax.annotate(lab, (x, r.auc), textcoords="offset points", xytext=xy, ha=ha, va=va, fontsize=6.5, color=INK)
            placed.append((x, r.auc))
        ax.set_xlim(XMIN - 0.01, max(0.04, pts.gain.max() + 0.02) if len(pts) else 0.04)
        ax.set_xticks([-0.10, -0.05, 0.0])
        ax.set_title(b)
        ax.set_xlabel("AUC change after radiologist correction\n(all cases; > 0 = correction helps)")
    axs[0][0].set_ylabel("mean AUC on unseen datasets\n(BUS-BRA, BUS-UCLM, BUSB, BrEaST)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{f} CBM") for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], color=INK2, lw=0.9, ls="--", label="black box"),
                plt.Line2D([], [], color=INK2, lw=0.9, ls=":", label="multitask black box")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out, "fig9_tradeoff")
    print(t.round(3).to_string(index=False))


def fig10(cfg, out):
    CR = cfg["work_root"] / "cbm_runs"
    runs = [("DINOv2", "penalty 0.1", "dinov2_b_medsam_noexp_ic_u12_rp01"), ("DINOv2", "penalty 1.0", "dinov2_b_medsam_noexp_ic_u12_rp1"),
            ("BiomedCLIP", "penalty 0.1", "biomedclip_medsam_noexp_ic_u12_rp01"), ("BiomedCLIP", "penalty 1.0", "biomedclip_medsam_noexp_ic_u12_rp1")]
    T = []
    for b, pen, root in runs:
        f = CR / root / "analysis" / "8_hybrid_concept_share.csv"
        if f.exists():
            T.append(pd.read_csv(f).assign(backbone=b, penalty=pen))
    if not T:
        print("  fig10 skipped (no 8_hybrid_concept_share.csv)")
        return
    t = pd.concat(T)
    t.to_csv(out / "fig10_concept_share.csv", index=False)
    hs = f13.ds_sorted(t.heldout)
    keys = list(dict.fromkeys(zip(t.backbone, t.penalty)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    wbar = 0.8 / len(keys)
    for i, (b, pen) in enumerate(keys):
        d = t[(t.backbone == b) & (t.penalty == pen)].set_index("heldout").reindex(hs)
        a1.bar(np.arange(len(hs)) + (i - (len(keys) - 1) / 2) * wbar, d.var_share_concepts * 100, width=wbar,
               color=SERIES[i], label=f"{b}, {pen}")
    a1.set_xticks(range(len(hs)), hs, rotation=25, ha="right")
    a1.set_ylabel("share of logit variance\nfrom named concepts (%)")
    a1.set_ylim(0, 105)
    a1.grid(axis="x", visible=False)
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    # AUC of parts, averaged over the four datasets with masks
    parts = [("auc_concept_part", "concept part"), ("auc_residual_part", "residual part"), ("auc_full", "full model")]
    for i, (b, pen) in enumerate(keys):
        d = t[(t.backbone == b) & (t.penalty == pen) & t.heldout.isin(DS4)]
        a2.plot(range(3), [d[c].mean() for c, _ in parts], "-o", ms=4, color=SERIES[i], mec="white", mew=0.8)
    a2.set_xticks(range(3), [n for _, n in parts])
    a2.set_ylabel("mean AUC (BUS-BRA, BUS-UCLM, BUSB, BrEaST)")
    a2.set_xlim(-0.3, 2.3)
    fig.tight_layout()
    save(fig, out, "fig10_concept_share")


def fig11(cfg, out, run, held, n_each=3, seed=0):
    rd = cfg["work_root"] / "cbm_runs" / run / f"hyb_s0_{held}"
    if not (rd / "preds_test.csv").exists():
        print(f"  fig11 skipped ({rd} not found)")
        return
    meta = json.load(open(rd / "run.json"))
    w, b0, names = np.ravel(meta["diag_w"]), meta["diag_b"][0], meta["concepts"]
    p = pd.read_csv(rd / "preds_test.csv")
    C = p[[f"c{j}" for j in range(len(w))]].values
    contrib = C * w
    p["concept_part"] = contrib.sum(1) + b0
    p["residual"] = p.logit - p.concept_part
    p["prob"] = 1 / (1 + np.exp(-p.logit))
    p = p.merge(pd.read_csv(roi_manifest_path(cfg))[["uid", "roi_path"]], on="uid", how="left")
    sf = cfg["work_root"] / "medsam" / "sam_geometry.csv"
    if sf.exists():
        p = p.merge(pd.read_csv(sf)[["uid", "sam_mask_path"]], on="uid", how="left")
    rng = np.random.default_rng(seed)  # random cases (not hand-picked): n_each benign + n_each malignant
    pick = [i for lab in (1, 0) for i in rng.choice(np.flatnonzero(p.label.values == lab), n_each, replace=False)]
    p.iloc[pick][["uid", "label", "prob", "concept_part", "residual"]].to_csv(out / "fig11_hybrid_cases.csv", index=False)
    order = list(range(len(names)))
    labels = [NICE.get(n, n) for n in names] + ["Image residual"]
    lim = np.abs(np.c_[contrib[pick], p.residual.values[pick]]).max() * 1.1
    fig, axs = plt.subplots(len(pick), 2, figsize=(6.4, 1.8 * len(pick)), gridspec_kw=dict(width_ratios=[1, 2.6]))
    for row, i in zip(axs, pick):
        r = p.iloc[i]
        img = cv2.imread(str(r.roi_path), cv2.IMREAD_GRAYSCALE)
        row[0].imshow(img if img is not None else np.zeros((8, 8)), cmap="gray")
        mp = r.get("sam_mask_path")
        if isinstance(mp, str) and img is not None:
            mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mk is not None:
                if mk.shape != img.shape:
                    mk = cv2.resize(mk, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
                row[0].contour(mk > 127, levels=[0.5], colors=["#eda100"], linewidths=1)
        row[0].set_axis_off()
        lab = "malignant" if r.label == 1 else "benign"
        ok = (r.prob >= 0.5) == (r.label == 1)
        row[0].set_title(f"{lab}, {'correct' if ok else 'wrong'}\nP(malignant) = {r.prob:.2f}", fontsize=7.5, loc="left")
        v = np.r_[contrib[i][order], r.residual]
        y = np.arange(len(v))[::-1]
        cols = [POS if x > 0 else NEG for x in v[:-1]] + ["#9a9995"]
        row[1].barh(y, v, height=0.7, color=cols)
        row[1].axhline(y[-1] + 0.5, color=INK2, lw=0.6, ls=":")
        row[1].axvline(0, color=INK2, lw=0.8)
        row[1].set_xlim(-lim, lim)
        row[1].set_yticks(y, labels, fontsize=6.3)
        row[1].grid(axis="y", visible=False)
        cs = abs(r.concept_part - p.concept_part.mean()) / (abs(r.concept_part - p.concept_part.mean()) + abs(r.residual - p.residual.mean()) + 1e-9)
        row[1].set_title(f"concepts {r.concept_part:+.2f}, residual {r.residual:+.2f} (concept share {cs * 100:.0f}%)",
                         fontsize=7, loc="right", color=INK2)
    axs[-1][1].set_xlabel("contribution to malignancy logit  (← benign | malignant →)")
    fig.suptitle(f"Hybrid CBM ({run.split('_')[0]}), held-out {held}, randomly sampled cases (seed {seed}); outline = MedSAM mask",
                 fontsize=8, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig11_hybrid_cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--case_run", default="biomedclip_medsam_noexp_ic_u12_rp01")
    ap.add_argument("--case_held", default="BUS-BRA")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    out = cfg["work_root"] / "figures"
    out.mkdir(exist_ok=True)
    for name, fn in (("fig9", lambda: fig9(cfg, out)), ("fig10", lambda: fig10(cfg, out)),
                     ("fig11", lambda: fig11(cfg, out, a.case_run, a.case_held))):
        try:
            fn()
        except Exception as ex:
            print(f"  {name} failed: {ex!r}")
    print("done")
