"""Paper figures and tables from the saved results (CPU, < 1 min). Run after 09, 10 and 11.

Output: <work_root>/figures/
  table1_main.csv / .tex   AUC [95% CI] per held-out dataset for the proposed model, ablations and baselines
  fig2_cbm_vs_bb.pdf       paired AUC difference CBM - black box (95% CI), both backbones
  fig3_faithfulness.pdf    Spearman(predicted concept, radiologist-mask value) per concept and held-out set
  fig4_intervention.pdf    AUC change when correcting concepts (curves) + what each concept adds to correction
  fig5_influence.pdf       signed influence of each concept on the diagnosis (weight x spread)
  fig6_examples.pdf        example lesions: ROI + MedSAM outline + per-concept contributions to the logit
  fig7_sweep.pdf           accuracy vs. intervention-aware training rate p (all analysed _int* runs of --main)
  fig8_fewshot.pdf         few-shot adaptation on each held-out dataset, CBM vs black box (+ fig8_fewshot.csv)
Every figure is saved as .pdf (vector, for the paper) and .png (300 dpi, for slides).

Usage:  python 13_figures.py [--main dinov2_b_medsam_int50_noexp_ic_u12] [--second biomedclip_medsam_int50_noexp_ic_u12]
                             [--example_held BUS-BRA]
"""
import argparse
import json
import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import load_cfg, roi_manifest_path

# ----------------------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
POS, NEG, MID = "#e34948", "#2a78d6", "#f0efec"  # diverging: toward malignant / toward benign / neutral
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "lines.linewidth": 2, "pdf.fonttype": 42, "savefig.bbox": "tight"})
DS_ORDER = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST", "BUSI"]
NICE = {"g_solidity": "Solidity", "g_circularity": "Circularity", "g_ellipse_misfit": "Ellipse misfit",
        "g_radial_cv": "Radial spread", "g_spiculation": "Spiculation", "g_taller_than_wide": "Taller than wide",
        "i_hypoecho": "Darkness (cyst-like)", "i_heterogeneity": "Heterogeneity", "i_post_shadow": "Posterior shadowing",
        "i_margin_blur": "Margin blur", "i_halo": "Echogenic halo",
        "c_shape_irregular": "Irregular shape (expert)", "c_margin_noncirc": "Non-circ. margin (expert)"}
MODELS = [  # (main_table key pattern, display name, group)
    ("{main}:cbm", "Label-free CBM (ours), {mb}", "Proposed"),
    ("{second}:cbm", "Label-free CBM (ours), {sb}", "Proposed"),
    ("{mainbase}_u12:bb", "Black box, full fine-tuning, {mb}", "Black box"),
    ("{secondbase}_u12:bb", "Black box, full fine-tuning, {sb}", "Black box"),
    ("baseline:dinov2_b:probe_lr", "Linear probe (frozen), DINOv2", "Black box"),
    ("{mainbase}_int25_noexp_ic_u12:cbm", "CBM, 11 label-free concepts, p = 0.25, {mb}", "Ablation"),
    ("{mainbase}_int50_noexp_u12:cbm", "CBM, 6 geometry concepts only (no echo), p = 0.5, {mb}", "Ablation"),
    ("{secondbase}_int50_noexp_u12:cbm", "CBM, 6 geometry concepts only (no echo), p = 0.5, {sb}", "Ablation"),
    ("{mainbase}_int25_noexp_u12:cbm", "CBM, 6 geometry concepts only, p = 0.25, {mb}", "Ablation"),
    ("{mainbase}_int25_ic_u12:cbm", "CBM, 11 concepts + 2 expert, p = 0.25, {mb}", "Ablation"),
    ("{mainbase}_int25_u12:cbm", "CBM, 6 geometry + 2 expert, p = 0.25, {mb}", "Ablation"),
    ("{mainbase}:cbm", "CBM, last 2 blocks tuned, {mb}", "Ablation"),
    ("{mainbase}:bb", "Black box, last 2 blocks tuned, {mb}", "Ablation"),
    ("baseline:dinov2_b:posthoc_cbm_ic", "Post-hoc CBM, DINOv2", "Interpretable baseline"),
    ("baseline:biomedclip:posthoc_cbm_ic", "Post-hoc CBM, BiomedCLIP", "Interpretable baseline"),
    ("baseline:biomedclip:text_cbm", "Text label-free CBM, BiomedCLIP", "Interpretable baseline"),
    ("baseline:biomedclip:text_zeroshot", "Text zero-shot, BiomedCLIP", "Interpretable baseline"),
    ("baseline:-:geomint_lr_medsam", "Segment-then-measure, MedSAM mask*", "Needs mask at test"),
    ("baseline:-:geomint_lr_radiologist", "Segment-then-measure, radiologist mask*", "Needs mask at test"),
]


def bname(run):
    return "DINOv2" if run.startswith("dinov2") else "BiomedCLIP" if run.startswith("biomedclip") else run.split("_")[0]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}")


def ci_parse(s):
    v = re.findall(r"[-+]?\d*\.\d+", str(s))
    return (float(v[0]), float(v[1])) if len(v) == 2 else (np.nan, np.nan)


def ds_sorted(xs):
    xs = list(dict.fromkeys(xs))
    return [d for d in DS_ORDER if d in xs] + sorted(d for d in xs if d not in DS_ORDER)


def read(run_dir, name):
    f = run_dir / "analysis" / f"{name}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if name.startswith("4") and "mode" in d.columns:  # new 09 also reports seq / shuf: keep the proposed cbm
        d = d[d["mode"] == "cbm"]
    return d


# ----------------------------------------------------------------------------- table 1
def table1(cfg, out, a):
    f = cfg["work_root"] / "summary" / "main_table.csv"
    if not f.exists():
        print("  table1 skipped (run 11_summary.py)")
        return
    t = pd.read_csv(f)
    base = lambda r: re.sub(r"_(int\d+|noexp|ic|u\d+)", "", r)
    keys = dict(main=a.main, second=a.second, mainbase=base(a.main), secondbase=base(a.second),
                mb=bname(a.main), sb=bname(a.second))
    ds = ds_sorted(t.heldout)
    rows, seen = [], set()
    for pat, name, grp in MODELS:
        k = pat.format(**keys)
        d = t[t.model == k].set_index("heldout")
        if d.empty or k in seen:
            continue
        seen.add(k)
        r = {"group": grp, "model": name.format(**keys)}
        for h in ds:
            r[h] = f"{d.auc[h]:.3f} {d.ci[h]}" if h in d.index and d.auc[h] == d.auc[h] else "-"
        r["mean (all)"] = f"{d.auc.mean():.3f}" if d.auc.notna().all() and len(d) == len(ds) else "-"
        r["mean (w/o BUSI)"] = f"{d.auc.drop('BUSI', errors='ignore').mean():.3f}"
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "table1_main.csv", index=False)
    cols = ["model"] + ds + ["mean (all)", "mean (w/o BUSI)"]
    lines = [r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(["Model"] + ds + ["Mean", "Mean$^\\dagger$"]) + r" \\", r"\midrule"]
    for g, d in tab.groupby("group", sort=False):
        lines.append(rf"\multicolumn{{{len(cols)}}}{{l}}{{\textit{{{g}}}}} \\")
        for _, r in d.iterrows():
            cells = [r["model"]] + [str(r[h]).replace(" [", r" \scriptsize{[").replace("]", "]}") if r[h] != "-" else "-"
                                     for h in ds] + [r["mean (all)"], r["mean (w/o BUSI)"]]
            lines.append(" & ".join(cells).replace("_", r"\_") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% * needs a lesion mask at test time.  $\dagger$ mean without BUSI (no masks, near-duplicate images)."]
    (out / "table1_main.tex").write_text("\n".join(lines) + "\n")
    print("  table1_main (.csv, .tex)")


# ----------------------------------------------------------------------------- fig 2
def fig2(runs, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ds, n = None, len(runs)
    for i, (run, rd) in enumerate(runs):
        d = read(rd, "1b_paired_difference_vs_black_box")
        if d is None:
            continue
        d = d[d.comparison == "cbm-bb"].set_index("heldout")
        ds = ds or ds_sorted(d.index)
        y = np.arange(len(ds))[::-1] + (i - (n - 1) / 2) * 0.22
        for yy, h in zip(y, ds):
            if h not in d.index:
                continue
            lo, hi = ci_parse(d.ci[h])
            ax.plot([lo, hi], [yy, yy], color=SERIES[i], lw=2, solid_capstyle="round")
            ax.plot(d.delta[h], yy, "o", ms=6, color=SERIES[i], mec="white", mew=1, label=bname(run) if h == ds[0] else None)
    if ds is None:
        plt.close(fig)
        return
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(np.arange(len(ds))[::-1], ds)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC difference, CBM - black box (95% CI)\n> 0 favours the concept bottleneck")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, out, "fig2_cbm_vs_bb")


# ----------------------------------------------------------------------------- fig 3
def fig3(runs, out):
    data = [(run, read(rd, "3_concept_faithfulness")) for run, rd in runs]
    data = [(r, d[d["mode"] == "cbm"]) for r, d in data if d is not None and (d["mode"] == "cbm").any()]
    if not data:
        return
    cmap = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    rows = [c for c in NICE if any(c in set(d.concept) for _, d in data)]
    fig, axs = plt.subplots(1, len(data), figsize=(2.3 * len(data) + 1.2, 3.4), squeeze=False)
    for ax, (run, d) in zip(axs[0], data):
        p = d.pivot_table(index="concept", columns="heldout", values="spearman")
        p = p.reindex(rows)[ds_sorted(p.columns)]
        im = ax.imshow(p.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(p.values):
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else INK)
        ax.set_xticks(range(p.shape[1]), p.columns, rotation=35, ha="right")
        ax.set_yticks(range(p.shape[0]), [NICE[c] for c in p.index] if ax is axs[0][0] else [""] * p.shape[0])
        ax.set_title(bname(run))
        ax.grid(False)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Spearman with radiologist-mask value")
    cb.outline.set_visible(False)
    save(fig, out, "fig3_faithfulness")


# ----------------------------------------------------------------------------- fig 4
def fig4(runs, out):
    curves = [(run, read(rd, "4_concept_intervention")) for run, rd in runs]
    curves = [(r, d[d.source == "radiologist"]) for r, d in curves if d is not None]
    brk = [(run, read(rd, "4c_intervention_breakdown")) for run, rd in runs]
    brk = [(r, d) for r, d in brk if d is not None]
    if not curves:
        return
    fig = plt.figure(figsize=(7.0, 5.6 if brk else 2.7))
    gs = fig.add_gridspec(2 if brk else 1, len(curves), height_ratios=[1, 1.25] if brk else [1], hspace=0.45, wspace=0.25)
    ylim = [0, 0]
    axs = []
    for k, (run, d) in enumerate(curves):
        ax = fig.add_subplot(gs[0, k], sharey=axs[0] if axs else None)
        axs.append(ax)
        for i, h in enumerate(ds_sorted(d.heldout)):
            dh = d[d.heldout == h].sort_values("fraction")
            dv = dh.auc.values - dh.auc.values[0]
            ax.plot(dh.fraction * 100, dv, "-o", ms=4, color=SERIES[i], mec="white", mew=0.8, label=h)
            ylim = [min(ylim[0], dv.min()), max(ylim[1], dv.max())]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(bname(run))
        ax.set_xlabel("cases corrected (%)")
        if k == 0:
            ax.set_ylabel("AUC change")
            ax.legend(frameon=False, loc="lower left")
        else:
            ax.tick_params(labelleft=False)
    pad = 0.15 * (ylim[1] - ylim[0] + 1e-3)
    axs[0].set_ylim(ylim[0] - pad, ylim[1] + pad)
    if brk:
        ax = fig.add_subplot(gs[1, :])
        rows = []
        for run, d in brk:
            d = d[d.fraction == 1.0]
            allv = d[d.intervene == "all"].set_index("heldout").delta_auc
            for c in NICE:
                dc = d[d.intervene == f"all but {c}"].set_index("heldout").delta_auc
                if len(dc):
                    rows.append(dict(run=run, concept=c, contrib=(allv - dc).mean()))
        r = pd.DataFrame(rows)
        cs = [c for c in NICE if c in set(r.concept)]
        y = np.arange(len(cs))[::-1]
        for i, (run, _) in enumerate(brk):
            v = r[r.run == run].set_index("concept").contrib.reindex(cs).values
            ax.barh(y + (i - (len(brk) - 1) / 2) * 0.38, v, height=0.36, color=SERIES[i], label=bname(run))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [NICE[c] for c in cs])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC gained by also correcting this concept (correct all minus correct all-but-this; 100% of cases)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, out, "fig4_intervention")


# ----------------------------------------------------------------------------- fig 5
def influence(rd):
    """Signed influence (weight x SD of predicted concept) of EVERY bottleneck unit, incl. expert concepts,
    averaged over seeds and held-out sets, straight from the run folders."""
    rows = []
    for s in sorted(rd.glob("cbm_s*")):
        if not (s / "run.json").exists() or not (s / "preds_test.csv").exists():
            continue
        meta = json.load(open(s / "run.json"))
        w = np.ravel(meta["diag_w"])
        p = pd.read_csv(s / "preds_test.csv")
        for j, c in enumerate(meta["concepts"]):
            rows.append(dict(concept=c, signed=w[j] * p[f"c{j}"].std()))
    return pd.DataFrame(rows).groupby("concept").signed.mean() if rows else None


def fig5(runs, out):
    data = [(run, influence(rd)) for run, rd in runs]
    data = [(r, d) for r, d in data if d is not None]
    if not data:
        return
    for run, d in data:
        ex = d[d.index.str.startswith("c_")].abs().sum()
        print(f"    {bname(run)}: expert concepts carry {100 * ex / d.abs().sum():.0f}% of total |influence|")
    m = pd.concat({run: d for run, d in data}, axis=1)
    cs = m[data[0][0]].abs().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    y = np.arange(len(cs))
    for i, (run, _) in enumerate(data):
        ax.barh(y + (i - (len(data) - 1) / 2) * 0.38, m.loc[cs, run].values, height=0.36, color=SERIES[i], label=bname(run))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [NICE.get(c, c) for c in cs])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("influence on malignancy logit (weight x SD of predicted concept)\n> 0: a higher value pushes toward malignant")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, out, "fig5_influence")


# ----------------------------------------------------------------------------- fig 6
def fig6(cfg, run, rd, held, out, n_each=2):
    seeds = sorted(rd.glob(f"cbm_s*_{held}"))
    if not seeds:
        print(f"  fig6 skipped (no cbm runs for {held})")
        return
    P, W, B = [], [], []
    for s in seeds:
        meta = json.load(open(s / "run.json"))
        w = np.ravel(meta["diag_w"])
        p = pd.read_csv(s / "preds_test.csv")
        C = p[[f"c{j}" for j in range(len(w))]].values
        P.append(p), W.append(C * w), B.append(meta["diag_b"][0])
        names = meta["concepts"]
    p = P[0][["uid", "label"]].copy()
    contrib = np.mean(W, 0)
    p["logit"] = contrib.sum(1) + np.mean(B)
    p["prob"] = 1 / (1 + np.exp(-p.logit))
    man = pd.read_csv(roi_manifest_path(cfg))[["uid", "roi_path"]]
    p = p.merge(man, on="uid", how="left")
    sf = cfg["work_root"] / "medsam" / "sam_geometry.csv"
    if sf.exists():
        sm = pd.read_csv(sf)
        if "sam_mask_path" in sm:
            p = p.merge(sm[["uid", "sam_mask_path"]], on="uid", how="left")
    thr = 0.5
    pick = []
    for lab, want_correct, tag in [(1, True, "malignant, correct"), (0, True, "benign, correct"),
                                   (1, False, "malignant, missed"), (0, False, "benign, false alarm")]:
        correct = (p.prob >= thr) == (lab == 1)
        d = p[(p.label == lab) & (correct == want_correct)]
        d = d.sort_values("prob", ascending=(lab == 0) == want_correct)
        if want_correct:  # typical correct cases at spread-out confidence (quantiles), not the most extreme ones
            qs = [(j + 1) / (n_each + 1) for j in range(n_each)]
            idx = list(dict.fromkeys(d.index[min(len(d) - 1, int(q * len(d)))] for q in qs)) if len(d) else []
        else:  # most confident errors
            idx = d.index[:1]
        pick += [(i, tag) for i in idx]
    if not pick:
        return
    fig, axs = plt.subplots(len(pick), 2, figsize=(6.4, 1.75 * len(pick)), gridspec_kw=dict(width_ratios=[1, 2.6]))
    axs = np.atleast_2d(axs)
    order = list(range(len(names)))
    lim = np.abs(contrib[[i for i, _ in pick]]).max() * 1.1
    for row, (i, tag) in zip(axs, pick):
        r = p.loc[i]
        img = cv2.imread(str(r.roi_path), cv2.IMREAD_GRAYSCALE)
        row[0].imshow(img if img is not None else np.zeros((8, 8)), cmap="gray")
        mp = r.get("sam_mask_path")
        if isinstance(mp, str) and img is not None:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != img.shape:
                    m = cv2.resize(m, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
                row[0].contour(m > 127 if m.max() > 1 else m > 0, levels=[0.5], colors=["#eda100"], linewidths=1)
        row[0].set_axis_off()
        row[0].set_title(f"{tag}\nP(malignant) = {r.prob:.2f}", fontsize=7.5, loc="left")
        v = contrib[i]
        y = np.arange(len(order))[::-1]
        row[1].barh(y, v[order], height=0.7, color=[POS if x > 0 else NEG for x in v[order]])
        row[1].axvline(0, color=INK2, lw=0.8)
        row[1].set_xlim(-lim, lim)
        row[1].set_yticks(y, [NICE.get(names[j], names[j]) for j in order], fontsize=6.5)
        row[1].grid(axis="y", visible=False)
    axs[-1][1].set_xlabel("contribution to malignancy logit  (← benign | malignant →)")
    fig.suptitle(f"{bname(run)} label-free CBM, held-out {held} (never seen in training); outline = MedSAM mask",
                 fontsize=8, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out, "fig6_examples")


# ----------------------------------------------------------------------------- fig 7
def fig7(cfg, main, out):
    """Accuracy vs. safe intervention across --int_p (same backbone / concept set as --main)."""
    cr = cfg["work_root"] / "cbm_runs"
    pat = re.sub(r"_int\d+", "_int*", main) if "_int" in main else None
    dirs = sorted(cr.glob(pat), key=lambda d: int(re.search(r"_int(\d+)", d.name).group(1))) if pat else []
    rows = []
    for d in dirs:
        za, iv = read(d, "1_zero_shot_auc"), read(d, "4_concept_intervention")
        if za is None or iv is None:
            print(f"    fig7: {d.name} has no 09 analysis yet, skipped")
            continue
        p = int(re.search(r"_int(\d+)", d.name).group(1))
        za = za[za["mode"] == "cbm"]
        r = dict(int_p=p, auc=za.ens_auc.mean())
        bbf = cfg["work_root"] / "cbm_runs" / re.sub(r"_(int\d+|noexp|ic)", "", d.name) / "analysis" / "1_zero_shot_auc.csv"
        if bbf.exists():
            bb = pd.read_csv(bbf)
            r["bb_auc"] = bb[bb["mode"] == "bb"].ens_auc.mean()
        for src in ("radiologist", "train_label"):
            s = iv[iv.source == src].pivot_table(index="heldout", columns="fraction", values="auc")
            if len(s) and 1.0 in s and 0.0 in s:
                r[f"gain_{src}"] = (s[1.0] - s[0.0]).mean()
                r[f"gain50_{src}"] = (s[0.5] - s[0.0]).mean() if 0.5 in s else np.nan
        rows.append(r)
    if len(rows) < 2:
        print("  fig7 skipped (needs >= 2 analysed int_p settings)")
        return
    t = pd.DataFrame(rows).sort_values("int_p")
    t.to_csv(out / "fig7_sweep.csv", index=False)
    x = t.int_p / 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a1.plot(x, t.auc, "-o", color=SERIES[0], ms=6, mec="white", mew=1, label="label-free CBM")
    if "bb_auc" in t and t.bb_auc.notna().any():
        a1.axhline(t.bb_auc.dropna().iloc[0], color=INK2, lw=1, ls="--")
        a1.text(x.iloc[-1], t.bb_auc.dropna().iloc[0] - 0.003, "black box, full fine-tuning", ha="right", va="top",
                color=INK2, fontsize=7)
    a1.set_xlabel("intervention-aware training rate $p$")
    a1.set_ylabel("mean AUC (5 held-out sets)")
    a1.set_title("Accuracy")
    for i, (src, lab) in enumerate([("radiologist", "radiologist-mask values"), ("train_label", "MedSAM (training) values")]):
        if f"gain_{src}" in t:
            a2.plot(x, t[f"gain_{src}"], "-o", color=SERIES[i], ms=6, mec="white", mew=1, label=lab)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel("intervention-aware training rate $p$")
    a2.set_ylabel("AUC change, all cases corrected")
    a2.set_title("Correcting concepts")
    a2.legend(frameon=False, loc="lower right")
    for ax in (a1, a2):
        ax.set_xticks(x, [f"{v:g}" for v in x])
    fig.tight_layout()
    save(fig, out, "fig7_sweep")
    print(t.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- fig 8
FEW = [  # (mode, method, label, colour index, linestyle)
    ("cbm", "kshot_concepts", "CBM: 11 concepts, k-shot LR", 0, "-"),
    ("cbm", "fused", "CBM: zero-shot + k-shot fused", 1, "-"),
    ("bb", "kshot_emb", "black box: features, k-shot LR", 2, "-"),
    ("bb", "fused", "black box: zero-shot + k-shot fused", 3, "-"),
]


def fig8(runs, out):
    """Few-shot adaptation on each held-out dataset (09 section 5), one row per backbone."""
    tabs = [(run, read(rd, "5_few_shot_auc_on_target")) for run, rd in runs]
    tabs = [(r, d) for r, d in tabs if d is not None and len(d)]
    if not tabs:
        print("  fig8 skipped (no 5_few_shot_auc_on_target.csv)")
        return
    hs = ds_sorted(pd.concat([d.heldout for _, d in tabs]))
    fig, axs = plt.subplots(len(tabs), len(hs), figsize=(7.0, 1.9 * len(tabs) + 0.5), squeeze=False, sharex=True)
    summ = []
    for ri, (row, (run, d)) in enumerate(zip(axs, tabs)):
        g = d.groupby(["heldout", "mode", "k", "method"]).auc.mean().reset_index()
        for ax, h in zip(row, hs):
            gh = g[g.heldout == h]
            for mode, meth, lab, ci, ls in FEW:
                s = gh[(gh["mode"] == mode) & (gh.method == meth)].sort_values("k")
                if len(s):
                    ax.plot(s.k, s.auc, ls, marker="o", ms=3, color=SERIES[ci], mec="white", mew=0.6, label=lab)
                    summ += [dict(backbone=bname(run), heldout=h, mode=mode, method=meth, k=k, auc=v) for k, v in zip(s.k, s.auc)]
            z = gh[(gh["mode"] == "cbm") & (gh.method == "src")].auc.mean()
            if z == z:
                ax.axhline(z, color=INK2, lw=0.9, ls="--", label="CBM zero-shot (no target labels)")
            ax.set_xscale("log", base=2)
            ks = sorted(gh.k.unique())
            ax.set_xticks(ks, [f"{v:g}" for v in ks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.tick_params(labelsize=6.5)
            if h == hs[0]:
                ax.set_ylabel(f"{bname(run)}\nAUC on target")
            if ri == 0:
                ax.set_title(h, fontsize=8)
            if ri == len(tabs) - 1:
                ax.set_xlabel("k labelled cases / class", fontsize=7)
    hdl, lab = axs[0][0].get_legend_handles_labels()
    fig.legend(hdl, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig8_fewshot")
    if summ:
        s = pd.DataFrame(summ)
        s.to_csv(out / "fig8_fewshot.csv", index=False)
        print(s.pivot_table(index=["backbone", "k"], columns=["mode", "method"], values="auc", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--main", default="dinov2_b_medsam_int50_noexp_ic_u12")
    ap.add_argument("--second", default="biomedclip_medsam_int50_noexp_ic_u12")
    ap.add_argument("--example_held", default="BUS-BRA")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    out = cfg["work_root"] / "figures"
    out.mkdir(exist_ok=True)
    cr = cfg["work_root"] / "cbm_runs"
    runs = [(r, cr / r) for r in (a.main, a.second) if (cr / r).exists()]
    print(f"runs: {[r for r, _ in runs]}\nwriting to {out}")
    table1(cfg, out, a)
    for f in (fig2, fig3, fig4, fig5, fig8):
        try:
            f(runs, out)
        except Exception as ex:
            print(f"  {f.__name__} failed: {ex!r}")
    if runs:
        try:
            fig6(cfg, runs[0][0], runs[0][1], a.example_held, out)
        except Exception as ex:
            print(f"  fig6 failed: {ex!r}")
    try:
        fig7(cfg, a.main, out)
    except Exception as ex:
        print(f"  fig7 failed: {ex!r}")
    print("done")
