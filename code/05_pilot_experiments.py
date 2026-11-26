"""Step 5: go/no-go pilot for BI-RADS-Bridge (uses cached features only; CPU is fine).

EXP 1  Concept transfer MG -> US (no US labels used)
   Train BI-RADS concept probes (shape irregular, margin non-circumscribed, margin spiculated) on CBIS-DDSM masses.
   Evaluate: internal concept AUC (CBIS test), external MG concept AUC (MIAS), and on each US dataset:
     - AUC of each transferred concept score for malignancy (zero-shot)
     - AUC of a MG *malignancy* probe applied directly to US (label transfer baseline)
     - Spearman correlation of concept scores with mask-geometry descriptors (do the concepts mean the same thing?)
     - VLM text zero-shot concept scores (BiomedCLIP) and geometry descriptors as references
EXP 2  Few-shot US diagnosis (k labelled lesions per class, many random episodes, patient-disjoint)
     feat_lr      logistic regression on frozen features           (strong standard baseline)
     feat_proto   cosine prototype classifier
     concept_lr   LR on transferred concept scores only            (interpretable, 3-7 inputs)
     fused        z(feat_lr) + z(concept_lr) late fusion           (BI-RADS-Bridge proxy)
     geom_lr      LR on mask-geometry descriptors (needs masks; reference)
GO criterion: fused > feat_lr at k in {4, 8, 16} on >= 2 of 3 US datasets for at least one backbone.

Usage:  python 05_pilot_experiments.py [--backbones biomedclip dinov2_b] [--episodes 100]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import GEOM_COLS, load_cfg, roi_manifest_path

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CONCEPTS = ["shape_irregular", "margin_noncirc", "margin_spiculated"]
GEOM_SIGN = {"g_solidity": -1, "g_circularity": -1}  # flip so that higher = more suspicious


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ok = ~np.isnan(s) & ~np.isnan(y)
    return roc_auc_score(y[ok], s[ok]) if ok.sum() > 5 and len(np.unique(y[ok])) == 2 else np.nan


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def zs(x, ref=None):
    ref = x if ref is None else ref
    return (x - np.nanmean(ref, 0)) / (np.nanstd(ref, 0) + 1e-8)


def load_feats(cfg, name, df):
    z = np.load(cfg["work_root"] / "features" / f"{name}.npz", allow_pickle=True)  # our own file
    pos = {u: i for i, u in enumerate(z["uid"])}
    idx = np.array([pos[u] for u in df.uid])
    text = {k: z[k] for k in z.files if k.startswith("text_")}
    return z["feat"][idx], text


def concept_scores(F, df, src, text):
    """Return dict name -> score vector for all rows (MG-trained probes + optional VLM text concepts)."""
    S, probes_info = {}, []
    tr = src & (df.official_split == "train").values
    for c in CONCEPTS:
        y = df[f"c_{c}"].values
        m = tr & ~np.isnan(y)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        clf = lr(0.5).fit(F[m], y[m].astype(int))
        S[f"mg_{c}"] = clf.decision_function(F)
        probes_info.append((c, int(m.sum())))
    clf = lr(0.5).fit(F[tr], df.label.values[tr].astype(int))
    S["mg_malignancy"] = clf.decision_function(F)
    for c in CONCEPTS + ["malignant"]:
        for mod in ["MG", "US"]:
            kp, kn = f"text_{mod}_{c}_pos", f"text_{mod}_{c}_neg"
            if kp in text:
                s = F @ text[kp] - F @ text[kn]
                S.setdefault(f"txt_{c}", np.full(len(df), np.nan))
                S[f"txt_{c}"][df.modality.values == mod] = s[df.modality.values == mod]
    return S, probes_info


def exp1(df, S, backbone):
    rows = []
    src = (df.dataset == "CBIS-DDSM").values
    te = src & (df.official_split == "test").values
    mi = (df.dataset == "MIAS").values
    for c in CONCEPTS:
        if f"mg_{c}" in S:
            yc = df[f"c_{c}"].values
            rows.append(dict(backbone=backbone, dataset="CBIS-DDSM(test)", score=f"mg_{c}", target=f"concept:{c}",
                             value=auc(yc[te], S[f"mg_{c}"][te])))
            rows.append(dict(backbone=backbone, dataset="MIAS", score=f"mg_{c}", target=f"concept:{c}",
                             value=auc(yc[mi], S[f"mg_{c}"][mi])))
    for ds in df.dataset.unique():
        m = (df.dataset == ds).values & ~(src & (df.official_split == "train").values)
        y = df.label.values[m]
        for k, s in S.items():
            rows.append(dict(backbone=backbone, dataset=ds, score=k, target="malignancy", value=auc(y, s[m])))
        for g in GEOM_COLS:
            gv = df[g].values[m] * GEOM_SIGN.get(g, 1)
            rows.append(dict(backbone=backbone, dataset=ds, score=g, target="malignancy", value=auc(y, gv)))
            for k in S:
                if k.startswith("mg_") and k != "mg_malignancy":
                    ok = ~np.isnan(gv)
                    if ok.sum() > 10:
                        rows.append(dict(backbone=backbone, dataset=ds, score=k, target=f"spearman:{g}",
                                         value=spearmanr(S[k][m][ok], gv[ok]).statistic))
    return rows


def episode_split(rng, y, pid, k):
    sup = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        if len(idx) < k + 5:
            return None, None
        sup += list(rng.choice(idx, k, replace=False))
    sup = np.array(sup)
    test = np.flatnonzero(~np.isin(pid, pid[sup]))  # patient-disjoint
    return sup, test


def exp2(df, F, S, backbone, targets, shots, episodes, seed):
    rows = []
    cnames = [k for k in S if k.startswith("mg_") and k != "mg_malignancy"] + [k for k in S if k.startswith("txt_")]
    for ds in targets:
        m = (df.dataset == ds).values
        if m.sum() < 30:
            continue
        X, y, pid = F[m], df.label.values[m].astype(int), df.patient_id.values[m]
        C = np.column_stack([zs(S[k][m]) for k in cnames])  # label-free standardisation on target pool
        C = np.nan_to_num(C)
        G = np.nan_to_num(zs(df[GEOM_COLS].values[m].astype(float)))
        has_geom = not np.isnan(df[GEOM_COLS].values[m].astype(float)).all()
        # k = 0: zero-shot transfer
        rows.append(dict(backbone=backbone, dataset=ds, k=0, episode=0, method="zs_mg_malignancy",
                         auc=auc(y, S["mg_malignancy"][m])))
        rows.append(dict(backbone=backbone, dataset=ds, k=0, episode=0, method="zs_concept_mean",
                         auc=auc(y, C[:, : sum(k.startswith("mg_") for k in cnames)].mean(1))))
        rng = np.random.default_rng(seed)
        for k in shots:
            for e in range(episodes):
                sup, test = episode_split(rng, y, pid, k)
                if sup is None:
                    break
                ys, yt = y[sup], y[test]
                res = {}
                f = lr(1.0).fit(X[sup], ys)
                res["feat_lr"] = f.decision_function(X[test])
                p1, p0 = X[sup][ys == 1].mean(0), X[sup][ys == 0].mean(0)
                res["feat_proto"] = X[test] @ (p1 / np.linalg.norm(p1) - p0 / np.linalg.norm(p0))
                c = lr(1.0).fit(C[sup], ys)
                res["concept_lr"] = c.decision_function(C[test])
                res["fused"] = zs(res["feat_lr"]) + zs(res["concept_lr"])
                if has_geom:
                    res["geom_lr"] = lr(1.0).fit(G[sup], ys).decision_function(G[test])
                for meth, s in res.items():
                    rows.append(dict(backbone=backbone, dataset=ds, k=k, episode=e, method=meth, auc=auc(yt, s)))
        print(f"   {backbone:12s} {ds:9s} done")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backbones", nargs="*")
    ap.add_argument("--episodes", type=int)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    P = cfg["pilot"]
    episodes = a.episodes or P["episodes"]
    out = cfg["work_root"] / "results"
    out.mkdir(exist_ok=True)

    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[(df.lesion_type == "mass") & df.label.notna()].reset_index(drop=True)
    for c in CONCEPTS:
        df[f"c_{c}"] = df[f"c_{c}"].astype(float)
    # CBIS-DDSM official split has patients in both train and test -> drop them from test (leakage fix)
    src = df.dataset == P["source_dataset"]
    trp = set(df.patient_id[src & (df.official_split == "train")])
    leak = src & (df.official_split == "test") & df.patient_id.isin(trp)
    print(f"CBIS-DDSM: removing {leak.sum()} test lesions whose patient is also in train")
    df = df[~leak].reset_index(drop=True)
    print(df.groupby("dataset").label.agg(["count", "sum"]).rename(columns={"sum": "malignant"}))

    e1, e2 = [], []
    for bb in a.backbones or cfg["features"]["backbones"]:
        try:
            F, text = load_feats(cfg, bb, df)
        except FileNotFoundError:
            print(f"!! no features for {bb}, skipped")
            continue
        S, info = concept_scores(F, df, (df.dataset == P["source_dataset"]).values, text)
        print(f"[{bb}] concept probes trained on CBIS-DDSM: {info}")
        e1 += exp1(df, S, bb)
        e2 += exp2(df, F, S, bb, P["us_targets"], P["shots"], episodes, P["seed"])

    e1 = pd.DataFrame(e1)
    e1.to_csv(out / "exp1_concept_transfer.csv", index=False)
    e2 = pd.DataFrame(e2)
    e2.to_csv(out / "exp2_fewshot_episodes.csv", index=False)

    summ = (e2.groupby(["backbone", "dataset", "k", "method"]).auc
            .agg(mean="mean", std="std", n="count").reset_index())
    summ["ci95"] = 1.96 * summ["std"] / np.sqrt(summ["n"].clip(lower=1))
    summ.to_csv(out / "exp2_summary.csv", index=False)

    # ------------------------------------------------------------- verdict
    lines = ["EXP 1: zero-shot malignancy AUC on US from MG-trained scores (no US labels)"]
    t1 = e1[(e1.target == "malignancy") & e1.dataset.isin(P["us_targets"])]
    lines.append(t1.pivot_table(index=["backbone", "score"], columns="dataset", values="value").round(3).to_string())
    lines.append("\nEXP 1: MG concept AUC (internal CBIS test / external MIAS)")
    t2 = e1[e1.target.str.startswith("concept")]
    lines.append(t2.pivot_table(index=["backbone", "score"], columns="dataset", values="value").round(3).to_string())
    lines.append("\nEXP 2: few-shot AUC (mean over episodes)")
    lines.append(summ.pivot_table(index=["backbone", "dataset", "k"], columns="method", values="mean")
                 .round(3).to_string())
    lines.append("\nGO / NO-GO (fused - feat_lr, paired over episodes, k in 4/8/16)")
    piv = e2[e2.k.isin([4, 8, 16])].pivot_table(index=["backbone", "dataset", "k", "episode"], columns="method",
                                                  values="auc")
    piv["delta"] = piv["fused"] - piv["feat_lr"]
    g = piv.groupby(["backbone", "dataset"]).delta.agg(mean_delta="mean", frac_positive=lambda d: (d > 0).mean())
    lines.append(g.round(4).to_string())
    wins = g[(g.mean_delta > 0) & (g.frac_positive > 0.6)].reset_index().groupby("backbone").dataset.nunique()
    go = bool((wins >= 2).any())
    lines.append(f"\nVERDICT: {'GO' if go else 'NO-GO'}  (datasets won per backbone: {wins.to_dict()})")
    txt = "\n".join(lines)
    (out / "verdict.txt").write_text(txt)
    print(txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for bb in summ.backbone.unique():
            sub = summ[(summ.backbone == bb) & (summ.k > 0)]
            dss = sub.dataset.unique()
            fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 4), squeeze=False)
            for ax, ds in zip(axes[0], dss):
                for meth, d in sub[sub.dataset == ds].groupby("method"):
                    ax.errorbar(d.k, d["mean"], yerr=d.ci95, marker="o", capsize=3, label=meth)
                ax.set_xscale("log", base=2)
                ax.set_title(f"{bb} | {ds}")
                ax.set_xlabel("k (labelled lesions per class)")
                ax.set_ylabel("AUROC")
                ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"fewshot_{bb}.png", dpi=150)
            plt.close(fig)
    except Exception as ex:
        print("plotting skipped:", ex)
    print(f"\nresults in {out}")
