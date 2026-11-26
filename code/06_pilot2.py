"""Pilot 2: learn ultrasound BI-RADS concepts from FREE mask-derived labels, then diagnose from the image alone.

Uses the cached features from step 4 (CPU only). Source = an US dataset WITH masks (BUS-BRA, BUSB);
targets = the other US datasets. BUSI has no masks at all, so it is a true image-only test.

TEST A  Concept bottleneck from mask geometry
   features --Ridge--> 6 geometry concepts (trained on source masks)  --LR--> malignancy (trained on source labels)
   Compared at the target with:
     src_bb       black box: features --LR--> malignancy, trained on the same source labels (fair baseline)
     src_cbm      the concept bottleneck above (k = 0, no target labels)
     feat_lr      k-shot target-only LR on features
     cbm_k        k-shot LR on the 6 predicted concepts
     fused_bb     z(feat_lr) + z(src_bb)       few-shot + black-box prior
     fused_cbm    z(feat_lr) + z(src_cbm)      few-shot + concept prior   <- BI-RADS-Bridge (US-only form)
   Also reports concept fidelity: Spearman(predicted concept, true mask geometry) on targets that have masks.

TEST B  Do mammography concepts help when combined with ultrasound weak concepts?
   Two shared BI-RADS concepts: shape_irregular, margin_noncirc.
     MG labels : CBIS-DDSM expert labels (train split)
     US labels : mask-geometry proxies on the source (top 40% -> 1, bottom 40% -> 0, middle dropped)
   Probes per concept:  mg (MG only) | us (US only) | joint (MG+US, modality-balanced) |
                        text (BiomedCLIP prompts, no training) | joint+text
   Scored by: malignancy AUC of a 2-concept bottleneck (k = 0 from source labels, and k-shot), concept AUC vs target
   proxies, and MG concept AUC on CBIS test / MIAS (does joint training hurt mammography?).

--geom sam  : use geometry from automatic SAM/MedSAM masks (07_sam_masks.py) instead of radiologist masks (TEST C).

Usage:  python 06_pilot2.py [--backbones biomedclip] [--geom gt|sam] [--episodes 100]
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from common import GEOM_COLS, load_cfg, roi_manifest_path

p1 = importlib.import_module("05_pilot_experiments")
auc, lr, zs, episode_split, load_feats = p1.auc, p1.lr, p1.zs, p1.episode_split, p1.load_feats
warnings.filterwarnings("ignore")

SHARED = ["shape_irregular", "margin_noncirc"]


def us_proxies(G: pd.DataFrame) -> pd.DataFrame:
    """Continuous US proxies for the shared BI-RADS concepts, higher = more suspicious (z-scored within dataset)."""
    z = lambda c: (G[c] - G[c].mean()) / (G[c].std() + 1e-8)
    return pd.DataFrame({"shape_irregular": (z("g_ellipse_misfit") - z("g_solidity")) / 2,
                         "margin_noncirc": z("g_spiculation")}, index=G.index)


def binarise(s: pd.Series, q=0.4):
    lo, hi = s.quantile(q), s.quantile(1 - q)
    out = pd.Series(np.nan, index=s.index)
    out[s <= lo], out[s >= hi] = 0, 1
    return out


def episodes_engine(y, pid, shots, episodes, seed, fit_mats, fixed, combos, tag):
    """fit_mats: name -> X (LR fitted on the k-shot support); fixed: name -> score (no fitting);
    combos: (name, [parts]) -> z-sum of part scores on the test set."""
    rows = [dict(**tag, k=0, episode=0, method=n, auc=auc(y, s)) for n, s in fixed.items()]
    rng = np.random.default_rng(seed)
    for k in shots:
        for e in range(episodes):
            sup, test = episode_split(rng, y, pid, k)
            if sup is None:
                break
            res = {n: lr(1.0).fit(X[sup], y[sup]).decision_function(X[test]) for n, X in fit_mats.items()}
            res.update({n: s[test] for n, s in fixed.items()})
            for n, parts in combos:
                res[n] = sum(zs(res[p]) for p in parts)
            rows += [dict(**tag, k=k, episode=e, method=n, auc=auc(y[test], s)) for n, s in res.items()]
    return rows


# ============================================================================ TEST A
def test_a(df, F, bb, src, targets, P, episodes):
    zero, fid, few = [], [], []
    s = ((df.dataset == src) & df[GEOM_COLS].notna().all(1)).values
    if s.sum() < 50:
        print(f"   A: source {src} has too few masks, skipped")
        return zero, fid, few
    G = df.loc[s, GEOM_COLS].values.astype(float)
    mu, sd = G.mean(0), G.std(0) + 1e-8
    G = (G - mu) / sd
    ys, gs = df.label.values[s].astype(int), df.patient_id.values[s]
    # out-of-fold concept predictions on the source, so the concept->label layer sees realistic (noisy) concepts
    oof = np.zeros_like(G)
    for tr, te in GroupKFold(5).split(F[s], ys, gs):
        oof[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(F[s][tr], G[tr]).predict(F[s][te])
    reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(F[s], G)
    Ghat = reg.predict(F)
    cbm = lr(1.0).fit(oof, ys)
    blk = lr(0.5).fit(F[s], ys)
    geo = lr(1.0).fit(G, ys)  # oracle: true geometry -> label
    in_r2 = [spearmanr(oof[:, j], G[:, j]).statistic for j in range(len(GEOM_COLS))]
    for j, c in enumerate(GEOM_COLS):
        fid.append(dict(backbone=bb, source=src, target=f"{src}(oof)", concept=c, spearman=in_r2[j]))
    for t in targets:
        m = (df.dataset == t).values
        if t == src or m.sum() < 30:
            continue
        y, pid = df.label.values[m].astype(int), df.patient_id.values[m]
        tag = dict(backbone=bb, source=src, target=t)
        src_cbm, src_bb = cbm.decision_function(Ghat[m]), blk.decision_function(F[m])
        zero += [dict(**tag, method="src_bb", auc=auc(y, src_bb)), dict(**tag, method="src_cbm", auc=auc(y, src_cbm))]
        Gt = df.loc[m, GEOM_COLS].values.astype(float)
        if not np.isnan(Gt).all():
            ok = ~np.isnan(Gt).any(1)
            zero.append(dict(**tag, method="oracle_mask_geometry", auc=auc(y[ok], geo.decision_function(((Gt - mu) / sd)[ok]))))
            for j, c in enumerate(GEOM_COLS):
                fid.append(dict(backbone=bb, source=src, target=t, concept=c,
                                spearman=spearmanr(Ghat[m][ok, j], Gt[ok, j]).statistic))
        few += episodes_engine(
            y, pid, P["shots"], episodes, P["seed"],
            fit_mats={"feat_lr": F[m], "cbm_k": zs(Ghat[m])},
            fixed={"src_bb": src_bb, "src_cbm": src_cbm},
            combos=[("fused_bb", ["feat_lr", "src_bb"]), ("fused_cbm", ["feat_lr", "src_cbm"]),
                    ("fused_cbm_bb", ["feat_lr", "src_cbm", "src_bb"])],
            tag=tag)
    return zero, fid, few


# ============================================================================ TEST B
def test_b(df, F, text, bb, src, targets, P, episodes):
    zero, mgrows, few = [], [], []
    us_src = ((df.dataset == src) & df[GEOM_COLS].notna().all(1)).values
    mg_tr = ((df.dataset == "CBIS-DDSM") & (df.official_split == "train")).values
    if us_src.sum() < 50:
        return zero, mgrows, few
    prox = pd.DataFrame(index=df.index, columns=SHARED, dtype=float)
    for d in df.dataset.unique():  # proxies per US dataset with masks (z within dataset, label-free)
        md = ((df.dataset == d) & (df.modality == "US") & df[GEOM_COLS].notna().all(1))
        if md.sum() > 10:
            prox.loc[md] = us_proxies(df.loc[md, GEOM_COLS]).values
    scores = {}  # probe name -> concept -> score for all rows
    for c in SHARED:
        y_mg = df[f"c_{c}"].values.astype(float)
        y_us = binarise(prox.loc[us_src, c]).reindex(df.index).values
        mg_m, us_m = mg_tr & ~np.isnan(y_mg), us_src & ~np.isnan(y_us)
        X_j = np.vstack([F[mg_m], F[us_m]])
        y_j = np.concatenate([y_mg[mg_m], y_us[us_m]]).astype(int)
        w_j = np.concatenate([np.full(mg_m.sum(), 0.5 / mg_m.sum()), np.full(us_m.sum(), 0.5 / us_m.sum())])
        w_j *= len(w_j)
        probes = {"mg": lr(0.5).fit(F[mg_m], y_mg[mg_m].astype(int)),
                  "us": lr(0.5).fit(F[us_m], y_us[us_m].astype(int)),
                  "joint": lr(0.5).fit(X_j, y_j, sample_weight=w_j)}
        for n, pr in probes.items():
            scores.setdefault(n, {})[c] = pr.decision_function(F)
        if f"text_US_{c}_pos" in text:
            t = np.zeros(len(df))
            for mod in ["MG", "US"]:
                mm = (df.modality == mod).values
                t[mm] = F[mm] @ (text[f"text_{mod}_{c}_pos"] - text[f"text_{mod}_{c}_neg"])
            scores.setdefault("text", {})[c] = t
            j = np.zeros(len(df))
            for d in df.dataset.unique():  # label-free z within each dataset, then add
                md = (df.dataset == d).values
                j[md] = zs(scores["joint"][c][md]) + zs(t[md])
            scores.setdefault("joint+text", {})[c] = j
    # MG side check
    te = ((df.dataset == "CBIS-DDSM") & (df.official_split == "test")).values
    mi = (df.dataset == "MIAS").values
    for n, sc in scores.items():
        for c in SHARED:
            yc = df[f"c_{c}"].values.astype(float)
            mgrows += [dict(backbone=bb, source=src, probe=n, concept=c, dataset="CBIS-DDSM(test)", auc=auc(yc[te], sc[c][te])),
                       dict(backbone=bb, source=src, probe=n, concept=c, dataset="MIAS", auc=auc(yc[mi], sc[c][mi]))]
    ys = df.label.values[us_src].astype(int)
    for t in targets:
        m = (df.dataset == t).values
        if t == src or m.sum() < 30:
            continue
        y, pid = df.label.values[m].astype(int), df.patient_id.values[m]
        fit_mats, fixed = {}, {}
        for n, sc in scores.items():
            S_src = np.column_stack([zs(sc[c][us_src]) for c in SHARED])
            S_t = np.column_stack([zs(sc[c][m]) for c in SHARED])
            fixed[f"{n}_cbm0"] = lr(1.0).fit(S_src, ys).decision_function(S_t)
            fit_mats[f"{n}_cbmk"] = S_t
            for c in SHARED:
                pc = prox.loc[m, c].values.astype(float)
                if not np.isnan(pc).all():
                    zero.append(dict(backbone=bb, source=src, target=t, probe=n, concept=c,
                                     metric="concept_auc_vs_mask_proxy", value=auc(binarise(pd.Series(pc)).values, sc[c][m])))
                zero.append(dict(backbone=bb, source=src, target=t, probe=n, concept=c,
                                 metric="malignancy_auc_raw_concept", value=auc(y, sc[c][m])))
        few += episodes_engine(y, pid, P["shots"], episodes, P["seed"], fit_mats, fixed, [],
                               dict(backbone=bb, source=src, target=t))
    return zero, mgrows, few


def summarise(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return d, d
    s = d.groupby([c for c in d.columns if c not in ("episode", "auc")]).auc.agg(["mean", "std", "count"]).reset_index()
    s["ci95"] = 1.96 * s["std"].fillna(0) / np.sqrt(s["count"])
    return d, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backbones", nargs="*")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--geom", default="gt", help="gt (radiologist masks) or the --name used in 07_sam_masks.py")
    ap.add_argument("--sources", nargs="*", default=["BUS-BRA", "BUSB"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    P = cfg["pilot"]
    episodes = a.episodes or P["episodes"]
    out = cfg["work_root"] / f"results2_{a.geom}"
    out.mkdir(exist_ok=True)

    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[(df.lesion_type == "mass") & df.label.notna()].reset_index(drop=True)
    src_mg = df.dataset == P["source_dataset"]
    trp = set(df.patient_id[src_mg & (df.official_split == "train")])
    df = df[~(src_mg & (df.official_split == "test") & df.patient_id.isin(trp))].reset_index(drop=True)
    if a.geom != "gt":
        sam = pd.read_csv(cfg["work_root"] / a.geom / "sam_geometry.csv")
        df = df.drop(columns=GEOM_COLS).merge(sam[["uid"] + GEOM_COLS], on="uid", how="left")
        print(f"using {a.geom} geometry for {df[GEOM_COLS[0]].notna().sum()} lesions")
    targets = P["us_targets"]

    A0, Af, Ak, B0, Bm, Bk = [], [], [], [], [], []
    for bb in a.backbones or cfg["features"]["backbones"]:
        try:
            F, text = load_feats(cfg, bb, df)
        except FileNotFoundError:
            print(f"!! no features for {bb}")
            continue
        for src in a.sources:
            print(f"[{bb}] source={src}", flush=True)
            z, f, k = test_a(df, F, bb, src, targets, P, episodes)
            A0 += z; Af += f; Ak += k
            z, mg, k = test_b(df, F, text, bb, src, targets, P, episodes)
            B0 += z; Bm += mg; Bk += k

    A0, Af, B0, Bm = map(pd.DataFrame, (A0, Af, B0, Bm))
    Ak, Aks = summarise(Ak)
    Bk, Bks = summarise(Bk)
    for name, d in dict(testA_zeroshot=A0, testA_concept_fidelity=Af, testA_fewshot_summary=Aks,
                        testB_concepts=B0, testB_mg_side=Bm, testB_fewshot_summary=Bks).items():
        d.to_csv(out / f"{name}.csv", index=False)
    Ak.to_csv(out / "testA_fewshot_episodes.csv", index=False)

    L = [f"PILOT 2  (geometry source: {a.geom})", "",
         "TEST A - zero-shot cross-dataset malignancy AUC (no target labels)"]
    if len(A0):
        L.append(A0.pivot_table(index=["backbone", "source", "target"], columns="method", values="auc").round(3).to_string())
        L += ["", "TEST A - concept fidelity (Spearman predicted vs true mask geometry)"]
        L.append(Af.pivot_table(index=["backbone", "source", "target"], columns="concept", values="spearman").round(2).to_string())
        L += ["", "TEST A - few-shot AUC (mean over episodes)"]
        L.append(Aks[Aks.k > 0].pivot_table(index=["backbone", "source", "target", "k"], columns="method",
                                              values="mean").round(3).to_string())
        piv = Ak[Ak.k.isin([4, 8, 16])].pivot_table(index=["backbone", "source", "target", "k", "episode"],
                                                     columns="method", values="auc")
        g = pd.DataFrame({"cbm_vs_bb_prior": (piv.fused_cbm - piv.fused_bb), "cbm_vs_feat": (piv.fused_cbm - piv.feat_lr)})
        g = g.groupby(["backbone", "source", "target"]).agg(["mean", lambda x: (x > 0).mean()])
        g.columns = ["dCBM-BB", "win%(CBM>BB)", "dCBM-feat", "win%(CBM>feat)"]
        L += ["", "TEST A - paired deltas at k = 4/8/16", g.round(3).to_string()]
    if len(B0):
        L += ["", "TEST B - malignancy AUC, 2-concept bottleneck trained on source US labels (k = 0)"]
        b0 = Bks[(Bks.k == 0)].copy()
        b0["probe"] = b0.method.str.replace("_cbm0", "")
        L.append(b0.pivot_table(index=["backbone", "source", "target"], columns="probe", values="mean").round(3).to_string())
        L += ["", "TEST B - k-shot 2-concept bottleneck AUC (k = 4 / 16)"]
        bk = Bks[Bks.k.isin([4, 16])].copy()
        bk["probe"] = bk.method.str.replace("_cbmk", "").str.replace("_cbm0", "(k0 prior)")
        L.append(bk.pivot_table(index=["backbone", "source", "target", "k"], columns="probe", values="mean").round(3).to_string())
        L += ["", "TEST B - concept AUC vs target mask proxies"]
        cb = B0[B0.metric == "concept_auc_vs_mask_proxy"]
        if len(cb):
            L.append(cb.pivot_table(index=["backbone", "source", "target", "concept"], columns="probe", values="value").round(3).to_string())
        L += ["", "TEST B - mammography side (concept AUC)"]
        L.append(Bm.pivot_table(index=["backbone", "source", "concept", "dataset"], columns="probe", values="auc").round(3).to_string())
    txt = "\n".join(L)
    (out / "verdict2.txt").write_text(txt)
    print(txt)
    print(f"\nresults in {out}")
