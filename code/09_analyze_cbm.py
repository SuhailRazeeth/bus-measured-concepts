"""Analyse pilot-3 runs (08_train_cbm.py). CPU only.

For every run folder (backbone_geom[_mg]) and every held-out US dataset:
  1  Zero-shot AUC per mode (mean +- sd over seeds) + seed-ensemble AUC with patient-cluster bootstrap 95% CI,
     paired bootstrap difference cbm-bb and mt-bb (95% CI, one-sided p), ECE of the ensemble.
  2  Mammography side (--mg runs): AUC on CBIS-DDSM test / INbreast / MIAS.
  3  Concept faithfulness: Spearman(predicted, radiologist-mask geometry); expert BI-RADS concept AUC (BrEaST).
  4  Concept intervention (cbm): replace predicted geometry concepts with radiologist-mask values for the
     most uncertain f% of cases -> AUC.  (clinician-in-the-loop correction)
  4c Intervention breakdown: change in AUC when correcting only one concept group / one concept / all-but-one
     concept (radiologist masks, f = 50% and 100%), plus each concept's weight x spread in the diagnosis head.
  5  Few-shot adaptation on the target (k labelled lesions/class, patient-disjoint episodes).
  6  Split-conformal (class-conditional) "refer" sets: coverage, malignant miss rate, refer rate, per scanner.

Usage:  python 09_analyze_cbm.py [--run biomedclip_medsam] [--episodes 100] [--boot 1000]
Output: <work_root>/cbm_runs/<run>/analysis/report.txt + CSVs + plots
"""
import argparse
import importlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import GEOM_COLS, load_cfg

p1 = importlib.import_module("05_pilot_experiments")
auc, lr, zs, episode_split = p1.auc, p1.lr, p1.zs, p1.episode_split
warnings.filterwarnings("ignore")
EXPERT = ["c_shape_irregular", "c_margin_noncirc"]
RUN_RE = re.compile(r"^(bb|mt|cbm|seq|shuf|hyb)_s(\d+)_(.+)$")
BOTTLENECK = ("cbm", "seq", "shuf", "hyb")


def load_runs(root):
    runs = []
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "run.json").exists():
            continue
        mode, seed, held = m.group(1), int(m.group(2)), m.group(3)
        meta = json.load(open(d / "run.json"))
        for f in d.glob("preds_*.csv"):
            tag = f.stem.replace("preds_", "")
            p = pd.read_csv(f)
            p["mode"], p["seed"], p["heldout"], p["tag"] = mode, seed, held, tag
            runs.append(dict(mode=mode, seed=seed, heldout=held, tag=tag, preds=p, dir=d, meta=meta))
    return runs


def cluster_boot(y, scores: dict, pid, B, rng):
    """Patient-cluster bootstrap. scores: name -> score array. Returns name -> array of B AUCs."""
    up = np.unique(pid)
    idx_by = {u: np.flatnonzero(pid == u) for u in up}
    out = {k: [] for k in scores}
    for _ in range(B):
        s = np.concatenate([idx_by[u] for u in rng.choice(up, len(up))])
        for k, v in scores.items():
            out[k].append(auc(y[s], v[s]))
    return {k: np.array(v) for k, v in out.items()}


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def conformal(y, p, pid, n_cal, alpha, reps, rng, groups=None):
    """Class-conditional split conformal on probability of malignancy p. Returns list of dicts."""
    rows = []
    up = np.unique(pid)
    for r in range(reps):
        perm = rng.permutation(up)
        cal_p, n = [], 0
        for u in perm:
            cal_p.append(u)
            n += (pid == u).sum()
            if n >= n_cal:
                break
        cal = np.isin(pid, cal_p)
        te = ~cal
        q = {}
        for c in (0, 1):
            sc = 1 - (p[cal & (y == c)] if c == 1 else 1 - p[cal & (y == c)])
            k = len(sc)
            if k == 0:
                q[c] = 1.0
                continue
            lvl = min(1.0, np.ceil((k + 1) * (1 - alpha)) / k)
            q[c] = np.quantile(sc, lvl, method="higher")
        inc1 = (1 - p[te]) <= q[1]
        inc0 = p[te] <= q[0]
        yt = y[te]
        cov = np.where(yt == 1, inc1, inc0)
        res = dict(rep=r, n_cal=n_cal, coverage=cov.mean(), mal_miss=1 - inc1[yt == 1].mean() if (yt == 1).any() else np.nan,
                   ben_cov=inc0[yt == 0].mean() if (yt == 0).any() else np.nan, set_size=(inc0.astype(int) + inc1).mean(),
                   refer=(inc0 & inc1).mean(), empty=(~inc0 & ~inc1).mean())
        if groups is not None:
            gt = groups[te]
            for g in np.unique(gt):
                m = (gt == g) & (yt == 1)
                res[f"mal_miss[{g}]"] = 1 - inc1[m].mean() if m.any() else np.nan
        rows.append(res)
    return rows


def intervene(Rm, y, gt, f, idx):
    """Mean AUC over seeds after replacing concepts idx (columns of gt) by gt for the f most uncertain cases."""
    a_s = []
    for r in Rm:
        w = np.array(r["meta"]["diag_w"]).ravel()
        b = r["meta"]["diag_b"][0]
        C = r["preds"][[f"c{j}" for j in range(len(w))]].values.copy()
        lo = r["preds"].logit.values.astype(float)
        resid = lo - (C @ w + b)  # 0 for cbm / seq; the image residual for hyb (left unchanged by a correction)
        sel = np.argsort(np.abs(lo - np.median(lo)))[: int(round(f * len(lo)))]
        for j in idx:
            v = gt[sel, j]
            C[sel, j] = np.where(np.isnan(v), C[sel, j], v)  # correct every concept that has a value
        a_s.append(auc(y, C @ w + b + resid))
    return float(np.mean(a_s))


def analyse(root, P, episodes, B):
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    runs = load_runs(root)
    if not runs:
        print(f"no finished runs in {root}")
        return
    L = [f"PILOT 3 analysis: {root.name}", ""]
    # cbm-only variants (_int25, _noexp): borrow bb / mt runs from the matching base folder for paired comparisons
    base_root = root.parent / re.sub(r"_(int\d+|noexp|ic|rp\d+|rb|shufT)", "", root.name)
    if base_root != root and base_root.exists():
        have = {r["mode"] for r in runs}
        borrowed = [r for r in load_runs(base_root) if r["mode"] not in have and r["mode"] in ("bb", "mt")]
        if borrowed:
            runs += borrowed
            L.append(f"(bb/mt runs borrowed from {base_root.name})")
    rng = np.random.default_rng(0)
    t_auc, t_diff, t_mg, t_fid, t_exp, t_int, t_few, t_conf = [], [], [], [], [], [], [], []
    t_grp, t_w, t_share = [], [], []
    test = [r for r in runs if r["tag"] == "test"]
    for held in sorted({r["heldout"] for r in test}):
        R = [r for r in test if r["heldout"] == held]
        base = R[0]["preds"]
        y, pid = base.label.values.astype(int), base.patient_id.values
        ens, conc = {}, {}
        for mode in sorted({r["mode"] for r in R}):
            Rm = sorted([r for r in R if r["mode"] == mode], key=lambda r: r["seed"])
            per_seed = [auc(y, r["preds"].logit.values) for r in Rm]
            ens[mode] = np.mean([zs(r["preds"].logit.values) for r in Rm], 0)
            conc[mode] = np.mean([r["preds"][[c for c in r["preds"].columns if re.fullmatch(r"c\d+", c)]].values for r in Rm], 0)
            pr = 1 / (1 + np.exp(-np.mean([r["preds"].logit.values for r in Rm], 0)))
            t_auc.append(dict(heldout=held, mode=mode, n=len(y), seeds=len(Rm), auc_mean=np.mean(per_seed),
                              auc_sd=np.std(per_seed), ens_auc=auc(y, ens[mode]), ece=ece(y, pr)))
            # continuous concepts of this run (geometry [+ intensity]), in bottleneck order
            CC = [c for c in Rm[0]["meta"].get("concepts", GEOM_COLS) if c not in EXPERT]
            # 4 intervention
            if mode in BOTTLENECK:
                # source "radiologist" = radiologist-mask geometry; "train_label" = the concept labels the model
                # was trained on (MedSAM/SAM geometry; only saved by the new 08)
                srcs = {"radiologist": [f"gt_{g}" for g in CC], "train_label": [f"t_{g}" for g in CC]}
                for src, cols in srcs.items():
                    if not set(cols) <= set(base.columns):
                        continue
                    gt = base[cols].values.astype(float)
                    if np.isnan(gt).all():
                        continue
                    allc = list(range(len(CC)))
                    for f in [0, 0.1, 0.25, 0.5, 1.0]:
                        t_int.append(dict(mode=mode, heldout=held, source=src, fraction=f, auc=intervene(Rm, y, gt, f, allc)))
                    if src != "radiologist":
                        continue
                    # 4c breakdown: groups, single concepts, all-but-one
                    a0 = intervene(Rm, y, gt, 0, allc)
                    has = [j for j in allc if not np.isnan(gt[:, j]).all()]
                    groups = {"all": has, "geometry": [j for j in has if CC[j].startswith("g_")],
                              "echo": [j for j in has if CC[j].startswith("i_")]}
                    groups.update({f"only {CC[j]}": [j] for j in has})
                    groups.update({f"all but {CC[j]}": [k for k in has if k != j] for j in has})
                    for gname, idx in groups.items():
                        if idx:
                            for f in [0.5, 1.0]:
                                t_grp.append(dict(mode=mode, heldout=held, intervene=gname, fraction=f,
                                                  delta_auc=intervene(Rm, y, gt, f, idx) - a0))
                    # how much each concept moves the diagnosis: |weight| x sd(predicted concept), mean over seeds
                    for j, c in enumerate(CC):
                        t_w.append(dict(mode=mode, heldout=held, concept=c, weight=np.mean([np.ravel(r["meta"]["diag_w"])[j] for r in Rm]),
                                        influence=np.mean([abs(np.ravel(r["meta"]["diag_w"])[j]) * r["preds"][f"c{j}"].std() for r in Rm])))
            # 8 hybrid: how much of the decision do the named concepts carry?
            if mode == "hyb":
                parts = []
                for r in Rm:
                    w = np.array(r["meta"]["diag_w"]).ravel()
                    C = r["preds"][[f"c{j}" for j in range(len(w))]].values
                    cp = C @ w + r["meta"]["diag_b"][0]
                    rp = r["preds"].logit.values - cp
                    parts.append((cp, rp))
                cpe = np.mean([zs(c_) for c_, _ in parts], 0)
                rpe = np.mean([zs(r_) for _, r_ in parts], 0)
                t_share.append(dict(heldout=held, auc_full=auc(y, ens[mode]), auc_concept_part=auc(y, cpe),
                                    auc_residual_part=auc(y, rpe),
                                    var_share_concepts=np.mean([np.var(c_) / (np.var(c_) + np.var(r_) + 1e-12) for c_, r_ in parts]),
                                    abs_share_concepts=np.mean([np.mean(np.abs(c_ - c_.mean()) / (np.abs(c_ - c_.mean()) + np.abs(r_ - r_.mean()) + 1e-12))
                                                                for c_, r_ in parts])))
            # 3 faithfulness
            if mode in ("mt",) + BOTTLENECK:
                for j, g in enumerate(CC):
                    if f"gt_{g}" not in base.columns:
                        continue
                    gv = base[f"gt_{g}"].values
                    ok = ~np.isnan(gv)
                    if ok.sum() > 10:
                        t_fid.append(dict(heldout=held, mode=mode, concept=g, spearman=spearmanr(conc[mode][ok, j], gv[ok]).statistic))
                for j, e in enumerate(EXPERT):
                    ev = base[e].values.astype(float)
                    ok = ~np.isnan(ev)
                    if ok.sum() > 10 and conc[mode].shape[1] > len(CC) + j:
                        sh =conc[mode][:, GEOM_COLS.index("g_ellipse_misfit")] - conc[mode][:, GEOM_COLS.index("g_solidity")]
                        mg = conc[mode][:, GEOM_COLS.index("g_spiculation")]
                        t_exp.append(dict(heldout=held, mode=mode, concept=e, expert_head_auc=auc(ev[ok], conc[mode][ok, len(CC) + j]),
                                          geometry_proxy_auc=auc(ev[ok], (sh if "shape" in e else mg)[ok])))
            # 5 few-shot (seed 0 embeddings)
            r0 = Rm[0]
            X = np.load(r0["dir"] / "emb_test.npy").astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            fit = {"kshot_emb": X}
            if mode in BOTTLENECK:
                fit["kshot_concepts"] = zs(conc[mode])
            combos = [("fused", ["kshot_emb", "src"])]
            rows = []
            srs = np.random.default_rng(P["seed"])
            for k in [kk for kk in P["shots"] if kk >= 4] or [4]:
                for e in range(episodes):
                    sup, te = episode_split(srs, y, pid, k)
                    if sup is None:
                        break
                    res = {n: lr(1.0).fit(M[sup], y[sup]).decision_function(M[te]) for n, M in fit.items()}
                    res["src"] = ens[mode][te]
                    for n, parts in combos:
                        res[n] = sum(zs(res[q]) for q in parts)
                    rows += [dict(heldout=held, mode=mode, k=k, method=n, auc=auc(y[te], s)) for n, s in res.items()]
            t_few += rows
            # 6 conformal
            groups = base.device.fillna("").astype(str).values if base.device.fillna("").astype(str).str.len().gt(0).any() else None
            for n_cal in [20, 50, 100]:
                if n_cal < len(y) // 2:
                    t_conf += [dict(heldout=held, mode=mode, **d) for d in conformal(y, pr, pid, n_cal, 0.1, 200, rng, groups)]
        # 1 bootstrap CI and paired differences
        bs = cluster_boot(y, ens, pid, B, rng)
        for mode in ens:
            lo, hi = np.nanpercentile(bs[mode], [2.5, 97.5])
            for t in t_auc:
                if t["heldout"] == held and t["mode"] == mode:
                    t["ens_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        for m2 in ("cbm", "mt", "seq", "shuf", "hyb"):
            if m2 in bs and "bb" in bs:
                d = bs[m2] - bs["bb"]
                t_diff.append(dict(heldout=held, comparison=f"{m2}-bb", delta=auc(y, ens[m2]) - auc(y, ens["bb"]),
                                   ci=f"[{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]", p_one_sided=(d <= 0).mean()))
    # 2 mammography side
    for r in runs:
        if r["tag"].startswith("extra_"):
            t_mg.append(dict(dataset=r["tag"][6:], heldout=r["heldout"], mode=r["mode"], seed=r["seed"],
                             auc=auc(r["preds"].label.values, r["preds"].logit.values)))

    def emit(title, d, **pv):
        nonlocal L
        d = pd.DataFrame(d)
        if d.empty:
            return d
        fname = re.sub(r"[^a-z0-9]+", "_", title.split("(")[0].lower()).strip("_")
        d.to_csv(out / f"{fname}.csv", index=False)
        L += ["", title, (d.pivot_table(**pv) if pv else d).round(3).to_string()]
        return d

    emit("1 zero-shot AUC", t_auc)
    emit("1b paired difference vs black box (ensemble, patient bootstrap)", t_diff)
    if t_mg:
        emit("2 mammography side AUC (mean over seeds/held-outs)", t_mg, index="dataset", columns="mode", values="auc", aggfunc="mean")
    emit("3 concept faithfulness (Spearman vs radiologist-mask geometry)", t_fid, index=["heldout", "mode"], columns="concept", values="spearman")
    emit("3b expert BI-RADS concept AUC", t_exp)
    emit("4 concept intervention (cbm): AUC after correcting f% most uncertain cases", t_int, index=["mode", "source", "heldout"], columns="fraction", values="auc")
    if t_grp:
        g = pd.DataFrame(t_grp)
        order = list(dict.fromkeys(g.intervene))
        for md in sorted(g["mode"].unique()):
            for f in [1.0, 0.5]:
                gf = g[(g["mode"] == md) & (g.fraction == f)].pivot_table(index="intervene", columns="heldout",
                                                                          values="delta_auc").reindex(order)
                gf["mean"] = gf.mean(axis=1)
                L += ["", f"4c [{md}] intervention breakdown: change in AUC, radiologist values, {int(f * 100)}% of cases",
                      gf.round(3).to_string()]
        g.to_csv(out / "4c_intervention_breakdown.csv", index=False)
        emit("4d diagnosis head: weight and influence (|w| x sd of predicted concept)", t_w,
             index=["mode", "concept"], columns="heldout", values="influence")
    # 7 test-time box robustness (preds_test_jit*.csv written by 08 --box_jitter)
    t_jit = []
    jit = [r for r in runs if r["tag"].startswith("test_jit")]
    for (held, mode), _ in pd.DataFrame([(r["heldout"], r["mode"]) for r in jit]).groupby([0, 1]) if jit else []:
        Rm = [r for r in jit if r["heldout"] == held and r["mode"] == mode]
        R0 = sorted([r for r in test if r["heldout"] == held and r["mode"] == mode and r["seed"] in {q["seed"] for q in Rm}],
                    key=lambda r: r["seed"])
        y0 = R0[0]["preds"].label.values.astype(int) if R0 else None
        row = dict(heldout=held, mode=mode, seeds=len(R0),
                   jit0=auc(y0, np.mean([zs(r["preds"].logit.values) for r in R0], 0)) if R0 else np.nan)
        for tag in sorted({r["tag"] for r in Rm}):
            Rt = sorted([r for r in Rm if r["tag"] == tag], key=lambda r: r["seed"])
            row[tag.replace("test_", "")] = auc(Rt[0]["preds"].label.values.astype(int),
                                                np.mean([zs(r["preds"].logit.values) for r in Rt], 0))
        t_jit.append(row)
    if t_share:
        emit("8 hybrid concept share (AUC of concept part, residual part and full logit, share of the logit carried by the concepts)", t_share)
    if t_jit:
        emit("7 test-time box jitter: ensemble AUC with ROIs re-cropped from shifted / rescaled boxes", t_jit)
    few = emit("5 few-shot AUC on target", t_few, index=["heldout", "mode", "k"], columns="method", values="auc", aggfunc="mean")
    conf = pd.DataFrame(t_conf)
    if len(conf):
        conf.to_csv(out / "6_conformal_reps.csv", index=False)
        cs = conf.groupby(["heldout", "mode", "n_cal"]).mean(numeric_only=True).drop(columns="rep")
        L += ["", "6 conformal (alpha = 0.10; mean over 200 calibration draws)", cs.round(3).to_string()]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt)
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(few):
            g = few.groupby(["heldout", "mode", "k", "method"]).auc.mean().reset_index()
            hs = g.heldout.unique()
            fig, ax = plt.subplots(1, len(hs), figsize=(4.5 * len(hs), 3.8), squeeze=False)
            for a_, h in zip(ax[0], hs):
                for (m, meth), d in g[g.heldout == h].groupby(["mode", "method"]):
                    a_.plot(d.k, d.auc, marker="o", label=f"{m}:{meth}")
                a_.set_xscale("log", base=2), a_.set_title(h), a_.set_xlabel("k per class"), a_.grid(alpha=.3)
            ax[0][0].set_ylabel("AUROC"), ax[0][0].legend(fontsize=6)
            fig.tight_layout(), fig.savefig(out / "fewshot.png", dpi=150), plt.close(fig)
    except Exception as ex:
        print("plot skipped:", ex)
    print(f"\nanalysis saved in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--run", nargs="*")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    base = cfg["work_root"] / "cbm_runs"
    for d in sorted(base.iterdir()) if base.exists() else []:
        if d.is_dir() and (not a.run or d.name in a.run):
            analyse(d, cfg["pilot"], a.episodes, a.boot)
