"""Second-round revision analyses (CPU, about 10 min). Reads finished run folders of 08; trains nothing.

A  Leakage probe. For each concept model and held-out dataset, 5-fold patient-grouped cross-validated logistic
   regression on the test lesions: AUC of the measured concepts alone, of measured + predicted concepts, and of the
   prediction error (predicted - measured) alone. If the bottleneck carries only the named concepts, adding the
   predictions or using their errors should add little; a large gain means extra (leaked) label information.
   Done with both reference values: radiologist masks (gt_) and the MedSAM training targets (t_).
B  Correction on the SEED ENSEMBLE (no single-network vs ensemble mismatch): AUC before and after replacing all
   concepts with radiologist values and with MedSAM values (the in-distribution control), with paired bootstrap CIs
   for the change and for corrected model minus black box.
C  Hybrid reliance: covariance-aware concept share Cov(z_c, z)/Var(z) (an exact additive split of Var(z)), the AUC of
   each part alone (other part held at its mean), per held-out dataset.
D  Operating points: specificity at 90% and 85% sensitivity (descriptive, threshold set on the test set) per model.
E  MedSAM Dice against radiologist masks for every dataset.

Output: <work_root>/stats/revision2/{leakage_probe, ensemble_correction, hybrid_reliance, operating_points, dice}.csv
        + report.txt
Usage:  python 17_revision2.py [--boot 2000]
"""
import argparse
import importlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import load_cfg

warnings.filterwarnings("ignore")
s15 = importlib.import_module("15_stats")
DS4 = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST"]
CONCEPT_MODES = ("cbm", "shuf", "seq", "hyb")


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def load_full(root, mode):
    """held-out -> list of (preds DataFrame indexed by uid, meta) per seed."""
    out = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        m = s15.RUN_RE.match(d.name)
        if not m or m.group(1) != mode or not (d / "preds_test.csv").exists():
            continue
        out.setdefault(m.group(3), []).append((pd.read_csv(d / "preds_test.csv").set_index("uid"), json.load(open(d / "run.json"))))
    return out


def concept_cols(meta):
    names = [c for c in meta.get("concepts", []) if c.startswith(("g_", "i_"))]
    return names, [f"c{j}" for j in range(len(names))]


def cv_auc(X, y, groups, reps=5):
    ok = ~np.isnan(X).any(1)
    X, y, groups = X[ok], y[ok], groups[ok]
    if len(np.unique(y)) < 2 or len(y) < 40:
        return np.nan
    res = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        ug = np.unique(groups)
        perm = dict(zip(ug, rng.permutation(len(ug))))
        g2 = np.array([perm[g] for g in groups])
        s = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g2):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            s[te] = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
        res.append(auc(y[~np.isnan(s)], s[~np.isnan(s)]))
    return float(np.nanmean(res))


def ensemble_scores(runs, src=None):
    """raw-logit seed ensemble; with src ('gt_' or 't_'), every concept is replaced by that reference value
    (only the concept part of the logit changes: z' = z + w^T (t - c_hat))."""
    zs = []
    for p, meta in runs:
        z = p.logit.values.astype(float)
        if src is not None:
            names, cc = concept_cols(meta)
            w = np.ravel(meta["diag_w"])[: len(names)]
            C = p[cc].values.astype(float)
            T = p[[f"{src}{n}" for n in names]].values.astype(float)
            T = np.where(np.isnan(T), C, T)
            z = z + (T - C) @ w
        zs.append(z)
    return np.mean(zs, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    CR = cfg["work_root"] / "cbm_runs"
    out = cfg["work_root"] / "stats" / "revision2"
    out.mkdir(parents=True, exist_ok=True)
    leak, corr, rel, ops = [], [], [], []

    for bk, bname in s15.BACKBONES.items():
        cfgs = s15.configs(bk)
        bb = load_full(CR / cfgs["Black box"][0], "bb")
        if not bb:
            continue
        for name, (root, mode) in cfgs.items():
            R = load_full(CR / root, mode)
            if not R:
                continue
            print(f"{bname} | {name}", flush=True)
            for held in [h for h in DS4 if h in R]:
                runs = R[held]
                p0, meta0 = runs[0]
                uids = p0.index
                if held in bb:
                    uids = uids.intersection(bb[held][0][0].index)
                runs = [(p.loc[uids], m) for p, m in runs]
                p0 = runs[0][0]
                y = p0.label.values.astype(int)
                pid = p0.patient_id.astype(str).values
                s = ensemble_scores(runs)
                # D operating points
                fpr, tpr, _ = roc_curve(y, s)
                ops.append(dict(backbone=bname, model=name, heldout=held,
                                spec_at_sens90=float(1 - fpr[np.argmax(tpr >= 0.90)]), spec_at_sens85=float(1 - fpr[np.argmax(tpr >= 0.85)])))
                if mode not in CONCEPT_MODES:
                    continue
                names, cc = concept_cols(meta0)
                Cm = np.mean([p[cc].values.astype(float) for p, _ in runs], 0)
                # A leakage probe
                for src, lab in (("gt_", "radiologist"), ("t_", "medsam")):
                    T = p0[[f"{src}{n}" for n in names]].values.astype(float)
                    if np.isnan(T).all():
                        continue
                    a_meas = cv_auc(T, y, pid)
                    a_both = cv_auc(np.c_[T, Cm], y, pid)
                    a_err = cv_auc(Cm - T, y, pid)
                    a_pred = cv_auc(Cm, y, pid)
                    leak.append(dict(backbone=bname, model=name, heldout=held, reference=lab, auc_measured=a_meas,
                                     auc_predicted=a_pred, auc_measured_plus_predicted=a_both,
                                     gain_over_measured=a_both - a_meas, auc_error_only=a_err))
                # B ensemble correction with paired bootstrap, vs uncorrected and vs black box
                if held in bb:
                    sb = ensemble_scores([(p.loc[uids], m) for p, m in bb[held]])
                    boot = s15.Boot(pid, a.boot, seed=sum(map(ord, held)))
                    b0 = boot.aucs(y, s)
                    bbb = boot.aucs(y, sb)
                    for src, lab in (("gt_", "radiologist"), ("t_", "medsam")):
                        if src + names[0] not in p0.columns:
                            continue
                        sc = ensemble_scores(runs, src)
                        b1 = boot.aucs(y, sc)
                        dg, dv = b1 - b0, b1 - bbb
                        corr.append(dict(backbone=bname, model=name, heldout=held, source=lab, auc_before=auc(y, s),
                                         auc_after=auc(y, sc), black_box=auc(y, sb),
                                         change=f"{auc(y, sc) - auc(y, s):+.3f} [{np.nanpercentile(dg, 2.5):+.3f}, {np.nanpercentile(dg, 97.5):+.3f}]",
                                         after_minus_black_box=f"{auc(y, sc) - auc(y, sb):+.3f} [{np.nanpercentile(dv, 2.5):+.3f}, {np.nanpercentile(dv, 97.5):+.3f}]",
                                         _dg=dg, _dv=dv))
                # C hybrid reliance
                if mode == "hyb":
                    shares, ac, ar = [], [], []
                    for p, m in runs:
                        nm, cc2 = concept_cols(m)
                        w = np.ravel(m["diag_w"])[: len(nm)]
                        zc = p[cc2].values.astype(float) @ w + m["diag_b"][0]
                        z = p.logit.values.astype(float)
                        zr = z - zc
                        shares.append(np.cov(zc, z)[0, 1] / np.var(z, ddof=1))
                        ac.append(auc(y, zc)), ar.append(auc(y, zr))
                    rel.append(dict(backbone=bname, model=name, heldout=held, cov_share_concepts=np.mean(shares),
                                    auc_concept_part=np.mean(ac), auc_residual_part=np.mean(ar), auc_full=auc(y, s)))

    # pooled correction effects over the four datasets (equal weight)
    cdf = pd.DataFrame(corr)
    pooled = []
    if len(cdf):
        for (b, m, src), g in cdf.groupby(["backbone", "model", "source"]):
            if len(g) == 4:
                dg = np.mean(np.vstack(g._dg.values), 0)
                dv = np.mean(np.vstack(g._dv.values), 0)
                pooled.append(dict(backbone=b, model=m, source=src,
                                   pooled_change=f"{np.nanmean(dg):+.3f} [{np.nanpercentile(dg, 2.5):+.3f}, {np.nanpercentile(dg, 97.5):+.3f}]",
                                   pooled_after_minus_black_box=f"{np.nanmean(dv):+.3f} [{np.nanpercentile(dv, 2.5):+.3f}, {np.nanpercentile(dv, 97.5):+.3f}]"))
        cdf = cdf.drop(columns=["_dg", "_dv"])

    # E MedSAM Dice per dataset
    dice = []
    sf = cfg["work_root"] / "medsam" / "sam_geometry.csv"
    if sf.exists():
        sg = pd.read_csv(sf)
        if "dice" in sg:
            for dname, g in sg.groupby("dataset"):
                dice.append(dict(dataset=dname, n=len(g), dice_mean=g.dice.mean(), dice_median=g.dice.median(),
                                 dice_q25=g.dice.quantile(.25), dice_q75=g.dice.quantile(.75)))

    L = ["SECOND-ROUND REVISION ANALYSES", ""]
    for nm, rows, title in (("leakage_probe", leak, "A  LEAKAGE PROBE (5-fold patient-grouped CV logistic regression on the test lesions)"),
                            ("ensemble_correction", cdf, "B  CORRECTION OF ALL CONCEPTS ON THE SEED ENSEMBLE [95% CI]"),
                            ("ensemble_correction_pooled", pooled, "B2 POOLED OVER BUS-BRA, BUS-UCLM, BUSB, BrEaST"),
                            ("hybrid_reliance", rel, "C  HYBRID RELIANCE (covariance share = Cov(z_c, z) / Var(z); part AUCs with the other part fixed)"),
                            ("operating_points", ops, "D  SPECIFICITY AT FIXED SENSITIVITY (seed ensemble, descriptive)"),
                            ("dice", dice, "E  MEDSAM DICE AGAINST RADIOLOGIST MASKS")):
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        d.to_csv(out / f"{nm}.csv", index=False)
        L += [title, d.round(3).to_string(index=False), ""]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"saved {out}")
