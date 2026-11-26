"""Local recalibration with a few labeled target lesions (CPU, a few minutes). No retraining.

Simulates a hospital that adopts a model and calibrates it on k labeled lesions per class of its own data.
For each model (raw-logit seed ensemble from 08), held-out dataset and k in {8, 16, 32}, 100 patient-disjoint
episodes: Platt scaling (logistic regression on the ensemble logit) is fitted on the support lesions and evaluated
on all remaining lesions of patients not in the support set. Reported per dataset as mean and 95% interval over episodes:
  ECE (10 bins) and Brier score before and after recalibration, sensitivity and specificity at probability 0.5
  before and after, and the AUC (unchanged by a monotone recalibration, shown as a check).

Output: <work_root>/stats/recalibration/{episodes.csv, summary.csv, report.txt}
Usage:  python 18_local_recalibration.py [--episodes 100] [--ks 8 16 32]
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import load_cfg

warnings.filterwarnings("ignore")
s15 = importlib.import_module("15_stats")
MODELS = ["Black box", "Multitask black box", "Joint CBM p=0.5", "Sequential CBM", "Hybrid CBM, penalty 0.1"]
DS = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST", "BUSI"]


def metrics(y, p):
    return dict(ece=s15.ece(y, p), brier=float(np.mean((p - y) ** 2)),
                sens=float(((p >= 0.5) & (y == 1)).sum() / max(y.sum(), 1)),
                spec=float(((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)),
                auc=s15.auc(y, p))


def episode(rng, y, pid, k):
    """k lesions per class from distinct patients; test = lesions of all other patients."""
    sup = []
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        used, pick = set(), []
        for i in idx:
            if pid[i] not in used:
                pick.append(i), used.add(pid[i])
            if len(pick) == k:
                break
        if len(pick) < k:
            return None, None
        sup += pick
    sp = set(pid[sup])
    te = np.array([i for i in range(len(y)) if pid[i] not in sp])
    return np.array(sup), te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 16, 32])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    CR = cfg["work_root"] / "cbm_runs"
    out = cfg["work_root"] / "stats" / "recalibration"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bk, bname in s15.BACKBONES.items():
        cfgs = s15.configs(bk)
        for name in MODELS:
            if name not in cfgs:
                continue
            root, mode = cfgs[name]
            data = s15.load(CR / root, mode)
            for held in [h for h in DS if h in data]:
                df = data[held]["df"]
                y = df.label.values.astype(int)
                pid = df.patient_id.astype(str).values
                z = df.score.values.astype(float)
                rng = np.random.default_rng(sum(map(ord, held + name)))
                for k in a.ks:
                    for e in range(a.episodes):
                        sup, te = episode(rng, y, pid, k)
                        if sup is None or len(np.unique(y[te])) < 2:
                            break
                        lr = LogisticRegression(C=1e4, max_iter=1000).fit(z[sup, None], y[sup])
                        before = metrics(y[te], 1 / (1 + np.exp(-z[te])))
                        after = metrics(y[te], lr.predict_proba(z[te, None])[:, 1])
                        rows.append(dict(backbone=bname, model=name, heldout=held, k=k, episode=e,
                                         **{f"{m}_before": v for m, v in before.items()},
                                         **{f"{m}_after": v for m, v in after.items()}))
                print(f"{bname} | {name} | {held}", flush=True)
    ep = pd.DataFrame(rows)
    ep.to_csv(out / "episodes.csv", index=False)
    agg = []
    for keys, g in ep.groupby(["backbone", "model", "heldout", "k"]):
        r = dict(zip(["backbone", "model", "heldout", "k"], keys))
        for m in ("ece", "brier", "sens", "spec"):
            for t in ("before", "after"):
                v = g[f"{m}_{t}"].values
                r[f"{m}_{t}"] = f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
        agg.append(r)
    sm = pd.DataFrame(agg)
    sm.to_csv(out / "summary.csv", index=False)
    # compact view: mean over the four datasets with patient identifiers, k = 16
    cmp_ = ep[ep.heldout.isin(DS[:4])].groupby(["backbone", "model", "k"])[
        ["ece_before", "ece_after", "brier_before", "brier_after", "sens_before", "sens_after", "spec_before", "spec_after"]].mean()
    L = ["LOCAL RECALIBRATION (Platt scaling on k labeled target lesions per class, evaluated on the remaining lesions)", "",
         "Mean over BUS-BRA, BUS-UCLM, BUSB, BrEaST and over episodes", cmp_.round(3).to_string(), "",
         "Per dataset, k = 16, mean [95% interval over episodes]",
         sm[sm.k == 16][["backbone", "model", "heldout", "ece_before", "ece_after", "sens_after", "spec_after"]].to_string(index=False)]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"saved {out}")
