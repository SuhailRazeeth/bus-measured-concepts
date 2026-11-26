"""Third-round revision analyses (CPU, about 20-30 min). Reads finished run folders of 08; trains nothing.

Addresses the third review round. Every interval is a 95% interval; "pooled" = equal-weight mean over the four
datasets with patient identifiers (BUS-BRA, BUS-UCLM, BUSB, BrEaST); all bootstraps resample patients.

A  Non-inferiority against the black box with three estimators, so the abstract can use one primary estimator:
     (1) pooled paired difference with a HIERARCHICAL bootstrap (seeds resampled, then patients), so training
         variance enters the interval, (2) DerSimonian-Laird random effects, (3) Hartung-Knapp-Sidik-Jonkman random
         effects (t distribution with K-1 df, recommended for few studies). Also every model against the black box of
         the OTHER encoder (best available black box).
B  Leakage by replacement with the MedSAM training targets, with CIs, for the joint CBM (p = 0.5 and p = 0), the
   independent and sequential CBMs, the hybrid and the shuffled controls. Excess loss (joint minus reference) with a
   paired CI. Scale control: the targets are first rescaled per concept to the mean and SD of the predicted
   concepts on the same test lesions (label free), which removes the shrinkage mismatch between predictions and targets.
C  Error probe with CIs: out-of-fold scores of the patient-grouped logistic regression on (predicted - target),
   patient bootstrap of the AUC, pooled over datasets.
D  Fair correction baseline: the black box with measured concepts as input (mode bbc), fed MedSAM values (as trained)
   or radiologist values (same information as the corrected models). Corrected hybrid minus this baseline.
E  Secondary outcomes with CIs: ECE, Brier score, specificity at 90% and 85% sensitivity, and their difference to
   the black box.
F  Concept share of the hybrids with CIs: covariance share kappa and variance share rho of the seed ensemble
   (patient bootstrap), and the range over seeds.
G  Multiplicity: two-sided bootstrap p values for every exploratory pooled contrast (sections B, D and the ablation
   contrasts of 15_stats.py), Holm adjusted within each encoder.
H  Box robustness of the hybrid (run folder <bk>_medsam_noexp_ic_u12_rp01_rb, written by 08 --box_jitter).
I  Local recalibration with intervals pooled over datasets (reads stats/recalibration/episodes.csv from 18).

Output: <work_root>/stats/revision3/*.csv + report.txt
Usage:  python 19_revision3.py [--boot 2000]
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import load_cfg

warnings.filterwarnings("ignore")
s15 = importlib.import_module("15_stats")
s17 = importlib.import_module("17_revision2")
DS4 = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST"]
BB = "Black box"
HYB = "Hybrid CBM, penalty 0.1"
LEAK_MODELS = ["Joint CBM p=0.5", "Joint CBM p=0", "Independent CBM", "Sequential CBM", HYB,
               "Shuffled-concept CBM", "Shuffled-concept hybrid"]
LEAK_EXCESS = [("Joint CBM p=0.5", "Sequential CBM"), ("Joint CBM p=0", "Sequential CBM"),
               ("Joint CBM p=0.5", "Independent CBM"), ("Joint CBM p=0.5", "Joint CBM p=0")]
SHARE_MODELS = [HYB, "Hybrid CBM, penalty 1.0", "Shuffled-concept hybrid"]
EXTRA = {  # new third-round configurations (trained by run_tier3.sh)
    "Joint CBM p=0": ("{bk}_medsam_noexp_ic_u12", "cbm"),
    "Independent CBM": ("{bk}_medsam_int100_noexp_ic_u12", "seq"),
    "Black box with measured concepts": ("{bk}_medsam_noexp_ic_u12", "bbc"),
}


def fauc(y, s):
    """Mann-Whitney AUC (fast, ties handled by mid-ranks)."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ci(v, d=3):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):+.{d}f} [{np.nanpercentile(v, 2.5):+.{d}f}, {np.nanpercentile(v, 97.5):+.{d}f}]"


def pval(v):
    """two-sided bootstrap p value of H0: difference = 0."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())) if len(v) else np.nan


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def ece(y, p):
    return s15.ece(y, p)


def spec_at(y, s, sens):
    fpr, tpr, _ = roc_curve(y, s)
    return float(1 - fpr[np.argmax(tpr >= sens)])


class Site:
    """one held-out dataset: aligned per-seed logits of every model, shared patient bootstrap draws."""

    def __init__(self, held, runs_by_model, B):
        uids = None
        for runs in runs_by_model.values():
            u = runs[0][0].index
            uids = u if uids is None else uids.intersection(u)
        self.uids = uids
        self.runs = {k: [(p.loc[uids], m) for p, m in r] for k, r in runs_by_model.items()}
        p0 = next(iter(self.runs.values()))[0][0]
        self.y = p0.label.values.astype(int)
        self.pid = p0.patient_id.astype(str).values
        self.idx = s15.Boot(self.pid, B, seed=sum(map(ord, held))).idx
        self.L = {k: np.column_stack([p.logit.values.astype(float) for p, _ in r]) for k, r in self.runs.items()}

    def score(self, k, src=None, scale=False):
        if src is None:
            return self.L[k].mean(1)
        return ensemble_replaced(self.runs[k], src, scale)

    def boot(self, s):
        return np.array([fauc(self.y[i], s[i]) for i in self.idx])

    def hboot(self, A, Bm, rng):
        """hierarchical bootstrap of AUC(A) - AUC(B): seeds of each model resampled, then patients."""
        LA, LB = self.L[A], self.L[Bm]
        out = np.empty(len(self.idx))
        for b, i in enumerate(self.idx):
            sa = LA[:, rng.integers(0, LA.shape[1], LA.shape[1])].mean(1)
            sb = LB[:, rng.integers(0, LB.shape[1], LB.shape[1])].mean(1)
            out[b] = fauc(self.y[i], sa[i]) - fauc(self.y[i], sb[i])
        return out


def ensemble_replaced(runs, src, scale=False):
    """seed ensemble with every concept replaced by the reference src ('t_' MedSAM targets or 'gt_' radiologist).
    scale=True first maps the reference to the mean and SD of the predicted concepts on the same lesions (label free)."""
    zs = []
    for p, meta in runs:
        names, cc = s17.concept_cols(meta)
        w = np.ravel(meta["diag_w"])[: len(names)]
        C = p[cc].values.astype(float)
        T = p[[f"{src}{n}" for n in names]].values.astype(float)
        if scale:
            mu_t, sd_t = np.nanmean(T, 0), np.nanstd(T, 0) + 1e-8
            T = np.nanmean(C, 0) + (T - mu_t) / sd_t * np.nanstd(C, 0)
        T = np.where(np.isnan(T), C, T)
        zs.append(p.logit.values.astype(float) + (T - C) @ w)
    return np.mean(zs, 0)


def oof_scores(X, y, groups, seed=0):
    ok = ~np.isnan(X).any(1)
    s = np.full(len(y), np.nan)
    Xo, yo, go = X[ok], y[ok], groups[ok]
    if len(np.unique(yo)) < 2 or len(yo) < 40:
        return s
    ug = np.unique(go)
    perm = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug))))
    g2 = np.array([perm[g] for g in go])
    so = np.full(len(yo), np.nan)
    for tr, te in GroupKFold(5).split(Xo, yo, g2):
        if len(np.unique(yo[tr])) < 2:
            continue
        sc = StandardScaler().fit(Xo[tr])
        so[te] = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xo[tr]), yo[tr]).decision_function(sc.transform(Xo[te]))
    s[ok] = so
    return s


def boot_nan(site, s):
    ok = ~np.isnan(s)
    return np.array([fauc(site.y[i][ok[i]], s[i][ok[i]]) for i in site.idx])


def random_effects(y, v, k_min=3):
    """DerSimonian-Laird and Hartung-Knapp-Sidik-Jonkman (with the max(1, q) safeguard)."""
    y, v = np.asarray(y, float), np.maximum(np.asarray(v, float), 1e-12)
    K = len(y)
    if K < k_min:
        return {}
    w = 1 / v
    ybar = (w * y).sum() / w.sum()
    Q = (w * (y - ybar) ** 2).sum()
    tau2 = max(0.0, (Q - (K - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    se_dl = np.sqrt(1 / ws.sum())
    q = (ws * (y - mu) ** 2).sum() / (K - 1)
    se_hk = np.sqrt(max(1.0, q) / ws.sum())
    tq = tdist.ppf(0.975, K - 1)
    return dict(mu=mu, tau2=tau2, I2=max(0.0, (Q - (K - 1)) / Q) if Q > 0 else 0.0,
                dl_lo=mu - 1.96 * se_dl, dl_hi=mu + 1.96 * se_dl, hk_lo=mu - tq * se_hk, hk_hi=mu + tq * se_hk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    CR = cfg["work_root"] / "cbm_runs"
    out = cfg["work_root"] / "stats" / "revision3"
    out.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("noninf", "leak", "leak_excess", "probe", "fair_correction", "secondary", "share",
                            "multiplicity", "box", "recal")}
    sites_all = {}

    for bk, bname in s15.BACKBONES.items():
        cfgs = dict(s15.configs(bk))
        cfgs.update({k: (r.format(bk=bk), m) for k, (r, m) in EXTRA.items()})
        R = {k: s17.load_full(CR / r, m) for k, (r, m) in cfgs.items()}
        R = {k: v for k, v in R.items() if v}
        if BB not in R:
            continue
        print(f"{bname}: {sorted(R)}", flush=True)
        sites = {}
        for h in DS4:
            have = {k: v[h] for k, v in R.items() if h in v}
            if BB in have:
                sites[h] = Site(h, have, a.boot)
        sites_all[bname] = sites
        rng = np.random.default_rng(7)
        H = list(sites)

        # ---------------------------------------------------------------- A non-inferiority, three estimators
        for k in R:
            if k == BB or not all(k in sites[h].L for h in H):
                continue
            d_pt = np.column_stack([sites[h].boot(sites[h].score(k)) - sites[h].boot(sites[h].score(BB)) for h in H])
            d_hb = np.column_stack([sites[h].hboot(k, BB, rng) for h in H])
            est = np.array([fauc(sites[h].y, sites[h].score(k)) - fauc(sites[h].y, sites[h].score(BB)) for h in H])
            re_pt = random_effects(est, np.nanvar(d_pt, 0))
            re_hb = random_effects(est, np.nanvar(d_hb, 0))
            pooled_hb = d_hb.mean(1)
            r = dict(backbone=bname, model=k, pooled=est.mean(),
                     pooled_patient_boot=ci(d_pt.mean(1)), pooled_seed_patient_boot=ci(pooled_hb),
                     lower_seed_patient=np.nanpercentile(pooled_hb, 2.5))
            if re_pt:
                r.update(dl=f"{re_pt['mu']:+.3f} [{re_pt['dl_lo']:+.3f}, {re_pt['dl_hi']:+.3f}]", tau2=re_pt["tau2"],
                         I2=re_pt["I2"], hksj=f"{re_pt['mu']:+.3f} [{re_pt['hk_lo']:+.3f}, {re_pt['hk_hi']:+.3f}]",
                         hksj_lower=re_pt["hk_lo"],
                         hksj_seed_patient=f"{re_hb['mu']:+.3f} [{re_hb['hk_lo']:+.3f}, {re_hb['hk_hi']:+.3f}]",
                         hksj_seed_patient_lower=re_hb["hk_lo"])
            for m in (0.02, 0.03):
                r[f"NI_{m}_seed_patient"] = bool(r["lower_seed_patient"] > -m)
                r[f"NI_{m}_hksj"] = bool(r.get("hksj_lower", -1) > -m)
            rows["noninf"].append(r)

        # ---------------------------------------------------------------- B leakage by replacement with targets
        drops = {}
        for k in LEAK_MODELS:
            if not all(k in sites[h].L for h in H):
                continue
            for scale in (False, True):
                per = []
                for h in H:
                    st = sites[h]
                    s0, s1 = st.score(k), st.score(k, "t_", scale)
                    per.append(st.boot(s1) - st.boot(s0))
                    rows["leak"].append(dict(backbone=bname, model=k, heldout=h, rescaled_targets=scale,
                                             auc_before=fauc(st.y, s0), auc_after=fauc(st.y, s1),
                                             change=ci(per[-1])))
                D = np.column_stack(per).mean(1)
                drops[(k, scale)] = D
                rows["leak"].append(dict(backbone=bname, model=k, heldout="pooled", rescaled_targets=scale, change=ci(D)))
        for A, Bm in LEAK_EXCESS:
            for scale in (False, True):
                if (A, scale) in drops and (Bm, scale) in drops:
                    v = drops[(A, scale)] - drops[(Bm, scale)]
                    rows["leak_excess"].append(dict(backbone=bname, A=A, B=Bm, rescaled_targets=scale,
                                                    change_A_minus_change_B=ci(v), p=pval(v), _v=v))

        # ---------------------------------------------------------------- C error probe with CIs
        for k in LEAK_MODELS + ["Hybrid CBM, penalty 1.0", "Joint CBM p=0.25", "Joint CBM p=0.75", "Joint CBM p=1.0"]:
            if not all(k in sites[h].L for h in H):
                continue
            per = []
            for h in H:
                st = sites[h]
                names, cc = s17.concept_cols(st.runs[k][0][1])
                Cm = np.mean([p[cc].values.astype(float) for p, _ in st.runs[k]], 0)
                T = st.runs[k][0][0][[f"t_{n}" for n in names]].values.astype(float)
                s = oof_scores(Cm - T, st.y, st.pid)
                b = boot_nan(st, s)
                per.append(b)
                rows["probe"].append(dict(backbone=bname, model=k, heldout=h, error_probe_auc=ci(b)))
            rows["probe"].append(dict(backbone=bname, model=k, heldout="pooled", error_probe_auc=ci(np.column_stack(per).mean(1))))

        # ---------------------------------------------------------------- D fair correction baseline
        BBC = "Black box with measured concepts"
        if all(BBC in sites[h].L for h in H):
            comps = {
                "measured-concept black box (MedSAM input) minus black box": (lambda st: st.score(BBC), lambda st: st.score(BB)),
                "measured-concept black box (radiologist input) minus black box": (lambda st: st.score(BBC, "gt_"), lambda st: st.score(BB)),
            }
            for k in (HYB, "Shuffled-concept hybrid", "Sequential CBM", "Joint CBM p=0.5"):
                if all(k in sites[h].L for h in H):
                    comps[f"{k} corrected minus measured-concept black box (radiologist input)"] = (
                        (lambda kk: lambda st: st.score(kk, "gt_"))(k), lambda st: st.score(BBC, "gt_"))
                    comps[f"{k} uncorrected minus measured-concept black box (MedSAM input)"] = (
                        (lambda kk: lambda st: st.score(kk))(k), lambda st: st.score(BBC))
            for q, (fa, fb) in comps.items():
                per = []
                r = dict(backbone=bname, contrast=q)
                for h in H:
                    st = sites[h]
                    sa, sb = fa(st), fb(st)
                    d = st.boot(sa) - st.boot(sb)
                    per.append(d)
                    r[h] = f"{fauc(st.y, sa) - fauc(st.y, sb):+.3f}"
                v = np.column_stack(per).mean(1)
                r.update(pooled=ci(v), p=pval(v), _v=v)
                rows["fair_correction"].append(r)

        # ---------------------------------------------------------------- E secondary outcomes with CIs
        for k in R:
            if not all(k in sites[h].L for h in H):
                continue
            acc = {m: [] for m in ("ece", "brier", "spec90", "spec85")}
            dif = {m: [] for m in acc}
            for h in H:
                st = sites[h]
                s, sb = st.score(k), st.score(BB)
                for i in st.idx[:500]:  # 500 resamples are enough for descriptive secondary intervals
                    y = st.y[i]
                    if len(np.unique(y)) < 2:
                        continue
                    pa, pb = 1 / (1 + np.exp(-s[i])), 1 / (1 + np.exp(-sb[i]))
                    va = dict(ece=ece(y, pa), brier=np.mean((pa - y) ** 2), spec90=spec_at(y, s[i], .9), spec85=spec_at(y, s[i], .85))
                    vb = dict(ece=ece(y, pb), brier=np.mean((pb - y) ** 2), spec90=spec_at(y, sb[i], .9), spec85=spec_at(y, sb[i], .85))
                    for m in acc:
                        acc[m].append((h, va[m]))
                        dif[m].append((h, va[m] - vb[m]))
            r = dict(backbone=bname, model=k,
                     auc=ci(np.column_stack([sites[h].boot(sites[h].score(k)) for h in H]).mean(1)).replace("+", ""))
            for m in acc:
                A_ = pd.DataFrame(acc[m], columns=["h", "v"]).groupby("h").v.apply(np.array)
                D_ = pd.DataFrame(dif[m], columns=["h", "v"]).groupby("h").v.apply(np.array)
                n = min(len(x) for x in A_)
                r[m] = ci(np.mean([x[:n] for x in A_], 0)).replace("+", "")
                r[f"{m}_minus_black_box"] = ci(np.mean([x[:n] for x in D_], 0))
            rows["secondary"].append(r)

        # ---------------------------------------------------------------- F concept share with CIs
        for k in SHARE_MODELS:
            if not all(k in sites[h].L for h in H):
                continue
            kap_b, rho_b, seed_kap = [], [], []
            for h in H:
                st = sites[h]
                ZC, Z = [], []
                for p, m in st.runs[k]:
                    nm, cc = s17.concept_cols(m)
                    w = np.ravel(m["diag_w"])[: len(nm)]
                    ZC.append(p[cc].values.astype(float) @ w + m["diag_b"][0])
                    Z.append(p.logit.values.astype(float))
                ZC, Z = np.column_stack(ZC), np.column_stack(Z)
                seed_kap.append([np.cov(ZC[:, j], Z[:, j])[0, 1] / np.var(Z[:, j], ddof=1) for j in range(Z.shape[1])])
                zc, z = ZC.mean(1), Z.mean(1)
                zr = z - zc
                kb, rb = [], []
                for i in st.idx[:1000]:
                    kb.append(np.cov(zc[i], z[i])[0, 1] / np.var(z[i], ddof=1))
                    rb.append(np.var(zc[i]) / (np.var(zc[i]) + np.var(zr[i])))
                kap_b.append(kb), rho_b.append(rb)
            sk = np.mean(np.array(seed_kap), 0)
            rows["share"].append(dict(backbone=bname, model=k, kappa=ci(np.mean(kap_b, 0)).replace("+", ""),
                                      rho=ci(np.mean(rho_b, 0)).replace("+", ""),
                                      kappa_seed_range=f"{sk.min():.3f} to {sk.max():.3f}"))

        # ---------------------------------------------------------------- G multiplicity (ablation contrasts)
        fam = []
        for A, Bm, q in s15.CONTRASTS:
            if all(A in sites[h].L and Bm in sites[h].L for h in H):
                v = np.column_stack([sites[h].boot(sites[h].score(A)) - sites[h].boot(sites[h].score(Bm)) for h in H]).mean(1)
                fam.append(dict(backbone=bname, family="ablation", contrast=f"{A} minus {Bm} ({q})", pooled=ci(v), p=pval(v)))
        for k in (HYB, "Shuffled-concept hybrid", "Joint CBM p=0.5", "Sequential CBM"):
            if all(k in sites[h].L for h in H):
                for src, lab in (("gt_", "radiologist"), ("t_", "MedSAM")):
                    for ref, nm in ((None, "uncorrected"), (BB, "black box")):
                        v = np.column_stack([sites[h].boot(sites[h].score(k, src)) -
                                             sites[h].boot(sites[h].score(k) if ref is None else sites[h].score(BB))
                                             for h in H]).mean(1)
                        fam.append(dict(backbone=bname, family="correction",
                                        contrast=f"{k} corrected with {lab} minus {nm}", pooled=ci(v), p=pval(v)))
        for r in rows["leak_excess"] + rows["fair_correction"]:
            if r["backbone"] == bname:
                fam.append(dict(backbone=bname, family="leakage" if "A" in r else "fair correction",
                                contrast=(f"{r['A']} minus {r['B']} (rescaled {r['rescaled_targets']})" if "A" in r else r["contrast"]),
                                pooled=r.get("change_A_minus_change_B", r.get("pooled")), p=r["p"]))
        if fam:
            adj = holm([f["p"] for f in fam])
            for f, pa in zip(fam, adj):
                f["p_holm"] = pa
            rows["multiplicity"] += fam

        # ---------------------------------------------------------------- H box robustness of the hybrid
        rb = CR / f"{bk}_medsam_noexp_ic_u12_rp01_rb"
        if rb.exists():
            for h in DS4:
                parts = {}
                for d in sorted(rb.glob(f"hyb_s*_{h}")):
                    for f in d.glob("preds_test*.csv"):
                        tag = f.stem.replace("preds_test", "") or "_jit0"
                        parts.setdefault(tag, []).append(pd.read_csv(f).set_index("uid").logit)
                if "_jit0" not in parts:
                    continue
                ref = pd.concat(parts["_jit0"], axis=1).mean(1)
                p0 = pd.read_csv(next(rb.glob(f"hyb_s*_{h}")) / "preds_test.csv").set_index("uid").loc[ref.index]
                y = p0.label.values.astype(int)
                idx = s15.Boot(p0.patient_id.astype(str).values, a.boot, seed=sum(map(ord, h))).idx
                b0 = np.array([fauc(y[i], ref.values[i]) for i in idx])
                for tag, L in sorted(parts.items()):
                    s = pd.concat(L, axis=1).mean(1).loc[ref.index].values
                    b1 = np.array([fauc(y[i], s[i]) for i in idx])
                    rows["box"].append(dict(backbone=bname, heldout=h, jitter=tag.replace("_jit", ""), seeds=len(L),
                                            auc=fauc(y, s), change_vs_exact_box=ci(b1 - b0), _d=b1 - b0))

    # A2 each encoder's hybrid against the black box of the OTHER encoder (best available black box)
    if len(sites_all) == 2:
        (n1, S1), (n2, S2) = sites_all.items()
        for (na, SA), (nb, SB) in (((n1, S1), (n2, S2)), ((n2, S2), (n1, S1))):
            per, est = [], []
            for h in DS4:
                if h in SA and h in SB and HYB in SA[h].runs and BB in SB[h].runs:
                    st = Site(h, {"A": SA[h].runs[HYB], "B": SB[h].runs[BB]}, a.boot)
                    per.append(st.boot(st.score("A")) - st.boot(st.score("B")))
                    est.append(fauc(st.y, st.score("A")) - fauc(st.y, st.score("B")))
            if len(per) == 4:
                v = np.column_stack(per).mean(1)
                re_ = random_effects(est, np.nanvar(np.column_stack(per), 0))
                rows["noninf"].append(dict(backbone=f"{na} hybrid vs {nb} black box", model=HYB, pooled=np.mean(est),
                                           pooled_patient_boot=ci(v), lower_seed_patient=np.nan,
                                           hksj=f"{re_['mu']:+.3f} [{re_['hk_lo']:+.3f}, {re_['hk_hi']:+.3f}]",
                                           hksj_lower=re_["hk_lo"]))

    # pooled box robustness
    bdf = pd.DataFrame(rows["box"])
    if len(bdf):
        for (b, j), g in bdf.groupby(["backbone", "jitter"]):
            if len(g) == 4:
                rows["box"].append(dict(backbone=b, heldout="pooled", jitter=j, auc=g.auc.mean(),
                                        change_vs_exact_box=ci(np.mean(np.vstack(g._d.values), 0))))

    # ---------------------------------------------------------------- I recalibration, pooled per episode
    ef = cfg["work_root"] / "stats" / "recalibration" / "episodes.csv"
    if ef.exists():
        ep = pd.read_csv(ef)
        ep = ep[ep.heldout.isin(DS4)]
        for (b, m, k), g in ep.groupby(["backbone", "model", "k"]):
            pe = g.groupby("episode")[["ece_before", "ece_after", "brier_before", "brier_after", "spec_after", "sens_after"]].mean()
            if g.heldout.nunique() < 4:
                continue
            rows["recal"].append(dict(backbone=b, model=m, k=k, ece_before=pe.ece_before.mean(), ece_after=ci(pe.ece_after).replace("+", ""),
                                      brier_after=ci(pe.brier_after).replace("+", ""), sens_after=ci(pe.sens_after).replace("+", ""),
                                      spec_after=ci(pe.spec_after).replace("+", "")))

    titles = {
        "noninf": "A  NON-INFERIORITY VS BLACK BOX: pooled with patient bootstrap, with seed+patient bootstrap, DL and HKSJ random effects",
        "leak": "B  REPLACEMENT WITH THE MEDSAM TRAINING TARGETS (change in AUC; rescaled = targets matched to the mean and SD of the predictions)",
        "leak_excess": "B2 EXCESS LOSS (change of A minus change of B, pooled) [95% CI]",
        "probe": "C  ERROR PROBE AUC (predicted minus MedSAM target), patient bootstrap [95% CI]",
        "fair_correction": "D  FAIR CORRECTION BASELINE (black box with measured concepts as input)",
        "secondary": "E  SECONDARY OUTCOMES, pooled [95% CI], and difference to the black box",
        "share": "F  CONCEPT SHARE OF THE HYBRIDS (seed ensemble), pooled [95% CI], range over seeds",
        "multiplicity": "G  EXPLORATORY CONTRASTS WITH BOOTSTRAP p AND HOLM ADJUSTMENT (within encoder)",
        "box": "H  HYBRID WITH SHIFTED / RESCALED LESION BOXES (change vs exact box)",
        "recal": "I  LOCAL RECALIBRATION POOLED OVER THE FOUR DATASETS (interval over 100 episodes)",
    }
    Lr = ["THIRD-ROUND REVISION ANALYSES (B = %d patient bootstrap resamples)" % a.boot, ""]
    for k, rr in rows.items():
        d = pd.DataFrame(rr)
        if d.empty:
            Lr += [titles[k], "(no runs found for this section)", ""]
            continue
        d = d[[c for c in d.columns if not c.startswith("_")]]
        d.to_csv(out / f"{k}.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 90):
            Lr += [titles[k], d.round(4).to_string(index=False), ""]
    txt = "\n".join(Lr)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"saved {out}")
