"""Round 5 analyses (CPU, about 10 min). Reads finished run folders of 08; trains nothing.

A  Leakage with REFITTED predictors. The replacement test of 19 feeds the targets t into the weights w_d that were
   fitted on the predicted concepts c_hat, so a loss can come from a mismatch between the two input distributions.
   Here every input gets its own freshly fitted predictor: a patient grouped five fold logistic regression on the
   test lesions of each held-out dataset predicts malignancy from (1) the MedSAM targets t, (2) the predicted concepts
   c_hat of the seed ensemble, and (3) both together. Out of fold AUCs, patient bootstrap, pooled over the four
   datasets. AUC(c_hat, t) - AUC(t) = task information in the predicted concepts BEYOND the measured concepts, with
   no scale or correlation mismatch. Excess of each joint CBM over the sequential / independent CBM, paired.
B  Descriptor validity adjusted for malignancy (BrEaST). Each oriented concept is residualized on the malignancy
   label and on the black box logit of the same encoder (linear regression over the BrEaST lesions), and the AUC of
   the residual for the radiologist descriptor is reported. A concept that only carries malignancy information drops
   to about 0.5, a concept that measures the descriptor keeps its AUC. Also adjusted for the model's own logit.
   Columns: radiologist mask, MedSAM mask, sequential / hybrid CBMs, shuffled CBM (null), both encoders.

Output: <work_root>/stats/round5/{refit_leakage.csv, refit_excess.csv, descriptor_adjusted.csv, report.txt}
Usage:  python 20_round5.py [--boot 2000]
"""
import argparse
import importlib
import re
import warnings

import numpy as np
import pandas as pd

from common import GEOM_COLS, INT_COLS, ds_dir, load_cfg, roi_manifest_path

warnings.filterwarnings("ignore")
s15 = importlib.import_module("15_stats")
s17 = importlib.import_module("17_revision2")
s19 = importlib.import_module("19_revision3")
s14 = importlib.import_module("14_birads_validation")
DS4 = s19.DS4
CC = GEOM_COLS + INT_COLS
REFIT_MODELS = ["Joint CBM p=0.5", "Joint CBM p=0", "Sequential CBM", "Independent CBM", "Hybrid CBM, penalty 0.1",
                "Shuffled-concept CBM", "Shuffled-concept hybrid"]
REFIT_EXCESS = [("Joint CBM p=0.5", "Sequential CBM"), ("Joint CBM p=0.5", "Independent CBM"),
                ("Joint CBM p=0", "Sequential CBM"), ("Hybrid CBM, penalty 0.1", "Sequential CBM")]
DESC_MODELS = ["Sequential CBM", "Hybrid CBM, penalty 0.1", "Shuffled-concept CBM"]


def cols_of(runs):
    names, cc = s17.concept_cols(runs[0][1])
    return names, cc


def ens_concepts(runs, uids):
    names, cc = cols_of(runs)
    return names, np.mean([p.loc[uids, cc].values.astype(float) for p, _ in runs], 0)


def resid(s, Z):
    ok = ~np.isnan(s) & ~np.isnan(Z).any(1)
    out = np.full(len(s), np.nan)
    A = np.column_stack([np.ones(ok.sum()), Z[ok]])
    beta, *_ = np.linalg.lstsq(A, s[ok], rcond=None)
    out[ok] = s[ok] - A @ beta
    return out


def boot_ci(y, s, B=1000, seed=0):
    """fast bootstrap interval of the AUC (rank based)."""
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    rng = np.random.default_rng(seed)
    v = [s19.fauc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return np.nanpercentile(v, [2.5, 97.5])


def expected_targets(meta, desc):
    """(row label, concept, sign, DataFrame uid,y) for every expected descriptor pair, as in 14."""
    out = []
    for dkey, ckey, concept, sign in s14.EXPECTED:
        for col in [c for c in desc if dkey in str(c).lower()]:
            tk = s14.tokens(meta[col])
            NA = {"not applicable", "not available", "none"}
            if ckey == "NONCIRC":
                cats = sorted({t for ts in tk for t in ts if t not in NA | {"circumscribed", "no"}})
            else:
                cats = sorted({t for ts in tk for t in ts if re.search(ckey, t)})
            if not cats:
                continue
            ydf = pd.DataFrame({"uid": meta.uid, "y": tk.apply(lambda ts: any(t in cats for t in ts)).astype(int)})
            ydf = ydf[tk.apply(lambda ts: len(ts) > 0 and not set(ts) <= NA).values]
            if ydf.y.sum() < 10:
                continue
            cat = "non-circumscribed" if ckey == "NONCIRC" else "/".join(cats)
            out.append((f"{col}: {cat}", concept, sign, ydf))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    W = cfg["work_root"]
    CR = W / "cbm_runs"
    out = W / "stats" / "round5"
    out.mkdir(parents=True, exist_ok=True)
    refit, excess, drows = [], [], []
    rep = ["ROUND 5 ANALYSES", ""]

    for bk, bname in s15.BACKBONES.items():
        cfgs = dict(s15.configs(bk))
        cfgs.update({k: (r.format(bk=bk), m) for k, (r, m) in s19.EXTRA.items()})
        R = {k: s17.load_full(CR / r, m) for k, (r, m) in cfgs.items() if k in REFIT_MODELS + DESC_MODELS + [s19.BB]}
        R = {k: v for k, v in R.items() if v}
        print(f"{bname}: {sorted(R)}", flush=True)

        # ------------------------------------------------------------ A refitted predictors
        boots = {}
        for h in DS4:
            have = {k: v[h] for k, v in R.items() if h in v and k in REFIT_MODELS}
            if not have:
                continue
            st = s19.Site(h, have, a.boot)
            for k in have:
                names, C = ens_concepts(st.runs[k], st.uids)
                T = st.runs[k][0][0][[f"t_{n}" for n in names]].values.astype(float)
                ok = ~np.isnan(T).any(1)  # lesions with measured targets (all four datasets have masks)
                res = {}
                for nm, X in (("t", T), ("c", C), ("ct", np.column_stack([C, T]))):
                    Xn = np.where(ok[:, None], X, np.nan)
                    sc = s19.oof_scores(Xn, st.y, st.pid)
                    res[nm] = s19.boot_nan(st, sc)
                    refit.append(dict(backbone=bname, model=k, heldout=h, inputs=nm,
                                      auc=s19.fauc(st.y[~np.isnan(sc)], sc[~np.isnan(sc)])))
                boots.setdefault(k, {})[h] = res
        for k, per in boots.items():
            if set(per) != set(DS4):
                continue
            g = {nm: np.column_stack([per[h][nm] for h in DS4]).mean(1) for nm in ("t", "c", "ct")}
            boots[k]["pooled"] = g
            refit.append(dict(backbone=bname, model=k, heldout="pooled", inputs="summary",
                              auc_t=s19.ci(g["t"]), auc_c=s19.ci(g["c"]), auc_ct=s19.ci(g["ct"]),
                              c_minus_t=s19.ci(g["c"] - g["t"]), ct_minus_t=s19.ci(g["ct"] - g["t"]),
                              p_ct_minus_t=s19.pval(g["ct"] - g["t"])))
        for A, Bm in REFIT_EXCESS:
            if "pooled" in boots.get(A, {}) and "pooled" in boots.get(Bm, {}):
                ga, gb = boots[A]["pooled"], boots[Bm]["pooled"]
                v = (ga["ct"] - ga["t"]) - (gb["ct"] - gb["t"])
                excess.append(dict(backbone=bname, A=A, B=Bm, gain_beyond_t_A_minus_B=s19.ci(v), p=s19.pval(v)))

        # ------------------------------------------------------------ B descriptor validity adjusted for malignancy
        meta = s14.load_table(ds_dir(cfg, "BrEaST"))
        meta["uid"] = "BrEaST/" + meta[s14.m01._find_col(meta, "case")].astype(str).str.strip()
        desc = [c for c in meta.columns if c != "uid" and not pd.api.types.is_numeric_dtype(meta[c])
                and not any(k in str(c).lower().replace(" ", "_") for k in s14.SKIP) and 2 <= meta[c].nunique() <= 60]
        man = pd.read_csv(roi_manifest_path(cfg))
        man = man[man.dataset == "BrEaST"]
        radiol = man[["uid", "label"] + GEOM_COLS].merge(pd.read_csv(W / "intensity" / "gt.csv")[["uid"] + INT_COLS], on="uid", how="left")
        medsam = pd.read_csv(W / "medsam" / "sam_geometry.csv")[["uid"] + GEOM_COLS] \
            .merge(pd.read_csv(W / "intensity" / "medsam.csv")[["uid"] + INT_COLS], on="uid", how="outer")
        if "BrEaST" not in R.get(s19.BB, {}):
            continue
        bb = pd.concat([p.logit for p, _ in R[s19.BB]["BrEaST"]], axis=1).mean(1).rename("bb_logit")
        sources = {"Radiologist mask": radiol.drop(columns="label").set_index("uid"), "MedSAM mask": medsam.set_index("uid")}
        own = {}
        for k in DESC_MODELS:
            runs = R.get(k, {}).get("BrEaST")
            if not runs:
                continue
            names, cc = cols_of(runs)
            u = runs[0][0].index
            sources[k] = pd.DataFrame(np.mean([p.loc[u, cc].values.astype(float) for p, _ in runs], 0), index=u, columns=names)
            own[k] = pd.concat([p.logit for p, _ in runs], axis=1).mean(1)
        lab = man.set_index("uid").label
        for row, concept, sign, ydf in expected_targets(meta, desc):
            y0 = ydf.set_index("uid").y
            for src, t in sources.items():
                if concept not in t:
                    continue
                u = y0.index.intersection(t.index).intersection(bb.index).intersection(lab.index)
                y = y0.loc[u].values
                s = sign * t.loc[u, concept].values.astype(float)
                L = lab.loc[u].values.astype(float)
                adj = {"raw": s, "adj_label": resid(s, L[:, None]),
                       "adj_label_bb": resid(s, np.column_stack([L, bb.loc[u].values]))}
                if src in own:
                    adj["adj_label_own"] = resid(s, np.column_stack([L, own[src].loc[u].values]))
                r = dict(backbone=bname, descriptor=row, concept=concept, source=src, n_pos=int(y.sum()), N=len(y))
                for nm, v in adj.items():
                    r[nm] = s14.auc(y, v)
                    lo, hi = boot_ci(y, v)
                    r[f"{nm}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
                drows.append(r)

    refit, excess, D = pd.DataFrame(refit), pd.DataFrame(excess), pd.DataFrame(drows)
    refit.to_csv(out / "refit_leakage.csv", index=False)
    excess.to_csv(out / "refit_excess.csv", index=False)
    D.to_csv(out / "descriptor_adjusted.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        rep += ["A  LEAKAGE WITH REFITTED PREDICTORS (out of fold AUC on the test lesions, pooled [95% CI])",
                "   t = MedSAM targets only, c = predicted concepts only, ct = both; ct - t = information beyond t",
                refit[refit.heldout == "pooled"].drop(columns=["heldout", "inputs", "auc"], errors="ignore").to_string(index=False), "",
                "   per dataset (out of fold AUC):",
                refit[refit.heldout != "pooled"].pivot_table(index=["backbone", "model", "heldout"], columns="inputs", values="auc").round(3).to_string(), "",
                "A2 EXCESS GAIN BEYOND THE TARGETS (A minus B, paired)", excess.to_string(index=False), "",
                "B  DESCRIPTOR AUC ON BrEaST, RAW AND ADJUSTED FOR MALIGNANCY (label, black box logit, own logit)",
                D[[c for c in D.columns if not c.endswith("_ci")]].round(3).to_string(index=False), "",
                "   summary over rows (number of rows with AUC > 0.5, mean AUC):"]
        for (bkn, src), g in D.groupby(["backbone", "source"]):
            rep.append(f"   {bkn:10s} {src:28s} " + "  ".join(
                f"{nm} {int((g[nm] > 0.5).sum())}/{g[nm].notna().sum()} mean {g[nm].mean():.3f}"
                for nm in ("raw", "adj_label", "adj_label_bb", "adj_label_own") if nm in g and g[nm].notna().any()))
    txt = "\n".join(rep)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nsaved {out}")
