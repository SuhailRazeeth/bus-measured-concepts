"""Tier 2 statistics for the revision (CPU, a few minutes). Reads the finished run folders of 08; trains nothing.

A  Main table: seed-ensemble AUC per held-out dataset from RAW logits (no test-set standardisation), mean over all
   five datasets and without BUSI, between-seed SD, internal validation AUC, calibration (ECE), and sensitivity /
   specificity at probability 0.5.
B  Non-inferiority against the black box with the same encoder: patient-cluster bootstrap of the paired AUC
   difference per dataset, pooled as the equal-weight mean over datasets, with margins 0.02 and 0.03.
C  Model selection on the internal validation split only (best_val_auc of each run, from the training datasets):
   intervention rate p of the CBM and residual penalty of the hybrid. No test data is used.
D  Ablation contrasts with paired bootstrap CIs (echo concepts, shuffled concepts, residual, penalties, p).
E  AUC per scanner within BUS-BRA.
F  Few-shot gain over zero-shot on the SAME lesions (episode-level, from 09 section 5), with 95% intervals.

Output: <work_root>/stats/{main_table.csv, noninferiority.csv, selection.csv, contrasts.csv, scanners.csv,
        fewshot_gain.csv, report.txt}
Usage:  python 15_stats.py [--boot 2000]
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_cfg

warnings.filterwarnings("ignore")
RUN_RE = re.compile(r"^(bbc|bb|mt|cbm|seq|shuf|hyb)_s(\d+)_(.+)$")
DS = ["BUS-BRA", "BUS-UCLM", "BUSB", "BrEaST", "BUSI"]
BACKBONES = {"dinov2_b": "DINOv2", "biomedclip": "BiomedCLIP"}


def configs(bk):
    """display name -> (root folder, mode). Missing folders are skipped."""
    return {
        "Black box": (f"{bk}_medsam_u12", "bb"),
        "Multitask black box": (f"{bk}_medsam_noexp_ic_u12", "mt"),
        "Joint CBM p=0.25": (f"{bk}_medsam_int25_noexp_ic_u12", "cbm"),
        "Joint CBM p=0.5": (f"{bk}_medsam_int50_noexp_ic_u12", "cbm"),
        "Joint CBM p=0.75": (f"{bk}_medsam_int75_noexp_ic_u12", "cbm"),
        "Joint CBM p=1.0": (f"{bk}_medsam_int100_noexp_ic_u12", "cbm"),
        "Joint CBM, 6 geometry concepts": (f"{bk}_medsam_int50_noexp_u12", "cbm"),
        "Shuffled-concept CBM": (f"{bk}_medsam_int50_noexp_ic_u12", "shuf"),
        "Sequential CBM": (f"{bk}_medsam_noexp_ic_u12", "seq"),
        "Hybrid CBM, penalty 0.1": (f"{bk}_medsam_noexp_ic_u12_rp01", "hyb"),
        "Hybrid CBM, penalty 1.0": (f"{bk}_medsam_noexp_ic_u12_rp1", "hyb"),
        "Shuffled-concept hybrid": (f"{bk}_medsam_noexp_ic_u12_rp01_shufT", "hyb"),
        "Shuffled-concept multitask": (f"{bk}_medsam_noexp_ic_u12_shufT", "mt"),
    }


CONTRASTS = [  # (A, B, question)
    ("Joint CBM p=0.5", "Joint CBM, 6 geometry concepts", "echo concepts"),
    ("Joint CBM p=0.5", "Shuffled-concept CBM", "named vs shuffled concepts"),
    ("Hybrid CBM, penalty 0.1", "Sequential CBM", "residual over faithful concepts"),
    ("Hybrid CBM, penalty 0.1", "Hybrid CBM, penalty 1.0", "weaker vs stronger residual penalty"),
    ("Joint CBM p=0.5", "Joint CBM p=0.25", "intervention rate 0.5 vs 0.25"),
    ("Hybrid CBM, penalty 0.1", "Joint CBM p=0.5", "hybrid vs joint CBM"),
    ("Hybrid CBM, penalty 0.1", "Shuffled-concept hybrid", "hybrid, named vs shuffled concepts"),
    ("Multitask black box", "Shuffled-concept multitask", "multitask, named vs shuffled concepts"),
]


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def ece(y, p, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def load(root, mode):
    """held-out -> dict(df with uid,label,patient_id,device,score, seeds, val, per_seed_auc)"""
    out = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or m.group(1) != mode or not (d / "preds_test.csv").exists() or not (d / "run.json").exists():
            continue
        held = m.group(3)
        p = pd.read_csv(d / "preds_test.csv", usecols=lambda c: c in ("uid", "label", "patient_id", "device", "logit"))
        meta = json.load(open(d / "run.json"))
        e = out.setdefault(held, dict(parts=[], val=[]))
        e["parts"].append(p.set_index("uid"))
        e["val"].append(meta.get("best_val_auc", np.nan))
    for held, e in out.items():
        base = e["parts"][0][["label", "patient_id"] + (["device"] if "device" in e["parts"][0] else [])].copy()
        L = np.column_stack([q.loc[base.index, "logit"].values for q in e["parts"]])
        base["score"] = L.mean(1)  # raw-logit seed ensemble (no test-set standardisation)
        y = base.label.values.astype(int)
        e.update(df=base, seeds=L.shape[1], per_seed=[auc(y, L[:, k]) for k in range(L.shape[1])])
    return out


class Boot:
    """patient-cluster bootstrap indices, shared by every model on one held-out dataset (paired comparisons)."""

    def __init__(self, pid, B, seed):
        rng = np.random.default_rng(seed)
        codes, uniq = pd.factorize(pd.Series(pid).fillna("").astype(str))
        groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
        self.idx = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

    def aucs(self, y, s):
        return np.array([auc(y[i], s[i]) for i in self.idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    CR = cfg["work_root"] / "cbm_runs"
    out = cfg["work_root"] / "stats"
    out.mkdir(exist_ok=True)
    L = ["TIER 2 STATISTICS (raw-logit seed ensembles, patient-cluster bootstrap B=%d)" % a.boot, ""]
    main_rows, ni_rows, sel_rows, con_rows, scan_rows = [], [], [], [], []

    for bk, bname in BACKBONES.items():
        data = {name: load(CR / root, mode) for name, (root, mode) in configs(bk).items()}
        data = {k: v for k, v in data.items() if v}
        if "Black box" not in data:
            print(f"{bname}: no black-box runs found, skipped")
            continue
        print(f"{bname}: {len(data)} configurations found: {list(data)}", flush=True)
        # common lesions per held-out dataset (inner join over all configurations) and shared bootstrap draws
        boots, Y, S = {}, {}, {}
        for held in [d for d in DS if d in data["Black box"]]:
            have = [k for k in data if held in data[k]]
            uids = set.intersection(*[set(data[k][held]["df"].index) for k in have])
            ref = data["Black box"][held]["df"].loc[sorted(uids)]
            Y[held] = ref.label.values.astype(int)
            boots[held] = Boot(ref.patient_id.values, a.boot, seed=sum(map(ord, held)))
            for k in have:
                S[(k, held)] = data[k][held]["df"].loc[ref.index, "score"].values
            if "device" in ref and held == "BUS-BRA":
                for k in have:
                    for dev, g in ref.groupby(ref.device.fillna("unknown").astype(str)):
                        if len(g) >= 30 and 0 < g.label.sum() < len(g):
                            ii = ref.index.get_indexer(g.index)
                            scan_rows.append(dict(backbone=bname, model=k, scanner=dev, n=len(g),
                                                  malignant=int(g.label.sum()), auc=auc(Y[held][ii], S[(k, held)][ii])))
        BA = {}  # bootstrap AUCs per (model, held-out)

        def bauc(k, held):
            if (k, held) not in BA:
                BA[(k, held)] = boots[held].aucs(Y[held], S[(k, held)])
            return BA[(k, held)]

        # A main table
        for k, dd in data.items():
            r = dict(backbone=bname, model=k)
            vals = []
            for held in [d for d in DS if (k, d) in S]:
                y, s = Y[held], S[(k, held)]
                p = 1 / (1 + np.exp(-s))
                b = bauc(k, held)
                r[held] = auc(y, s)
                r[f"{held}_ci"] = f"[{np.nanpercentile(b, 2.5):.3f}, {np.nanpercentile(b, 97.5):.3f}]"
                vals.append((held, r[held], np.std(dd[held]["per_seed"]), ece(y, p),
                             ((p >= 0.5) & (y == 1)).sum() / max(y.sum(), 1), ((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)))
            if not vals:
                continue
            r["mean_all"] = np.mean([v[1] for v in vals]) if len(vals) == 5 else np.nan
            r["mean_noBUSI"] = np.mean([v[1] for v in vals if v[0] != "BUSI"])
            r["seed_sd"] = np.mean([v[2] for v in vals])
            r["val_auc"] = np.nanmean([x for d in dd.values() for x in d["val"]])
            r["ECE"] = np.mean([v[3] for v in vals])
            r["sens@0.5"] = np.mean([v[4] for v in vals])
            r["spec@0.5"] = np.mean([v[5] for v in vals])
            main_rows.append(r)

        # B non-inferiority vs black box, pooled over datasets (equal weight)
        def contrast(A, Bm):
            hs = [h for h in DS if (A, h) in S and (Bm, h) in S]
            per, pooled_all, pooled_nob = {}, [], []
            for h in hs:
                d = bauc(A, h) - bauc(Bm, h)
                per[h] = (auc(Y[h], S[(A, h)]) - auc(Y[h], S[(Bm, h)]), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
            D = np.column_stack([bauc(A, h) - bauc(Bm, h) for h in hs])
            nob = [i for i, h in enumerate(hs) if h != "BUSI"]
            return hs, per, D.mean(1), D[:, nob].mean(1)

        for k in data:
            if k == "Black box":
                continue
            hs, per, pall, pnob = contrast(k, "Black box")
            r = dict(backbone=bname, model=k)
            for h, (dlt, lo, hi) in per.items():
                r[h] = f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            for tag, pv in (("pooled_noBUSI", pnob), ("pooled_all", pall)):
                r[tag] = f"{np.nanmean(pv):+.3f} [{np.nanpercentile(pv, 2.5):+.3f}, {np.nanpercentile(pv, 97.5):+.3f}]"
            lo_nb = np.nanpercentile(pnob, 2.5)
            r["noninferior_0.02"] = bool(lo_nb > -0.02)
            r["noninferior_0.03"] = bool(lo_nb > -0.03)
            # G: sensitivity of the pooled estimate. (i) leave one site out, (ii) DerSimonian-Laird random effects
            # over the four sites with patient identifiers (between-site heterogeneity tau^2)
            h4 = [h for h in hs if h != "BUSI"]
            if len(h4) >= 3:
                Dm = np.column_stack([bauc(k, h) - bauc("Black box", h) for h in h4])
                loso = []
                for i, h in enumerate(h4):
                    keep = [j for j in range(len(h4)) if j != i]
                    pv = Dm[:, keep].mean(1)
                    loso.append((h, np.nanmean(pv), np.nanpercentile(pv, 2.5), np.nanpercentile(pv, 97.5)))
                r["loso_worst_lower"] = min(q[2] for q in loso)
                r["loso"] = "; ".join(f"without {h} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]" for h, m, lo, hi in loso)
                y = np.array([per[h][0] for h in h4])
                v = np.array([np.nanvar(Dm[:, j]) for j in range(len(h4))])
                w = 1 / np.maximum(v, 1e-12)
                ybar = (w * y).sum() / w.sum()
                Q = (w * (y - ybar) ** 2).sum()
                tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
                ws = 1 / (v + tau2)
                re = (ws * y).sum() / ws.sum()
                se = np.sqrt(1 / ws.sum())
                r["random_effects"] = f"{re:+.3f} [{re - 1.96 * se:+.3f}, {re + 1.96 * se:+.3f}]"
                r["tau2"] = tau2
                r["re_noninferior_0.02"] = bool(re - 1.96 * se > -0.02)
            r["seed_sd_model"] = float(np.mean([np.std(data[k][h]["per_seed"]) for h in hs if h in data[k]]))
            ni_rows.append(r)

        # C selection on internal validation only
        for fam, members in (("CBM intervention rate p", [m for m in data if m.startswith("Joint CBM p=")]),
                             ("Hybrid residual penalty", [m for m in data if m.startswith("Hybrid")])):
            if len(members) < 2:
                continue
            v = {m: np.nanmean([x for d in data[m].values() for x in d["val"]]) for m in members}
            best = max(v, key=v.get)
            for m in members:
                sel_rows.append(dict(backbone=bname, family=fam, model=m, internal_val_auc=v[m], selected=(m == best)))

        # D ablation contrasts
        for A, Bm, q in CONTRASTS:
            if A in data and Bm in data:
                hs, per, pall, pnob = contrast(A, Bm)
                r = dict(backbone=bname, question=q, A=A, B=Bm)
                for h, (dlt, lo, hi) in per.items():
                    r[h] = f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                r["pooled_noBUSI"] = f"{np.nanmean(pnob):+.3f} [{np.nanpercentile(pnob, 2.5):+.3f}, {np.nanpercentile(pnob, 97.5):+.3f}]"
                con_rows.append(r)

    # F few-shot gain over zero-shot on the same lesions (episode level, from 09)
    few_rows = []
    for bk, bname in BACKBONES.items():
        for name, (root, mode) in configs(bk).items():
            f = CR / root / "analysis" / "5_few_shot_auc_on_target.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            t = t[t["mode"] == mode]
            for (h, k), g in t.groupby(["heldout", "k"]):
                src = g[g.method == "src"].auc.values
                for meth in ("fused", "kshot_concepts", "kshot_emb"):
                    v = g[g.method == meth].auc.values
                    if len(v) == len(src) and len(v):
                        dlt = v - src
                        few_rows.append(dict(backbone=bname, model=name, heldout=h, k=k, method=meth, zero_shot=src.mean(),
                                             few_shot=v.mean(), gain=dlt.mean(), gain_lo=np.percentile(dlt, 2.5),
                                             gain_hi=np.percentile(dlt, 97.5), episodes=len(v)))

    def save(name, rows, title, cols=None):
        global L
        d = pd.DataFrame(rows)
        if d.empty:
            return d
        d.to_csv(out / f"{name}.csv", index=False)
        show = d[cols] if cols else d
        L += [title, show.round(3).to_string(index=False), ""]
        return d

    mt = save("main_table", main_rows, "A  MAIN TABLE (AUC; mean_noBUSI = BUS-BRA, BUS-UCLM, BUSB, BrEaST)",
              ["backbone", "model"] + [d for d in DS] + ["mean_noBUSI", "mean_all", "seed_sd", "val_auc", "ECE", "sens@0.5", "spec@0.5"])
    save("noninferiority", ni_rows, "B  PAIRED DIFFERENCE VS BLACK BOX [95% CI]; pooled = equal-weight mean over datasets; non-inferior if pooled_noBUSI lower bound > -margin",
         ["backbone", "model", "pooled_noBUSI", "pooled_all", "noninferior_0.02", "noninferior_0.03"])
    if ni_rows and "random_effects" in ni_rows[0]:
        g = pd.DataFrame(ni_rows)
        L += ["G  SENSITIVITY OF THE PAIRED DIFFERENCE VS BLACK BOX: random effects over the 4 sites (DerSimonian-Laird), "
              "worst leave-one-site-out lower bound, between-seed SD",
              g[["backbone", "model", "random_effects", "tau2", "re_noninferior_0.02", "loso_worst_lower", "seed_sd_model"]]
              .round(4).to_string(index=False), "", "G2 LEAVE ONE SITE OUT (pooled over the other three)",
              g[["backbone", "model", "loso"]].to_string(index=False), ""]
    save("selection", sel_rows, "C  SELECTION ON INTERNAL VALIDATION ONLY (training datasets)")
    save("contrasts", con_rows, "D  ABLATION CONTRASTS (A minus B) [95% CI]", ["backbone", "question", "pooled_noBUSI"] +
         [d for d in DS if any(d in r for r in con_rows)])
    save("scanners", scan_rows, "E  AUC PER SCANNER WITHIN BUS-BRA")
    fg = pd.DataFrame(few_rows)
    if len(fg):
        fg.to_csv(out / "fewshot_gain.csv", index=False)
        s = fg[(fg.method == "fused") & fg.k.isin([4, 16])]
        L += ["F  FEW-SHOT GAIN OVER ZERO-SHOT ON THE SAME LESIONS (fused; mean over 100 episodes, 95% interval)",
              s[["backbone", "model", "heldout", "k", "zero_shot", "few_shot", "gain", "gain_lo", "gain_hi"]].round(3).to_string(index=False), ""]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"saved {out}")
