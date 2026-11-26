"""One table for the paper: zero-shot (leave-one-dataset-out) seed-ensemble AUC of every 08 setup/mode + 10 baselines.

Reads <work_root>/cbm_runs/*/analysis/1_zero_shot_auc.csv (run 09 first) and <work_root>/baselines/baselines.csv.
Output: <work_root>/summary/{main_table.csv, main_table.txt}
Usage:  python 11_summary.py
"""
import argparse

import pandas as pd

from common import load_cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    rows = []
    for f in sorted((cfg["work_root"] / "cbm_runs").glob("*/analysis/1_zero_shot_auc.csv")):
        setup = f.parent.parent.name
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            if any(t in setup for t in ("_int", "_noexp", "_ic")) and r["mode"] != "cbm":
                continue  # bb/mt there are borrowed copies of the base setup
            rows.append(dict(model=f"{setup}:{r['mode']}", heldout=r["heldout"], auc=r["ens_auc"], ci=r.get("ens_ci", "")))
    bf = cfg["work_root"] / "baselines" / "baselines.csv"
    if bf.exists():
        for _, r in pd.read_csv(bf).iterrows():
            rows.append(dict(model=f"baseline:{r['backbone']}:{r['method']}", heldout=r["heldout"], auc=r["auc"], ci=r["ci"]))
    t = pd.DataFrame(rows).drop_duplicates(["model", "heldout"])
    out = cfg["work_root"] / "summary"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "main_table.csv", index=False)
    piv = t.pivot_table(index="model", columns="heldout", values="auc")
    piv["mean"] = piv.mean(axis=1)
    txt = piv.sort_values("mean", ascending=False).round(3).to_string()
    (out / "main_table.txt").write_text(txt + "\n")
    print(txt + f"\n\nsaved {out}")
