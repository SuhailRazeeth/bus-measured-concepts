"""Step 0: print what is inside each dataset folder so the manifest builder can be checked/adjusted.

Usage:  python 00_inventory.py  > inventory.txt
"""
from collections import Counter

from common import ds_dir, load_cfg

cfg = load_cfg()
print(f"data_root = {cfg['data_root']}\n")
for name in list(cfg["datasets"]) + ["MRI"]:
    d = cfg["data_root"] / cfg["datasets"].get(name, name)
    print("=" * 80, f"\n{name}  ->  {d}")
    if not d.exists():
        print("  !! MISSING")
        continue
    files = [p for p in d.rglob("*") if p.is_file()]
    print(f"  files: {len(files)}")
    print("  extensions:", dict(Counter(p.suffix.lower() for p in files).most_common(10)))
    dirs = Counter(str(p.parent.relative_to(d)) for p in files)
    print(f"  folders with files: {len(dirs)} (top 12 shown)")
    for k, v in dirs.most_common(12):
        print(f"    {v:6d}  {k}")
    tables = [p for p in files if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt", ".json"}]
    print("  tables/metadata:")
    for p in tables[:15]:
        print("    ", p.relative_to(d))
        if p.suffix.lower() in {".csv", ".txt"}:
            try:
                with open(p, "r", errors="ignore") as f:
                    for _ in range(3):
                        print("        |", f.readline().rstrip()[:160])
            except Exception as e:
                print("        (unreadable)", e)
    print("  sample files:")
    for p in files[:8]:
        print("    ", p.relative_to(d))
