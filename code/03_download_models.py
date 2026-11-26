"""Step 3: download all backbone weights into hf_home. Run this on the LOGIN node (compute nodes often have no internet).

Usage:  python 03_download_models.py
"""
import os

from common import load_cfg

cfg = load_cfg()
os.environ["HF_HOME"] = str(cfg["hf_home"])
from backbones import load_backbone  # noqa: E402  (import after HF_HOME is set)

for name in cfg["features"]["backbones"]:
    print("downloading", name, "...", flush=True)
    load_backbone(name, "cpu")
    print("  ok")
print("all weights cached in", cfg["hf_home"])
