"""Step 4: extract frozen features for every ROI with every backbone, cache to disk.

Outputs  <work_root>/features/<backbone>.npz  with arrays  uid, feat (L2-normalised)
and, for VLM backbones, text concept embeddings  text_<modality>_<concept>_{pos,neg}.

Usage:  python 04_extract_features.py [--backbones biomedclip rad_dino]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import load_cfg, roi_manifest_path


class RoiDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backbones", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")  # use cache from step 3
    from backbones import CONCEPT_PROMPTS, TEMPLATES, load_backbone

    df = pd.read_csv(roi_manifest_path(cfg))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["work_root"] / "features"
    out_dir.mkdir(exist_ok=True)
    for name in a.backbones or cfg["features"]["backbones"]:
        print(f"== {name} on {dev}")
        model, prep, embed, text = load_backbone(name, dev)
        dl = DataLoader(RoiDS(df.roi_path.tolist(), prep), batch_size=cfg["features"]["batch_size"],
                        num_workers=a.workers)
        feats = []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for x in tqdm(dl):
                feats.append(embed(x.to(dev)).float().cpu())
        f = torch.cat(feats)
        f = torch.nn.functional.normalize(f, dim=1).numpy().astype(np.float32)
        extra = {}
        if text is not None:
            with torch.no_grad():
                for mod, tpl in TEMPLATES.items():
                    for c, (pos, neg) in CONCEPT_PROMPTS.items():
                        t = text([tpl.format(pos), tpl.format(neg)]).float()
                        t = torch.nn.functional.normalize(t, dim=1).cpu().numpy()
                        extra[f"text_{mod}_{c}_pos"], extra[f"text_{mod}_{c}_neg"] = t[0], t[1]
        np.savez_compressed(out_dir / f"{name}.npz", uid=np.array(df.uid.tolist(), dtype="U"), feat=f, **extra)
        print(f"  saved {f.shape} -> {out_dir / (name + '.npz')}")
        del model
        torch.cuda.empty_cache()
