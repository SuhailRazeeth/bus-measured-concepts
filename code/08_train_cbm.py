"""Pilot 3: trained BI-RADS concept-bottleneck model, leave-one-dataset-out over ultrasound datasets (GPU).

Modes (same backbone, same partial fine-tuning, same data):
  bb    black box          image -> encoder -> linear -> malignancy
  mt    multi-task          bb + auxiliary concept head (concepts supervised, diagnosis still from features)
  cbm   concept bottleneck  image -> encoder -> concepts -> linear -> malignancy   (the proposed model)
  seq   sequential CBM      same, but the diagnosis loss never reaches the encoder (concepts are detached), so the
                            bottleneck can only carry what the concept loss puts there: leakage-free reference
  shuf  shuffled control    cbm trained on concept targets permuted across training lesions (tests whether the
                            named concepts matter or the bottleneck just acts as an 11-d task channel)
  hyb   hybrid CBM          logit = faithful concept part (detached, as seq) + residual from the image features,
                            with an L2 penalty --res_pen on the residual; 09 reports the concept share
  bbc   black box + measured concepts as input (no concept head loss). Fair reference for the correction experiment:
                            it can receive the same radiologist-mask measurements at test time as the corrected models
  seq with --int_p 1.0  =  independent CBM (Koh et al.): the diagnosis head is trained on the measured targets only
  --box_jitter 0.1 0.2      after training, also test on ROIs re-cropped from shifted / rescaled lesion boxes

Concepts = 6 mask-geometry concepts (z-scored per dataset; from radiologist masks or SAM/MedSAM masks via --geom)
         + 2 expert BI-RADS concepts (shape irregular, margin not circumscribed) where available (BrEaST, CBIS-DDSM).
For each held-out US dataset the model is trained on all other US datasets (+ CBIS-DDSM with --mg) and tested on the
held-out one with ZERO target labels. Predictions, concepts and embeddings are saved for 09_analyze_cbm.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 08_train_cbm.py --backbone biomedclip --geom medsam --modes bb mt cbm --seeds 0 1 2
  options: --heldout BUSB BrEaST   --mg   --epochs 20   --unfreeze 2
  --int_p 0.25   intervention-aware training (cbm): each geometry concept fed to the diagnosis head is replaced by
                 its training label with probability p, so the head learns to use true concept values (less leakage)
  --no_expert    bottleneck = 6 geometry concepts only (the 2 expert concepts are unsupervised in US-only training
                 when BrEaST is held out and can act as a free "leak" channel)
  --unfreeze 12  with --modes bb = full fine-tuning baseline
  --intensity    add the 5 label-free echo concepts from 12_intensity_concepts.py (11 continuous concepts)
Run folder: cbm_runs/<backbone>_<geom>[_mg][_int25][_noexp][_ic][_u<N> if N != config]
"""
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common import GEOM_COLS, INT_COLS, load_cfg, mask_bbox, read_gray, read_mask, roi_manifest_path, square_crop

BOTTLENECK = ("cbm", "seq", "shuf", "hyb")  # modes whose diagnosis head reads the concepts
DETACHED = ("seq", "hyb")  # concept part of the logit gets no task gradient (faithful, leakage-free concept part)

EXPERT = ["c_shape_irregular", "c_margin_noncirc"]
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
IMNET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ----------------------------------------------------------------------------- backbone
def build_encoder(name: str, n_unfreeze: int, img_size: int):
    import timm

    if name == "biomedclip":
        import open_clip

        m, _ = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        enc, dim, norm = m.visual, 512, CLIP_NORM
        blocks, extra = list(enc.trunk.blocks), [enc.trunk.norm, enc.head]
    elif name.startswith("dinov2"):
        tn = {"dinov2_b": "vit_base_patch14_dinov2.lvd142m", "dinov2_s": "vit_small_patch14_dinov2.lvd142m"}[name]
        enc = timm.create_model(tn, pretrained=True, num_classes=0, img_size=img_size)
        dim, norm, blocks, extra = enc.num_features, IMNET_NORM, list(enc.blocks), [enc.norm]
    elif name.startswith("resnet"):
        tn = {"resnet50_in": "resnet50.a1_in1k", "resnet18_in": "resnet18.a1_in1k"}[name]
        enc = timm.create_model(tn, pretrained=True, num_classes=0)
        dim, norm, blocks, extra = enc.num_features, IMNET_NORM, [enc.layer3, enc.layer4], []
    else:
        raise ValueError(name)
    for p in enc.parameters():
        p.requires_grad = False
    for mod in (blocks[-n_unfreeze:] if n_unfreeze > 0 else []) + extra:
        for p in mod.parameters():
            p.requires_grad = True
    return enc, dim, norm


class Net(nn.Module):
    def __init__(self, enc, dim, mode, n_concepts):
        super().__init__()
        self.enc, self.mode = enc, mode
        self.concept = nn.Linear(dim, n_concepts)
        self.diag = nn.Linear(n_concepts if mode in BOTTLENECK else dim, 1)
        self.res = nn.Linear(dim, 1) if mode == "hyb" else None  # hyb: explicit residual from the image features
        # bbc: black box that also receives the MEASURED concepts as an input (logit = w_h^T h + w_c^T t). At test time the
        # measured values can come from MedSAM or from the radiologist mask, which gives the black box the same
        # information that the corrected concept models receive (fair baseline for the correction experiment)
        self.cin = nn.Linear(n_concepts, 1, bias=False) if mode == "bbc" else None
        self.res_out = None

    def forward(self, x, cin=None):
        h = self.enc(x).float()
        c = self.concept(h)
        # seq: sequential CBM, the task loss never reaches the encoder (no leakage channel through the concepts)
        z = c.detach() if self.mode in DETACHED else c
        logit = self.diag(z if self.mode in BOTTLENECK else h).squeeze(1)
        self.res_out = None
        if self.res is not None:  # logit = concept part (named, faithful) + residual part (reported separately)
            self.res_out = self.res(h).squeeze(1)
            logit = logit + self.res_out
        if self.cin is not None:  # bbc: the "concepts" are the measured input values themselves
            c = cin.float()
            logit = logit + self.cin(c).squeeze(1)
        return logit, c, h


def concept_input(g, e):
    """measured concept vector fed to the bbc model (missing values, e.g. BUSI without masks, set to the dataset mean 0)"""
    return torch.nan_to_num(torch.cat([g, e], 1))


# ----------------------------------------------------------------------------- data
class RoiSet(Dataset):
    def __init__(self, imgs, y, geo, exp, tf):
        self.imgs, self.y, self.geo, self.exp, self.tf = imgs, y, geo, exp, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(Image.fromarray(self.imgs[i]).convert("RGB"))
        return x, torch.tensor(self.y[i], dtype=torch.float32), torch.tensor(self.geo[i], dtype=torch.float32), \
            torch.tensor(self.exp[i], dtype=torch.float32)


def transforms(size, norm):
    train = T.Compose([T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                       T.RandomRotation(10), T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(*norm)])
    test = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(*norm)])
    return train, test


def patient_split(pids, frac, rng):
    u = np.array(sorted(set(pids)))
    rng.shuffle(u)
    val = set(u[: max(1, int(len(u) * frac))])
    return np.array([p in val for p in pids])


def masked_losses(c, geo, exp, n_geo):
    g, e = c[:, :n_geo], c[:, n_geo:]
    gm, em = ~torch.isnan(geo), ~torch.isnan(exp)
    lg = Fn.smooth_l1_loss(g[gm], geo[gm]) if gm.any() else c.sum() * 0
    le = Fn.binary_cross_entropy_with_logits(e[em], exp[em]) if em.any() else c.sum() * 0
    return lg, le


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    L, C, H = [], [], []
    for x, _, g, e in loader:
        cin = concept_input(g, e).to(dev) if model.cin is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lo, c, h = model(x.to(dev, non_blocking=True), cin)
        L.append(lo.float().cpu()), C.append(c.float().cpu()), H.append(h.float().cpu())
    return torch.cat(L).numpy(), torch.cat(C).numpy(), torch.cat(H).numpy()


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


# ----------------------------------------------------------------------------- one run
def run_one(df, imgs, args, tc, mode, seed, heldout, out_dir, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    amp = dev == "cuda"
    us_train = (df.modality == "US") & (df.dataset != heldout)
    tr_mask = us_train | ((df.dataset == "CBIS-DDSM") & (df.official_split == "train") if args.mg else False)
    tr_idx = np.flatnonzero(tr_mask.values)
    val_flag = patient_split(df.patient_id.values[tr_idx], tc["val_fraction"], rng)
    te_idx = np.flatnonzero((df.dataset == heldout).values)
    extra_te = {}
    if args.mg:
        for d in ["INbreast", "MIAS"]:
            extra_te[d] = np.flatnonzero(((df.dataset == d) & (df.lesion_type == "mass")).values)
        extra_te["CBIS-DDSM(test)"] = np.flatnonzero(((df.dataset == "CBIS-DDSM") & (df.official_split == "test")).values)

    enc, dim, norm = build_encoder(args.backbone, args.unfreeze, tc["img_size"])
    expert = [] if args.no_expert else EXPERT
    CC = args.cc  # continuous (label-free) concepts: geometry [+ intensity]
    n_geo, n_c = len(CC), len(CC) + len(expert)
    model = Net(enc, dim, mode, n_c).to(dev)
    tf_tr, tf_te = transforms(tc["img_size"], norm)
    Y, G = df.label.values.astype(np.float32), df[[f"t_{c}" for c in CC]].values.astype(np.float32)
    E = df[expert].values.astype(np.float32) if expert else np.zeros((len(df), 0), np.float32)
    int_p = args.int_p if mode in BOTTLENECK else 0.0
    if mode == "shuf" or args.shuffle:  # control: concept targets permuted across training lesions (same values, no link to the image)
        G, E = G.copy(), E.copy()
        prng = np.random.default_rng(1000 + seed)
        for d in df.dataset.iloc[tr_idx].unique():
            ii = tr_idx[(df.dataset.iloc[tr_idx] == d).values]
            G[ii], E[ii] = G[prng.permutation(ii)], E[prng.permutation(ii)]

    def loader(idx, tf, shuffle):
        return DataLoader(RoiSet([imgs[i] for i in idx], Y[idx], G[idx], E[idx], tf), batch_size=tc["batch_size"],
                          shuffle=shuffle, num_workers=args.workers, pin_memory=dev == "cuda", drop_last=shuffle)

    dl_tr = loader(tr_idx[~val_flag], tf_tr, True)
    dl_va = loader(tr_idx[val_flag], tf_te, False)
    y_va = Y[tr_idx[val_flag]]
    head_params = list(model.concept.parameters()) + list(model.diag.parameters())
    if model.res is not None:
        head_params += list(model.res.parameters())
    if model.cin is not None:
        head_params += list(model.cin.parameters())
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tc["lr_backbone"]},
                             {"params": head_params, "lr": tc["lr_head"]}], weight_decay=tc["weight_decay"])
    epochs = args.epochs or tc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl_tr))
    ytr = Y[tr_idx[~val_flag]]
    pos_w = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-3), device=dev)
    lam_c = 0.0 if mode in ("bb", "bbc") else tc["lambda_concept"]
    lam_e = 0.0 if mode in ("bb", "bbc") else tc["lambda_expert"]
    best, best_state, bad, log = -1, None, 0, []
    keep = {f"enc.{n}" for n, p in model.enc.named_parameters() if p.requires_grad}
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, g, e in dl_tr:
            x, y, g, e = (t.to(dev, non_blocking=True) for t in (x, y, g, e))
            cin = concept_input(g, e) if model.cin is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lo, c, _ = model(x, cin)
            lo, c = lo.float(), c.float()
            r = model.res_out.float() if model.res_out is not None else None
            if int_p > 0:  # random concept intervention during training
                cg = c[:, :n_geo].detach() if mode in DETACHED else c[:, :n_geo]
                swap = (torch.rand_like(cg) < int_p) & ~torch.isnan(g)
                cg = torch.where(swap, torch.nan_to_num(g), cg)
                ce = c[:, n_geo:].detach() if mode in DETACHED else c[:, n_geo:]
                lo = model.diag(torch.cat([cg, ce], 1)).squeeze(1) + (r if r is not None else 0)
            ld =Fn.binary_cross_entropy_with_logits(lo, y, pos_weight=pos_w)
            lg, le = masked_losses(c, g, e, n_geo) if lam_c or lam_e else (ld * 0, ld * 0)
            loss = ld + lam_c * lg + lam_e * le
            if r is not None:  # keep the residual small so the named concepts carry what they can
                loss = loss + args.res_pen * (r ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        lv, _, _ = predict(model, dl_va, dev, amp)
        auc = safe_auc(y_va, lv)
        log.append(dict(epoch=ep, loss=tot / max(len(dl_tr), 1), val_auc=auc, sec=time.time() - t0))
        print(f"    ep{ep:02d} loss {log[-1]['loss']:.3f} val_auc {auc:.3f} ({log[-1]['sec']:.0f}s)", flush=True)
        score = auc if auc == auc else -log[-1]["loss"]  # fall back to train loss if val AUC undefined
        if score > best + 1e-4 or best_state is None:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k.startswith(("concept", "diag", "res")) or k in keep}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state, strict=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in {heldout: te_idx, **extra_te}.items():
        if len(idx) == 0:
            continue
        lo, c, h = predict(model, loader(idx, tf_te, False), dev, amp)
        p = df.iloc[idx][["uid", "dataset", "label", "patient_id", "device", "birads"] + EXPERT +
                         [f"gt_{g}" for g in CC] + [f"t_{g}" for g in CC]].copy()
        p["logit"] = lo
        for j in range(n_c):
            p[f"c{j}"] = c[:, j]
        tag = "test" if name == heldout else f"extra_{name}"
        p.to_csv(out_dir / f"preds_{tag}.csv", index=False)
        np.save(out_dir / f"emb_{tag}.npy", h.astype(np.float16))
        print(f"    {name:16s} AUC {safe_auc(p.label.values, lo):.3f}  (n={len(idx)})")
    # test-time box robustness: re-crop the held-out ROIs from the original image with a shifted / rescaled box
    rc = args.roi_cfg
    for j in args.box_jitter:
        jr = np.random.default_rng(int(round(j * 1000)))
        jimgs = []
        for i in te_idx:
            r = df.iloc[i]
            mp = r.get("mask_path")
            if not (isinstance(mp, str) and mp):
                jimgs.append(imgs[i])  # no mask (whole-image ROI): unchanged
                continue
            im = read_gray(r["image_path"])
            m = read_mask(mp)
            if m.shape != im.shape:
                m = cv2.resize(m, im.shape[::-1], interpolation=cv2.INTER_NEAREST)
            x0, y0, x1, y1 = mask_bbox(m)
            w, h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2 + jr.uniform(-j, j) * w, (y0 + y1) / 2 + jr.uniform(-j, j) * h
            sw, sh = w * jr.uniform(1 - j, 1 + j), h * jr.uniform(1 - j, 1 + j)
            box = (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2)
            jimgs.append(square_crop(im, box, rc["context"], rc["min_box_px"], rc["out_size"])[0])
        dl_j = DataLoader(RoiSet(jimgs, Y[te_idx], G[te_idx], E[te_idx], tf_te), batch_size=tc["batch_size"],
                          shuffle=False, num_workers=args.workers)
        lo, c, _ = predict(model, dl_j, dev, amp)
        p = df.iloc[te_idx][["uid", "dataset", "label", "patient_id"]].copy()
        p["logit"] = lo
        for k in range(n_c):
            p[f"c{k}"] = c[:, k]
        p.to_csv(out_dir / f"preds_test_jit{round(j * 100)}.csv", index=False)
        print(f"    box jitter {j:.2f}: AUC {safe_auc(p.label.values, lo):.3f}")
    # bbc: diag_w holds the weights of the measured-concept input, so z' = z + w^T (t_new - t) swaps the input source
    wd = model.cin.weight if model.cin is not None else model.diag.weight
    json.dump(dict(diag_w=wd.detach().cpu().numpy().tolist(),
                   diag_b=model.diag.bias.detach().cpu().numpy().tolist(),
                   concepts=CC + expert, mode=mode, int_p=int_p, unfreeze=args.unfreeze, res_pen=args.res_pen,
                   best_val_auc=best, log=log),
              open(out_dir / "run.json", "w"), indent=1)
    del model, enc
    if dev == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backbone", default="biomedclip")
    ap.add_argument("--geom", default="medsam", help="gt | sam | medsam : source of the concept training labels")
    ap.add_argument("--modes", nargs="*", default=["bb", "mt", "cbm"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--heldout", nargs="*")
    ap.add_argument("--mg", action="store_true", help="also train on CBIS-DDSM masses (shared MG+US concept model)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--unfreeze", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--int_p", type=float, default=0.0, help="intervention-aware training probability (cbm)")
    ap.add_argument("--no_expert", action="store_true", help="drop the 2 expert concepts from the bottleneck")
    ap.add_argument("--intensity", action="store_true", help="add the 5 echo concepts (run 12_intensity_concepts.py first)")
    ap.add_argument("--box_jitter", nargs="*", type=float, default=[],
                    help="also test with ROIs re-cropped from boxes shifted/rescaled by up to this fraction, e.g. 0.1 0.2")
    ap.add_argument("--res_pen", type=float, default=0.1, help="hyb: L2 penalty on the residual logit")
    ap.add_argument("--shuffle", action="store_true", help="permute the concept targets for ANY mode (shuffled control for hyb / mt); use with --suffix")
    ap.add_argument("--suffix", default="", help="extra run-folder suffix, e.g. _rb for the box-robustness reruns")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    tc = cfg["train"]
    a.unfreeze = tc["unfreeze_blocks"] if a.unfreeze is None else a.unfreeze
    a.roi_cfg = cfg["roi"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(roi_manifest_path(cfg))
    df = df[(df.lesion_type == "mass") & df.label.notna()].reset_index(drop=True)
    for c in ["device", "birads"]:
        df[c] = df[c].fillna("").astype(str)
    for c in GEOM_COLS:
        df[f"gt_{c}"] = df[c]
    if a.geom != "gt":
        sam = pd.read_csv(cfg["work_root"] / a.geom / "sam_geometry.csv")[["uid"] + GEOM_COLS]
        df = df.drop(columns=GEOM_COLS).merge(sam, on="uid", how="left")
    a.cc = GEOM_COLS + (INT_COLS if a.intensity else [])
    if a.intensity:  # training labels from the --geom masks, gt_ (evaluation / intervention) from radiologist masks
        idir = cfg["work_root"] / "intensity"
        df = df.merge(pd.read_csv(idir / f"{a.geom}.csv")[["uid"] + INT_COLS], on="uid", how="left")
        gti = pd.read_csv(idir / "gt.csv")[["uid"] + INT_COLS].rename(columns={c: f"gt_{c}" for c in INT_COLS})
        df = df.merge(gti, on="uid", how="left")
    # label-free per-dataset z-scoring of the concept targets
    for c in a.cc:
        df[f"t_{c}"] = df.groupby("dataset")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
        df[f"gt_{c}"] = df.groupby("dataset")[f"gt_{c}"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    for c in EXPERT:
        df[c] = df[c].astype(float)
    # CBIS-DDSM leakage fix (patients in both official splits)
    cb = df.dataset == "CBIS-DDSM"
    trp = set(df.patient_id[cb & (df.official_split == "train")])
    df = df[~(cb & (df.official_split == "test") & df.patient_id.isin(trp))].reset_index(drop=True)
    if not a.mg:
        df = df[df.modality == "US"].reset_index(drop=True)

    us_sets = [d for d in cfg["pilot"]["us_targets"] if (df.dataset == d).any()]
    heldouts = a.heldout or us_sets
    print(f"US datasets: {us_sets}   held-out runs: {heldouts}   device: {dev}")
    print(df.groupby("dataset").agg(n=("label", "size"), malignant=("label", "sum"),
                                    geom=(f"t_{GEOM_COLS[0]}", "count"), expert=(EXPERT[0], "count")))
    print("loading ROIs into memory ...", flush=True)
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in df.roi_path]

    tag = f"{a.backbone}_{a.geom}{'_mg' if a.mg else ''}"
    tag += f"_int{round(a.int_p * 100)}" if a.int_p > 0 else ""
    tag += "_noexp" if a.no_expert else ""
    tag += "_ic" if a.intensity else ""
    tag += f"_u{a.unfreeze}" if a.unfreeze != tc["unfreeze_blocks"] else ""
    tag += a.suffix
    root = cfg["work_root"] / "cbm_runs" / tag
    for heldout in heldouts:
        for mode in a.modes:
            for seed in a.seeds:
                od = root / f"{mode}_s{seed}_{heldout}"
                if (od / "run.json").exists():
                    print(f"skip {od.name} (done)")
                    continue
                print(f"== {root.name} | held-out {heldout} | {mode} | seed {seed}", flush=True)
                run_one(df, imgs, a, tc, mode, seed, heldout, od, dev)
    print(f"all runs in {root}")
