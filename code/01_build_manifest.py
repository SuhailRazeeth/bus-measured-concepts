"""Step 1: build one lesion-level manifest (CSV) across all datasets.

One row = one lesion. Columns:
  uid, dataset, modality (MG/US), lesion_type, image_path, mask_path, circle (MIAS cx,cy,r),
  label (0 benign / 1 malignant), patient_id, birads, device, official_split,
  c_shape_irregular, c_margin_noncirc, c_margin_spiculated   (expert concepts; NaN if unknown)

Each dataset has its own parser that tries the common public layouts. If a parser finds nothing,
it prints what it expected; run 00_inventory.py and adjust the parser or config.

Usage:  python 01_build_manifest.py [--only BUSI BUS-BRA]
"""
from __future__ import annotations

import argparse
import plistlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import ds_dir, list_images, load_cfg, manifest_path, is_binary_image

NAN = np.nan


def row(**kw):
    base = dict(uid="", dataset="", modality="", lesion_type="mass", image_path="", mask_path="", circle="",
                label=NAN, patient_id="", birads="", device="", official_split="",
                c_shape_irregular=NAN, c_margin_noncirc=NAN, c_margin_spiculated=NAN)
    base.update(kw)
    return base


def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p, sep=None, engine="python")


# ============================================================================ BUSI
def parse_busi(root: Path) -> list[dict]:
    rows = []
    imgs = [p for p in list_images(root) if "_mask" not in p.stem.lower()]
    for p in imgs:
        cls = p.parent.name.lower()
        if cls not in {"benign", "malignant"}:
            continue  # 'normal' has no lesion
        masks = sorted(p.parent.glob(p.stem + "_mask*" + p.suffix))
        mask_path = masks[0] if masks else ""  # no mask -> whole image is used as ROI, no geometry
        if len(masks) > 1:  # several lesions drawn in separate masks -> OR them
            m = np.zeros_like(cv2.imread(str(masks[0]), 0))
            for q in masks:
                m = np.maximum(m, cv2.imread(str(q), 0))
            out = WORK / "busi_merged_masks" / (p.stem + "_mask.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), m)
            mask_path = out
        uid = f"BUSI/{cls}/{p.stem}"
        # BUSI has no patient IDs: each image is its own group (known limitation; dataset has near-duplicates)
        rows.append(row(uid=uid, dataset="BUSI", modality="US", image_path=str(p), mask_path=str(mask_path),
                        label=int(cls == "malignant"), patient_id=uid))
    return rows


# ============================================================================ BUS-BRA
def parse_busbra(root: Path) -> list[dict]:
    tables = [p for p in root.rglob("*.csv")] + [p for p in root.rglob("*.xlsx")]
    meta = None
    for t in tables:
        df = read_table(t)
        cols = {c.lower(): c for c in df.columns}
        if "pathology" in cols and "id" in cols:
            meta = df.rename(columns={v: k for k, v in cols.items()})
            break
    if meta is None:
        print("  BUS-BRA: no csv with ID + Pathology columns (expected bus_data.csv)")
        return []
    # skip derived copies such as busbra_birads/<class>/ -- use the original Images/ + Masks/ folders
    files = {p.stem: p for p in list_images(root) if "birads" not in str(p.relative_to(root)).lower()}
    rows = []
    for _, r in meta.iterrows():
        sid = str(r["id"]).strip()
        img = files.get(sid)
        mk = files.get(sid.replace("bus_", "mask_")) or files.get("mask_" + sid)
        if img is None or mk is None:
            continue
        num = re.findall(r"\d+", sid)
        case = str(r["case"]) if "case" in r and str(r["case"]) != "nan" else (num[0] if num else sid)
        rows.append(row(uid=f"BUS-BRA/{sid}", dataset="BUS-BRA", modality="US", image_path=str(img),
                        mask_path=str(mk), label=int(str(r["pathology"]).lower().startswith("malig")),
                        patient_id=f"BUS-BRA/{case}", birads=str(r.get("birads", "")),
                        device=str(r.get("device", ""))))
    return rows


# ============================================================================ BUSB (Yap et al. Dataset B)
def parse_busb(root: Path) -> list[dict]:
    imgs = list_images(root)
    is_mask = lambda d: any(k in Path(d).name.lower() for k in ["gt", "mask", "label", "segment"])
    rows = []
    # Layout A: benign/ malignant sub-folders
    by_cls = [p for p in imgs if p.parent.name.lower() in {"benign", "malignant"} and "_mask" not in p.stem.lower()]
    if by_cls:
        for p in by_cls:
            masks = sorted(p.parent.glob(p.stem + "_mask*")) or sorted(root.rglob(f"*mask*/{p.name}"))
            if not masks:
                continue
            cls = p.parent.name.lower()
            rows.append(row(uid=f"BUSB/{cls}/{p.stem}", dataset="BUSB", modality="US", image_path=str(p),
                            mask_path=str(masks[0]), label=int(cls == "malignant"), patient_id=f"BUSB/{p.stem}"))
        return rows
    # Layout B: original/ + GT/ folders + DatasetB.xlsx (columns like Image, Type)
    meta = None
    for t in list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")) + list(root.rglob("*.csv")):
        df = read_table(t)
        lc = {c.lower().strip(): c for c in df.columns}
        typ = next((lc[c] for c in lc if c in {"type", "class", "label", "pathology"}), None)
        name = next((lc[c] for c in lc if c in {"image", "image name", "name", "filename", "file"}), None)
        if typ and name:
            meta = {Path(str(n)).stem.lstrip("0") or "0": str(t_).lower() for n, t_ in zip(df[name], df[typ])}
            break
    if meta is None:
        print("  BUSB: neither benign/malignant folders nor a table with Image+Type columns found")
        return []
    masks = {p.stem.lstrip("0"): p for p in imgs if is_mask(p.parent)}
    for p in imgs:
        if is_mask(p.parent):
            continue
        key = p.stem.lstrip("0") or "0"
        if key not in meta or key not in masks:
            continue
        t = meta[key]
        if not (t.startswith("benign") or t.startswith("malig")):
            continue
        rows.append(row(uid=f"BUSB/{p.stem}", dataset="BUSB", modality="US", image_path=str(p),
                        mask_path=str(masks[key]), label=int(t.startswith("malig")), patient_id=f"BUSB/{p.stem}"))
    return rows


# ============================================================================ CBIS-DDSM
UID_RE = re.compile(r"^\d+(\.\d+){5,}$")


def _cbis_concepts(shape: str, margin: str):
    s, m = str(shape).upper(), str(margin).upper()
    shp = NAN
    if "IRREGULAR" in s:
        shp = 1
    elif any(k in s for k in ["OVAL", "ROUND", "LOBULATED"]):  # BI-RADS 5th ed.: lobulated -> oval
        shp = 0
    mar = NAN
    if any(k in m for k in ["SPICULATED", "ILL_DEFINED", "MICROLOBULATED"]):
        mar = 1
    elif "CIRCUMSCRIBED" in m:
        mar = 0
    spic = 1 if "SPICULATED" in m else (0 if mar == mar else NAN)  # mar == mar is False only for NaN
    return shp, mar, spic


def parse_cbis(root: Path) -> list[dict]:
    descs = [p for p in root.rglob("*.csv") if "mass_case_description" in p.name.lower().replace("-", "_")]
    if not descs:
        print("  CBIS-DDSM: no mass_case_description_*.csv found")
        return []
    folder_files: dict[str, list[Path]] = defaultdict(list)
    for p in list_images(root):
        for anc in [p.parent, p.parent.parent, p.parent.parent.parent]:
            folder_files[anc.name].append(p)
    # Kaggle JPEG version: dicom_info.csv maps SeriesInstanceUID -> jpeg path + description
    kaggle = {}
    di = [p for p in root.rglob("dicom_info.csv")]
    if di:
        d = pd.read_csv(di[0])
        for _, r in d.iterrows():
            ip = str(r.get("image_path", ""))
            jp = root / "jpeg" / Path(ip).parent.name / Path(ip).name if ip else None
            if jp is None or not jp.exists():
                cands = list(root.rglob(Path(ip).name)) if ip else []
                jp = next((c for c in cands if Path(ip).parent.name in str(c)), None)
            if jp is not None:
                kaggle.setdefault(str(r.get("SeriesInstanceUID")), []).append((str(r.get("SeriesDescription", "")).lower(), jp))

    def resolve(csv_path: str, want: str) -> Path | None:
        parts = [x.strip() for x in str(csv_path).replace("\\", "/").split("/") if x.strip()]
        uids = [x for x in parts if UID_RE.match(x)]
        for u in reversed(uids):  # Kaggle
            if u in kaggle:
                hits = [jp for desc, jp in kaggle[u] if want in desc] or [jp for _, jp in kaggle[u]]
                if hits:
                    return pick(hits, want)
        for key in list(reversed(uids)) + parts[:1]:  # TCIA DICOM tree
            if key in folder_files:
                return pick(folder_files[key], want)
        return None

    def pick(files: list[Path], want: str) -> Path | None:
        files = sorted(set(files))
        tag = {"full": "full mammogram", "roi mask": "roi mask", "cropped": "cropped"}[want]
        tagged = [f for f in files if tag in str(f).lower()]  # TCIA folder names carry the series description
        if tagged:
            files = tagged
        if len(files) == 1 or want == "full":
            return max(files, key=lambda f: f.stat().st_size) if want == "full" else files[0]
        bins = [f for f in files if is_binary_image(f)]
        if want == "roi mask":
            return bins[0] if bins else None
        return next((f for f in files if f not in bins), None)

    rows, miss = [], 0
    for csv in descs:
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        split = "test" if "test" in csv.name.lower() else "train"
        for _, r in df.iterrows():
            img = resolve(r["image file path"], "full")
            mk = resolve(r["roi mask file path"], "roi mask")
            if img is None or mk is None:
                miss += 1
                continue
            shp, mar, spic = _cbis_concepts(r.get("mass shape", ""), r.get("mass margins", ""))
            uid = f"CBIS-DDSM/{Path(str(r['image file path'])).parts[0]}_{r['abnormality id']}"
            rows.append(row(uid=uid, dataset="CBIS-DDSM", modality="MG", image_path=str(img), mask_path=str(mk),
                            label=int(str(r["pathology"]).upper() == "MALIGNANT"),
                            patient_id=f"CBIS-DDSM/{r['patient_id']}", birads=str(r.get("assessment", "")),
                            official_split=split, c_shape_irregular=shp, c_margin_noncirc=mar,
                            c_margin_spiculated=spic))
    if miss:
        print(f"  CBIS-DDSM: {miss} rows could not be resolved to image+mask files")
    return rows


# ============================================================================ MIAS
MIAS_RE = re.compile(r"^(mdb\d{3})\s+([FGD])\s+(CALC|CIRC|SPIC|MISC|ARCH|ASYM|NORM)"
                     r"(?:\s+([BM]))?(?:\s+(\d+))?(?:\s+(\d+))?(?:\s+(\d+))?", re.I)


def parse_mias(root: Path) -> list[dict]:
    files = {p.stem.lower(): p for p in list_images(root)}
    info_lines = []
    for t in root.rglob("*"):
        if t.is_file() and t.suffix.lower() in {".txt", ".info", ""} and t.stat().st_size < 200_000:
            try:
                lines = open(t, errors="ignore").read().splitlines()
            except Exception:
                continue
            if sum(bool(MIAS_RE.match(l.strip())) for l in lines) > 20:
                info_lines = lines
                break
    if not info_lines:
        print("  MIAS: Info.txt (mdbXXX lines) not found")
        return []
    rows, seen = [], defaultdict(int)
    for l in info_lines:
        m = MIAS_RE.match(l.strip())
        if not m:
            continue
        ref, _, cls, sev, x, y, r = m.groups()
        cls = cls.upper()
        if cls == "NORM" or sev is None or x is None or r is None or ref.lower() not in files:
            continue
        seen[ref] += 1
        n = int(ref[3:])
        mar = {"CIRC": 0, "SPIC": 1, "MISC": 1}.get(cls, NAN)
        spic = {"CIRC": 0, "SPIC": 1, "MISC": 0}.get(cls, NAN)
        rows.append(row(uid=f"MIAS/{ref}_{seen[ref]}", dataset="MIAS", modality="MG",
                        lesion_type="mass" if cls in {"CIRC", "SPIC", "MISC"} else cls.lower(),
                        image_path=str(files[ref.lower()]), circle=f"{x},{y},{r}", label=int(sev.upper() == "M"),
                        patient_id=f"MIAS/{(n + 1) // 2}", c_margin_noncirc=mar, c_margin_spiculated=spic))
    return rows


# ============================================================================ INbreast
def parse_inbreast(root: Path) -> list[dict]:
    meta = None
    for t in list(root.rglob("*.csv")) + list(root.rglob("*.xls")) + list(root.rglob("*.xlsx")):
        try:
            df = read_table(t)
        except Exception:
            continue
        lc = {c.lower().replace("-", "").replace(" ", ""): c for c in df.columns}
        if "birads" in lc and "filename" in lc:
            meta = {str(int(f)) if str(f).replace(".0", "").isdigit() else str(f): str(b).strip().lower()
                    for f, b in zip(df[lc["filename"]], df[lc["birads"]]) if str(f) != "nan"}
            break
    if meta is None:
        print("  INbreast: INbreast.csv/.xls with 'File Name' and 'Bi-Rads' columns not found")
        return []
    imgs = {p.name.split("_")[0]: p for p in list_images(root)}
    xmls = {p.stem: p for p in root.rglob("*.xml") if "pectoral" not in str(p).lower()}  # lesion XMLs only
    out_dir = WORK / "inbreast_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid, br in meta.items():
        if fid not in imgs or fid not in xmls:
            continue
        digit = re.match(r"(\d)", br)
        if not digit:
            continue
        label = int(int(digit.group(1)) >= 4)  # common convention: BI-RADS >= 4 malignant
        try:
            pl = plistlib.load(open(xmls[fid], "rb"))
        except Exception:
            continue
        rois = pl.get("Images", [{}])[0].get("ROIs", [])
        h, w = _img_hw(imgs[fid])
        k = 0
        for roi in rois:
            if str(roi.get("Name", "")).lower() != "mass":
                continue
            pts = [tuple(float(v) for v in re.findall(r"[-\d.]+", s)[:2]) for s in roi.get("Point_px", [])]
            if len(pts) < 3:
                continue
            k += 1
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
            mp = out_dir / f"{fid}_{k}.png"
            cv2.imwrite(str(mp), m)
            pid = imgs[fid].name.split("_")[1] if "_" in imgs[fid].name else fid
            rows.append(row(uid=f"INbreast/{fid}_{k}", dataset="INbreast", modality="MG", image_path=str(imgs[fid]),
                            mask_path=str(mp), label=label, patient_id=f"INbreast/{pid}", birads=br))
    return rows


def _img_hw(p: Path):
    if p.suffix.lower() == ".dcm":
        import pydicom
        d = pydicom.dcmread(str(p), stop_before_pixels=True)
        return int(d.Rows), int(d.Columns)
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return a.shape[:2]


# ============================================================================ BrEaST-Lesions_USG (TCIA)
def _binary_mask_union(paths: list[Path], out: Path) -> Path:
    m = None
    for q in paths:
        a = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
        if a is None:
            continue
        a = (a.max(axis=2) if a.ndim == 3 else a) > 0
        m = a if m is None else (m | a)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), (m.astype(np.uint8) * 255) if m is not None else np.zeros((8, 8), np.uint8))
    return out


def _find_col(df: pd.DataFrame, *keys, exclude=()):
    for c in df.columns:
        lc = str(c).lower().replace(" ", "_")
        if all(k in lc for k in keys) and not any(x in lc for x in exclude):
            return c
    return None


def parse_breast(root: Path) -> list[dict]:
    meta = None
    for t in list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")) + list(root.rglob("*.csv")):
        try:
            d = read_table(t)
        except Exception:
            continue
        if _find_col(d, "classification") is not None:
            meta = d
            break
    if meta is None:
        print("  BrEaST: clinical-data table with a 'Classification' column not found")
        return []
    files = {p.name.lower(): p for p in list_images(root)}
    c_cls, c_img = _find_col(meta, "classification"), _find_col(meta, "image", "file")
    c_mask, c_case = _find_col(meta, "mask", "tumor"), _find_col(meta, "case")
    c_shape, c_margin = _find_col(meta, "shape"), _find_col(meta, "margin")
    c_birads = _find_col(meta, "birads") or _find_col(meta, "bi-rads")
    rows = []
    for _, r in meta.iterrows():
        cls = str(r[c_cls]).strip().lower()
        if cls not in {"benign", "malignant"}:
            continue
        case = str(r[c_case]).strip() if c_case else str(r.name)
        img_name = str(r[c_img]).strip() if c_img else f"{case}.png"
        img = files.get(img_name.lower())
        if img is None:
            continue
        mnames = re.split(r"[;,]\s*", str(r[c_mask])) if c_mask and str(r[c_mask]) != "nan" else []
        mpaths = [files[n.lower()] for n in mnames if n.lower() in files]
        if not mpaths:
            mpaths = [p for n, p in files.items() if n.startswith(Path(img_name).stem.lower() + "_tumor")]
        mask = _binary_mask_union(mpaths, WORK / "breast_masks" / f"{Path(img_name).stem}_mask.png") if mpaths else ""
        shp = str(r[c_shape]).lower() if c_shape else ""
        mar = str(r[c_margin]).lower() if c_margin else ""
        s_irr = 1 if "irregular" in shp else (0 if ("oval" in shp or "round" in shp) else NAN)
        if "not" in mar or any(k in mar for k in ["angular", "indistinct", "microlobulated", "spiculated"]):
            m_nc = 1
        elif "circumscribed" in mar:
            m_nc = 0
        else:
            m_nc = NAN
        m_sp = 1 if "spiculated" in mar else (0 if m_nc == m_nc else NAN)
        rows.append(row(uid=f"BrEaST/{case}", dataset="BrEaST", modality="US", image_path=str(img), mask_path=str(mask),
                        label=int(cls == "malignant"), patient_id=f"BrEaST/{case}",
                        birads=str(r[c_birads]) if c_birads else "", c_shape_irregular=s_irr,
                        c_margin_noncirc=m_nc, c_margin_spiculated=m_sp))
    return rows


# ============================================================================ BUS-UCLM (Mendeley Data)
def parse_busuclm(root: Path) -> list[dict]:
    imgs = list_images(root)
    is_mask = lambda p: any(k in p.parent.name.lower() for k in ["mask", "gt", "label"])
    masks = {p.stem.lower(): p for p in imgs if is_mask(p)}
    # use only the original images/ folder; derived copies (e.g. bus_uclm_separated/<class>/) would duplicate lesions
    if any(p.parent.name.lower() == "images" for p in imgs):
        imgs = [p for p in imgs if p.parent.name.lower() in ("images", "masks")]
    labels = {}
    for t in list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")):
        try:
            d = read_table(t)
        except Exception:
            continue
        for c in d.columns:
            vals = d[c].astype(str).str.lower()
            if vals.isin(["benign", "malignant", "normal"]).mean() > 0.5:
                name_col = next((k for k in d.columns if k != c and d[k].astype(str).str.contains(r"\.(png|jpg)", case=False).mean() > 0.5), d.columns[0])
                labels = {Path(str(n)).stem.lower(): v for n, v in zip(d[name_col], vals)}
                break
        if labels:
            break
    rows = []
    for p in imgs:
        if is_mask(p):
            continue
        mp = masks.get(p.stem.lower())
        if mp is None:
            continue
        rgb = cv2.imread(str(mp), cv2.IMREAD_COLOR)
        if rgb is None or rgb.max() == 0:
            continue  # normal image, no lesion
        lab = labels.get(p.stem.lower())
        if lab is None:  # colour-coded mask: red = malignant, green = benign
            b, g, r_ = (rgb[..., i].astype(int).sum() for i in range(3))
            lab = "malignant" if r_ > g else "benign"
        if lab not in {"benign", "malignant"}:
            continue
        out = _binary_mask_union([mp], WORK / "busuclm_masks" / f"{p.stem}_mask.png")
        pid = p.stem.split("_")[0]
        rows.append(row(uid=f"BUS-UCLM/{p.stem}", dataset="BUS-UCLM", modality="US", image_path=str(p),
                        mask_path=str(out), label=int(lab == "malignant"), patient_id=f"BUS-UCLM/{pid}"))
    if not rows:
        print("  BUS-UCLM: no image/mask pairs found (expected images/ and masks/ folders)")
    return rows


PARSERS = {"CBIS-DDSM": parse_cbis, "MIAS": parse_mias, "INbreast": parse_inbreast,
           "BUSI": parse_busi, "BUSB": parse_busb, "BUS-BRA": parse_busbra,
           "BrEaST": parse_breast, "BUS-UCLM": parse_busuclm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    WORK = cfg["work_root"]
    out = manifest_path(cfg)
    old = pd.read_csv(out) if out.exists() and a.only else None
    all_rows = []
    for name, fn in PARSERS.items():
        if a.only and name not in a.only:
            continue
        root = ds_dir(cfg, name)
        print(f"[{name}] {root}")
        if not root.exists():
            print("  folder missing, skipped")
            continue
        rs = fn(root)
        print(f"  lesions: {len(rs)}  malignant: {sum(r['label'] == 1 for r in rs)}")
        all_rows += rs
    df = pd.DataFrame(all_rows)
    if old is not None:
        df = pd.concat([old[~old.dataset.isin(a.only)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} lesions)")
    if len(df):
        print(df.groupby(["dataset", "lesion_type"]).label.agg(["count", "sum"]).rename(columns={"sum": "malignant"}))
        cc = ["c_shape_irregular", "c_margin_noncirc", "c_margin_spiculated"]
        print("\nexpert concept labels available:\n", df.groupby("dataset")[cc].count())
