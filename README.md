# Measured BI-RADS Concepts and Concept Leakage in Interpretable Breast Ultrasound Diagnosis

Code and derived data for the paper

> M.S. Suhail Razeeth, F. Fathima Nuha, M.M. Mufassirin. *Measured BI-RADS Concepts and Concept Leakage in Interpretable Breast Ultrasound Diagnosis Across Unseen Hospitals.* Manuscript under review, 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## What this repository contains

Eleven BI-RADS aligned concepts (six of lesion geometry, five of echo pattern) are measured automatically on
box prompted MedSAM segmentations. They supervise fully fine-tuned DINOv2 and BiomedCLIP vision transformers
without expert concept labels. Several concept bottleneck variants and controls are compared under leave one
dataset out validation on five public breast ultrasound datasets (3,201 lesions):

| Model | Run folder (under `work_root/cbm_runs/`) | Mode |
|---|---|---|
| Black box | `<enc>_medsam_u12` | `bb` |
| Measured concept black box | `<enc>_medsam_noexp_ic_u12` | `bbc` |
| Multitask black box (and shuffled) | `<enc>_medsam_noexp_ic_u12` (`_shufT`) | `mt` |
| Joint CBM, replacement rate p | `<enc>_medsam_int{25,50,75,100}_noexp_ic_u12`, p = 0 in `<enc>_medsam_noexp_ic_u12` | `cbm` |
| Shuffled concept CBM | `<enc>_medsam_int50_noexp_ic_u12` | `shuf` |
| Sequential CBM | `<enc>_medsam_noexp_ic_u12` | `seq` |
| Independent CBM | `<enc>_medsam_int100_noexp_ic_u12` | `seq` |
| Hybrid CBM, β = 0.1 / 1.0 (and shuffled) | `<enc>_medsam_noexp_ic_u12_rp01`, `_rp1` (`_rp01_shufT`) | `hyb` |

`<enc>` is `dinov2_b` or `biomedclip`. Every configuration is trained with three seeds for each held-out dataset.

## Layout

```
code/      all scripts (run from inside this folder)
results/   summary statistics and figure data reported in the paper
data/      derived data (lesion lists, splits, measured concepts, MedSAM masks), see data/README.md
```

## Installation

```bash
conda create -n busconcepts python=3.10 -y
conda activate busconcepts
pip install -r code/requirements.txt
```

Experiments were run with PyTorch on NVIDIA RTX 6000 Ada GPUs with bfloat16 mixed precision.

## Reproducing the paper

1. Download the five datasets (links in `data/README.md`) into one folder.
2. `cp code/config.example.yaml code/config.yaml` and set `data_root`, `work_root`, and `hf_home`.
3. Run the whole pipeline, or the blocks of `code/run_all.sh` one at a time:

```bash
cd code
mkdir -p logs && nohup bash run_all.sh > logs/all.log 2>&1 &
```

Finished runs are skipped, so the pipeline can be restarted. The main steps are:

| Step | Script | Paper section | GPU |
|---|---|---|---|
| Manifest, ROIs | `01_build_manifest.py`, `02_make_rois.py` | III-A, III-B | no |
| MedSAM masks and geometry concepts | `07_sam_masks.py` | III-C, III-D | yes |
| Echo concepts | `12_intensity_concepts.py` | III-E | no |
| Training (all model families) | `08_train_cbm.py` | III-G to III-I | yes |
| Per run analysis | `09_analyze_cbm.py` | IV | no |
| Interpretable baselines | `04_extract_features.py`, `10_baselines.py` | III-K | yes / no |
| BI-RADS descriptor validation on BrEaST | `14_birads_validation.py` | IV-A | no |
| Main statistics, non-inferiority, contrasts | `15_stats.py`, `17_revision2.py` | IV-B to IV-E | no |
| Local recalibration | `18_local_recalibration.py` | IV-G | no |
| Hartung and Knapp estimator, leakage CIs, fair correction, Holm, secondary outcomes | `19_revision3.py` | IV | no |
| Leakage with refitted predictors, malignancy adjusted descriptor AUCs | `20_round5.py` | IV-A, IV-C | no |
| Figures | `13_figures.py`, `16_revision_figures.py` | figures | no |

`05_pilot_experiments.py` and `06_pilot2.py` belong to an earlier feasibility pilot and are kept because later
scripts import helper functions from them.

## Results

`results/stats/` holds the tables behind the paper (`main_table.csv`, `noninferiority.csv`, `contrasts.csv`,
`scanners.csv`, `fewshot_gain.csv`, `selection.csv`) and the full reports of the revision analyses
(`third_round_analyses.txt`, `round5_analyses.txt`). `results/figures/` holds the figures and their data.

## Notes

* Lesion boxes are taken from the reference masks of each dataset (jittered by up to 10% for the MedSAM prompts),
  so lesion localisation is expert supplied, as it would be from a box drawn by the reader.
* Concept targets are standardized within each dataset without labels. A new site therefore needs a set of
  unlabeled lesions.
* BUSI has no patient identifiers and contains near duplicate images, so its results are reported separately.
* GPU training is not bit exact. Rerunning a configuration can move a held-out AUC by about 0.01.

## Citation

Please cite the paper and this software (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses. Derived masks and concept values are released
only for datasets whose terms allow it (see `data/README.md`).
