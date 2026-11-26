# Data

No images are redistributed. Download the datasets from their original sources and place each in its own
sub-folder of `data_root` (names as in `code/config.example.yaml`).

| Dataset | Lesions used | Source | License (check before use) |
|---|---|---|---|
| BUS-BRA | 1,875 | Gomez-Flores et al., Zenodo | see source |
| BUSI | 647 | Al-Dhabyani et al. | see source |
| BUSB (Dataset B) | 163 | Yap et al., available on request | see source |
| BrEaST | 252 | Pawłowska et al., The Cancer Imaging Archive | see source |
| BUS-UCLM | 264 | Vallez et al., Mendeley Data | see source |

## Derived data in this folder

Created on the training server with `python code/export_release_data.py`:

| File | Content |
|---|---|
| `lesions.csv` | lesion identifier, dataset, malignancy label, patient identifier, scanner |
| `splits.csv` | role (train, val, test) of every lesion for each held-out dataset and seed |
| `concepts_medsam.csv` | eleven raw concept values on the MedSAM masks and the MedSAM Dice against the reference mask |
| `concepts_radiologist.csv` | the same concepts on the reference masks |
| `medsam_masks/<dataset>/` | MedSAM masks in the 384 by 384 ROI frame (PNG) |

Lesion identifiers have the form `<dataset>/<case>` and map to the original file names through
`code/01_build_manifest.py`. Masks are omitted for any dataset whose terms do not allow sharing derived data
(`--skip_mask_datasets`). In this release, BUSB masks are not included because BUSB is available only on
request. Its concept values are included. BUSI has no reference masks, so it has no concepts or masks.

Contents: 3,201 lesions (BUS-BRA 1,875, BUSI 647, BUS-UCLM 264, BrEaST 252, BUSB 163), 2,554 lesions with
measured concepts, 2,391 MedSAM masks, and 15 split assignments (5 held-out datasets by 3 seeds).
