#!/bin/bash
# Full pipeline of the paper, in order. Each block can be run on its own; finished training runs are skipped,
# so the script can be restarted after a crash. GPU steps use run_retry.sh, which waits for a free GPU.
# Usage (from this folder, after creating config.yaml):  mkdir -p logs && nohup bash run_all.sh > logs/all.log 2>&1 &
#
# The training commands of block 3 were reconstructed from the run folder names used by the analysis scripts
# (15_stats.py, configs()). Blocks 4 to 7 are the scripts that were run during the revision rounds, unchanged.
set -e

# ---------------------------------------------------------------- 1 data preparation (CPU, MedSAM on GPU)
python -u 01_build_manifest.py                      # lesion manifest, labels, patient identifiers
python -u 02_make_rois.py --workers 8               # square lesion ROIs and radiologist mask geometry
python -u 03_download_models.py                     # caches the encoder weights (needs internet)
python -u 07_sam_masks.py --model wanglab/medsam-vit-base --name medsam \
       --datasets BUS-BRA BUSB BrEaST BUS-UCLM      # box prompted MedSAM masks (box jitter 10%) and geometry concepts
python -u 12_intensity_concepts.py                  # five echo concepts on radiologist and MedSAM masks
python -u 04_extract_features.py --backbones biomedclip dinov2_b   # frozen features for the baselines

# ---------------------------------------------------------------- 2 interpretable baselines (CPU)
python -u 10_baselines.py

# ---------------------------------------------------------------- 3 main models (GPU)
C="python -u 08_train_cbm.py --geom medsam --unfreeze 12"
for bk in dinov2_b biomedclip; do
  bash run_retry.sh \
    "$C --backbone $bk --modes bb" \
    "$C --backbone $bk --modes cbm --no_expert --intensity --int_p 0.25" \
    "$C --backbone $bk --modes cbm --no_expert --intensity --int_p 0.5" \
    "$C --backbone $bk --modes cbm --no_expert --intensity --int_p 0.75" \
    "$C --backbone $bk --modes cbm --no_expert --intensity --int_p 1.0" \
    "$C --backbone $bk --modes cbm --no_expert --int_p 0.5"
done

# ---------------------------------------------------------------- 4 to 7 revision rounds (GPU, then CPU analyses)
bash run_tier1.sh     # sequential CBM, multitask, shuffled CBM, box robustness, BrEaST descriptor validation
bash run_tier1b.sh    # hybrid CBM with residual penalties 0.1 and 1.0
bash run_tier2b.sh    # shuffled multitask and hybrid controls, statistics (15, 17)
python -u 11_summary.py
python -u 13_figures.py
python -u 16_revision_figures.py
python -u 18_local_recalibration.py
bash run_tier3.sh     # joint CBM p = 0, independent CBM, measured concept black box, hybrid box robustness, 19
python -u 20_round5.py   # leakage with refitted predictors, descriptor AUCs adjusted for malignancy
echo "[$(date '+%F %T')] ALL DONE"
