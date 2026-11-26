#!/bin/bash
# Tier 1b: hybrid CBM (faithful concept part + penalized image residual), two residual penalties, both backbones.
# Usage (from the pilot folder):  nohup bash run_tier1b.sh > logs/tier1b.log 2>&1 &
C="python -u 08_train_cbm.py --geom medsam --no_expert --intensity --unfreeze 12 --modes hyb"
bash run_retry.sh \
  "$C --backbone dinov2_b --res_pen 0.1 --suffix _rp01" \
  "$C --backbone dinov2_b --res_pen 1.0 --suffix _rp1" \
  "$C --backbone biomedclip --res_pen 0.1 --suffix _rp01" \
  "$C --backbone biomedclip --res_pen 1.0 --suffix _rp1"
echo "[$(date '+%F %T')] training done, analysing"
for r in dinov2_b_medsam_noexp_ic_u12_rp01 dinov2_b_medsam_noexp_ic_u12_rp1 \
         biomedclip_medsam_noexp_ic_u12_rp01 biomedclip_medsam_noexp_ic_u12_rp1; do
  python -u 09_analyze_cbm.py --run "$r"
done
echo "[$(date '+%F %T')] TIER1B ALL DONE"
