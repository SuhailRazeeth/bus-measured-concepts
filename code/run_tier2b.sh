#!/bin/bash
# Second-round review: extra shuffled controls (GPU), then all CPU analyses.
# Usage (from the pilot folder):  nohup bash run_tier2b.sh > logs/tier2b.log 2>&1 &
C="python -u 08_train_cbm.py --geom medsam --no_expert --intensity --unfreeze 12"
bash run_retry.sh \
  "$C --backbone biomedclip --modes shuf --int_p 0.5" \
  "$C --backbone dinov2_b --modes hyb --res_pen 0.1 --shuffle --suffix _rp01_shufT" \
  "$C --backbone biomedclip --modes hyb --res_pen 0.1 --shuffle --suffix _rp01_shufT" \
  "$C --backbone dinov2_b --modes mt --shuffle --suffix _shufT" \
  "$C --backbone biomedclip --modes mt --shuffle --suffix _shufT"
echo "[$(date '+%F %T')] training done, analysing"
for r in biomedclip_medsam_int50_noexp_ic_u12 dinov2_b_medsam_noexp_ic_u12_rp01_shufT biomedclip_medsam_noexp_ic_u12_rp01_shufT \
         dinov2_b_medsam_noexp_ic_u12_shufT biomedclip_medsam_noexp_ic_u12_shufT; do
  python -u 09_analyze_cbm.py --run "$r"
done
# BrEaST descriptor validation of the proposed concept heads (sequential and hybrid, both encoders)
python -u 14_birads_validation.py --run dinov2_b_medsam_noexp_ic_u12 --mode seq
python -u 14_birads_validation.py --run biomedclip_medsam_noexp_ic_u12 --mode seq
python -u 14_birads_validation.py --run dinov2_b_medsam_noexp_ic_u12_rp01 --mode hyb
python -u 14_birads_validation.py --run biomedclip_medsam_noexp_ic_u12_rp01 --mode hyb
python -u 15_stats.py
python -u 17_revision2.py
echo "[$(date '+%F %T')] TIER2B ALL DONE"
