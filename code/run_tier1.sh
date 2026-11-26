#!/bin/bash
# Tier 1 revision experiments (reviewer concerns: leakage, multitask control, test-time box robustness).
# Runs one after another on whichever GPU is free (run_retry.sh). Finished runs are skipped, so it can be restarted.
# Usage (from the pilot folder):  nohup bash run_tier1.sh > logs/tier1.log 2>&1 &
C="python -u 08_train_cbm.py --geom medsam --no_expert --intensity --unfreeze 12"
bash run_retry.sh \
  "$C --backbone dinov2_b --modes seq mt --box_jitter 0.1 0.2" \
  "$C --backbone dinov2_b --modes shuf --int_p 0.5" \
  "$C --backbone dinov2_b --modes cbm bb --int_p 0.5 --seeds 0 --suffix _rb --box_jitter 0.1 0.2 0.3" \
  "$C --backbone biomedclip --modes seq mt --box_jitter 0.1 0.2"
echo "[$(date '+%F %T')] training done, analysing"
for r in dinov2_b_medsam_noexp_ic_u12 dinov2_b_medsam_int50_noexp_ic_u12 dinov2_b_medsam_int50_noexp_ic_u12_rb \
         biomedclip_medsam_noexp_ic_u12; do
  python -u 09_analyze_cbm.py --run "$r"
done
python -u 14_birads_validation.py
echo "[$(date '+%F %T')] TIER1 ALL DONE"
