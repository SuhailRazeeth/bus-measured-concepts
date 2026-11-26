#!/bin/bash
# Third review round: four new controls (GPU), then all CPU analyses.
# Usage (from the pilot folder):  mkdir -p logs && nohup bash run_tier3.sh > logs/tier3.log 2>&1 &
# Only the four datasets with masks and patient identifiers are held out (the analyses use these four).
# 4 configurations x 2 encoders x 4 held-out datasets x 3 seeds = 96 training runs. Finished runs are skipped,
# so the script can be restarted after a crash.
C="python -u 08_train_cbm.py --geom medsam --no_expert --intensity --unfreeze 12 --heldout BUS-BRA BUS-UCLM BUSB BrEaST"
bash run_retry.sh \
  "$C --backbone biomedclip --modes bbc" \
  "$C --backbone dinov2_b --modes bbc" \
  "$C --backbone biomedclip --modes hyb --res_pen 0.1 --suffix _rp01_rb --box_jitter 0.1 0.2 0.3" \
  "$C --backbone dinov2_b --modes hyb --res_pen 0.1 --suffix _rp01_rb --box_jitter 0.1 0.2 0.3" \
  "$C --backbone biomedclip --modes cbm" \
  "$C --backbone dinov2_b --modes cbm" \
  "$C --backbone biomedclip --modes seq --int_p 1.0" \
  "$C --backbone dinov2_b --modes seq --int_p 1.0"
echo "[$(date '+%F %T')] training done, analysing"

# BrEaST descriptor validation: shuffled concept heads as a null, and the independent CBM heads
for bk in dinov2_b biomedclip; do
  python -u 14_birads_validation.py --run ${bk}_medsam_int50_noexp_ic_u12 --mode shuf
  python -u 14_birads_validation.py --run ${bk}_medsam_int100_noexp_ic_u12 --mode seq
done
python -u 17_revision2.py
python -u 18_local_recalibration.py
python -u 19_revision3.py

# everything needed for the paper in one file
W=$(python -c "from common import load_cfg; print(load_cfg()['work_root'])")
{
  echo "===== 19 THIRD-ROUND ANALYSES"; cat "$W/stats/revision3/report.txt"
  for d in "$W"/birads_validation*; do
    echo "===== TABLE IV WITH CIs: $(basename "$d")"; cat "$d/expected_pairs.csv"
  done
} > "$W/stats/tier3_for_paper.txt"
echo "[$(date '+%F %T')] TIER3 ALL DONE. Send me: $W/stats/tier3_for_paper.txt"
