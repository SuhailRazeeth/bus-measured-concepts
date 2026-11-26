#!/bin/bash
# Run GPU jobs on the shared server: wait for a GPU with enough free memory, run, and if the job crashes
# (e.g. CUDA out of memory because another user's job grew) wait and retry. Finished runs are skipped by 08,
# so a retry only redoes the run that crashed.
# Usage:  nohup bash run_retry.sh "python -u 08_train_cbm.py ..." "python -u 08_train_cbm.py ..." > logs/x.log 2>&1 &
# Optional: NEED_MB=8000 (free memory required, default 11000)
NEED_MB=${NEED_MB:-11000}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cmd in "$@"; do
  for try in $(seq 1 50); do
    while true; do
      G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
          awk -F', ' -v n="$NEED_MB" '$3-$2>=n {print $1; exit}')
      [ -n "$G" ] && break
      sleep 120
    done
    echo "[$(date '+%F %T')] try $try on GPU $G: $cmd"
    CUDA_VISIBLE_DEVICES=$G $cmd && break
    rc=$?
    echo "[$(date '+%F %T')] crashed (exit $rc), retrying in 5 min"
    sleep 300
  done
done
echo "[$(date '+%F %T')] ALL DONE"
