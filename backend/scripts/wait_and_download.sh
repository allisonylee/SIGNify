#!/bin/bash
# Poll until the Kaggle throttle clears, then run the download at a safe pace.
# NOT SHIPPED -- an operational helper.
cd /Users/allisonlee/Desktop/AI/hophacks2026
PY=.venv/bin/python
for i in $(seq 1 120); do          # up to ~2 hours at 60s intervals
  if $PY - <<'PYEOF' >/dev/null 2>&1
from backend.app import config
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
a = KaggleApi(); a.authenticate()
idx = pd.read_parquet("backend/data/gislr/subset_index.parquet")
a.competition_download_file("asl-signs", idx.iloc[0].path, path="/tmp/kprobe", force=True)
PYEOF
  then
    echo "[$(date +%H:%M:%S)] throttle CLEARED after ${i} probe(s); starting download"
    $PY -m backend.training.download --signs 50 --per-sign 40 --workers 2 --rate 1.5
    exit $?
  fi
  echo "[$(date +%H:%M:%S)] probe $i: still throttled"
  sleep 60
done
echo "gave up after 2 hours"
exit 1
