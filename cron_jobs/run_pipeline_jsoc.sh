#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
UV=/home/ravito/.local/bin/uv
LOG="logs/cron_jsoc_$(date +%F).log"
{
  echo "=== JSOC pipeline start $(date) ==="
  $UV run python scripts/download_solar_images.py --source jsoc --config configs/pipeline_config.local.yaml && \
  $UV run python scripts/preprocess_solar_images.py --source jsoc --config configs/pipeline_config.local.yaml && \
  $UV run python scripts/crop_solar_images.py --source jsoc --config configs/pipeline_config.local.yaml
  echo "=== JSOC pipeline end   $(date) (exit $?) ==="
} >> "$LOG" 2>&1
