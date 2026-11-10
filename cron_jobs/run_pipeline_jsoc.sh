#!/bin/bash
cd /export/solardl/solar_image_processing || exit 1
UV=/home/ravito/.local/bin/uv
LOG="logs/cron_jsoc_$(date +%F).log"
{
  echo "=== JSOC pipeline start $(date) ==="
  $UV run python scripts/download_solar_images.py --jsoc && \
  $UV run python scripts/preprocess_solar_images.py --source jsoc && \
  $UV run python scripts/crop_solar_images.py --source jsoc
  echo "=== JSOC pipeline end   $(date) (exit $?) ==="
} >> "$LOG" 2>&1
