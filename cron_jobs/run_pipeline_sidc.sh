#!/bin/bash
cd /export/solardl/solar_image_processing || exit 1
UV=/home/ravito/.local/bin/uv
LOG="logs/cron_sidc_$(date +%F).log"
{
  echo "=== SIDC pipeline start $(date) ==="
  $UV run python scripts/download_solar_images.py --sidc && \
  $UV run python scripts/preprocess_solar_images.py --source sidc && \
  $UV run python scripts/crop_solar_images.py --source sidc
  echo "=== SIDC pipeline end   $(date) (exit $?) ==="
} >> "$LOG" 2>&1
