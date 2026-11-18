#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
LOG="logs/cron_sidc_$(date +%F).log"
{
  echo "=== SIDC pipeline start $(date) ==="
  $UV run python scripts/download_solar_images.py --source sidc --config configs/pipeline_config.local.yaml && \
  $UV run python scripts/preprocess_solar_images.py --source sidc --config configs/pipeline_config.local.yaml && \
  $UV run python scripts/crop_solar_images.py --source sidc --config configs/pipeline_config.local.yaml
  echo "=== SIDC pipeline end   $(date) (exit $?) ==="
} >> "$LOG" 2>&1
