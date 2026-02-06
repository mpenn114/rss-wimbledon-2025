#!/usr/bin/env bash

set -euo pipefail

# Configuration
SCRIPT_NAME="alternative_models.py"

MODELS=(
  "bradley_terry_model"
  "elo_model"
  "random_forest_model"
)

TOURNAMENTS=(
  "Australian Open"
  "French Open"
  "Wimbledon"
  "US Open"
)

YEARS=(
  2024
  2025
)

for model in "${MODELS[@]}"; do
  for tournament in "${TOURNAMENTS[@]}"; do
    for year in "${YEARS[@]}"; do

      # Exclusion logic: Skip US Open for the year 2025
      if [[ "$tournament" == "US Open" && "$year" -eq 2025 ]]; then
        echo "Skipping $tournament for $year as requested."
        continue
      fi

      # 1. Run Male Data (Default)
      echo "Executing: $model | $tournament | $year | Male"
      uv run "$SCRIPT_NAME" \
        --model "$model" \
        --tournament "$tournament" \
        --year "$year"

      # 2. Run Female Data (Using the --female flag)
      echo "Executing: $model | $tournament | $year | Female"
      uv run "$SCRIPT_NAME" \
        --model "$model" \
        --tournament "$tournament" \
        --year "$year" \
        --female

    done
  done
done

echo "Batch processing complete!"
