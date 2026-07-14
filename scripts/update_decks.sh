#!/bin/bash
# Regenerate BOTH COMPASS presentations from the latest result JSONs/figures.
# Run via SLURM (deepnet2), never the login node:
#   sbatch ... scripts/slurm_run.sh bash scripts/update_decks.sh
set -e
python scripts/make_pptx.py --out /data1/yansari/TrajectoryDiff/COMPASS_results.pptx
python scripts/make_deck.py --out /data1/yansari/TrajectoryDiff/COMPASS_results.pdf
echo "[decks] both regenerated $(date)"
