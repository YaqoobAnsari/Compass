#!/bin/bash
# Submit a COMPASS command to deepnet2 via SLURM. ALL compute must go through here
# (never run training/experiments on the login node).
#
# Usage:  bash scripts/submit.sh <mig_profile> <command...>
#   bash scripts/submit.sh 1g.18gb python scripts/exp_order_matters.py --epochs 60
#   COMPASS_TIME=04:00:00 COMPASS_JOB=engine bash scripts/submit.sh 7g.141gb python scripts/train.py
#
# MIG profiles on deepnet2: 1g.18gb (x28), 2g.35gb (x6), 7g.141gb (x2).
set -euo pipefail

PROFILE="${1:?usage: submit.sh <mig_profile: 1g.18gb|2g.35gb|7g.141gb> <command...>}"
shift
[ "$#" -ge 1 ] || { echo "error: no command given"; exit 1; }

mkdir -p /data1/yansari/Compass/experiments/logs
sbatch --job-name="${COMPASS_JOB:-compass}" \
       --gres="gpu:nvidia_h200_${PROFILE}:1" \
       --time="${COMPASS_TIME:-00:30:00}" \
       /data1/yansari/Compass/scripts/slurm_run.sh "$@"
