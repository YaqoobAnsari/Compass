#!/bin/bash
#SBATCH --job-name=compass
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/data1/yansari/Compass/experiments/logs/%x_%j.out
#SBATCH --error=/data1/yansari/Compass/experiments/logs/%x_%j.err
#
# Generic COMPASS SLURM job: runs whatever command is passed as arguments,
# on deepnet2 (H200/MIG) with the compass conda env. The --gres (MIG profile)
# and --time are supplied by scripts/submit.sh on the sbatch command line.
#
#   sbatch --gres=gpu:nvidia_h200_1g.18gb:1 scripts/slurm_run.sh python scripts/foo.py --x 1
#
set -euo pipefail

export PATH=/data1/yansari/.conda/envs/compass/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
cd /data1/yansari/Compass

echo "=============================================="
echo "COMPASS SLURM job ${SLURM_JOB_ID:-local}"
echo "node=$(hostname)  date=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "CMD: $*"
echo "=============================================="

exec "$@"
