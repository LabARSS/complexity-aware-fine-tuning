#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

singularity exec --nv \
  --bind $PWD:/src \
  uv.sif \
  bash -c "cd /src && uv run python -m torch.distributed.run \
    --nproc_per_node=4 \
    --standalone \
    ${EXPERIMENT_RELATIVE_PATH}"