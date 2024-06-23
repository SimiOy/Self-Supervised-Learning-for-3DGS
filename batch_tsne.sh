#!/bin/bash
#SBATCH --job-name="pointnet_uniform_sampling"
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-eemcs-courses-cse3000

# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Set variables
IMG_DIR="/scratch/asimionescu/RP_datasets/ModelNet10_captures"
POINT_DIR="/scratch/asimionescu/gaussian-splatting/output"
LOG_DIR=$SLURM_JOB_NAME

BATCH_SIZE=32
NUM_WORKERS=16

# Activate conda, run job, deactivate conda
conda activate CV3dgs

cmd="srun python embeddings_visualization.py --img_dir $IMG_DIR --point_dir $POINT_DIR --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS --log_dir $LOG_DIR --num_views 64"

# Run the command
eval $cmd

conda deactivate
