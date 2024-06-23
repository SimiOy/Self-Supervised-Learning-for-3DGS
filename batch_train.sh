#!/bin/bash
#SBATCH --job-name="scale_rotation"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
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

MODEL='pointnet_cls'
BATCH_SIZE=32
NUM_WORKERS=16
NUM_ITERATIONS=80000
EPOCH=100
OPTIMIZER='SGD'
LEARNING_RATE=0.00001
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
DECAY_STEPS=5000
DECAY_RATE=0.1

NUM_POINT=2048
BETA=3
SCALE_ROTATION=True
USE_NORMALS=False
USE_COLORS=False
FURTHEST_POINT_SAMPLE=True

# Activate conda, run job, deactivate conda
conda activate CV3dgs

cmd="srun python train_model.py --img_dir $IMG_DIR --point_dir $POINT_DIR --model $MODEL --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS --num_iterations $NUM_ITERATIONS --epoch $EPOCH --optimizer $OPTIMIZER \
  --learning_rate $LEARNING_RATE --momentum $MOMENTUM --weight_decay $WEIGHT_DECAY --decay_steps $DECAY_STEPS \
  --decay_rate $DECAY_RATE --num_point $NUM_POINT --beta $BETA --log_dir $LOG_DIR"
  
# Add boolean flags
if [ "$SCALE_ROTATION" = "True" ]; then
  cmd="$cmd --use_scale_and_rotation"
fi

if [ "$USE_NORMALS" = "True" ]; then
  cmd="$cmd --use_normals"
fi

if [ "$USE_COLORS" = "True" ]; then
  cmd="$cmd --use_colors"
fi

if [ "$FURTHEST_POINT_SAMPLE" = "True" ]; then
  cmd="$cmd --furthest_point_sample"
fi

# Run the command
eval $cmd

conda deactivate
