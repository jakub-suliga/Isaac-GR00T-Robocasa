#!/bin/bash
#SBATCH -p accelerated-h200-8
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -J train_flower
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


BASE_DIR="/hkfs/work/workspace/scratch/uhtfz-groot-robo"
CKPT_DIR="$BASE_DIR/outputs/gr00t_finetune/"
DATA_DIR="$BASE_DIR/data/robocasa_mg_gr00t_100"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1




source ~/.bashrc
conda activate gr00t

python /hkfs/work/workspace/scratch/uhtfz-groot-robo/Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path /hkfs/work/workspace/scratch/uhtfz-groot-robo/data/robocasa_mg_gr00t_100 \
    --output-dir /hkfs/work/workspace/scratch/uhtfz-groot-robo/outputs/gr00t_finetune3/ \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --batch-size 128 \
    --num-gpus 8 \
    --max-steps 300000 \
    --save-steps 10000 
