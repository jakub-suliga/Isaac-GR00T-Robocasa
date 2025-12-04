#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -J train_flower
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Set default SLURM_ARRAY_TASK_ID if not set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

HOME_DIR=$(pwd)
BASE_DIR=/hkfs/work/workspace/scratch/uhtfz-groot-robo
CONDA_PATH=/home/hk-project-p0024638/uhtfz/miniconda3

CKPT_PATH="/hkfs/work/workspace/scratch/uhtfz-groot-robo/outputs/gr00t_finetune/checkpoint-60000"
CKPT_NAME="gr00t_finetune"
CKPT_STEP="checkpoint-60000"

PYTHON_BIN="$CONDA_PATH/envs/gr00t/bin/python"

$PYTHON_BIN $BASE_DIR/Isaac-GR00T/scripts/inference_service.py --server \
    --port 897$SLURM_ARRAY_TASK_ID \
    --model_path $CKPT_PATH \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment &
SERVE_PID=$!



TASK_NAMES=(
#   "TurnSinkSpout"
#   "TurnOnStove"
#   "TurnOnSinkFaucet"
#   "TurnOnMicrowave"
#   "TurnOffStove"
#   "TurnOffSinkFaucet"
#   "TurnOffMicrowave"
#   "PnPStoveToCounter" #7
#   "PnPSinkToCounter"
#   "PnPMicrowaveToCounter"
#   "PnPCounterToStove"
#   "PnPCounterToSink"
#   "PnPCounterToMicrowave"
#   "PnPCounterToCab"
#   "PnPCabToCounter"
#   "OpenSingleDoor"
#   "OpenDrawer"
#   "OpenDoubleDoor"
#   "CoffeeSetupMug"
#   "CoffeeServeMug"
#   "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
 "CloseDoubleDoor"
)

MAIN_PIDS=()

# Select tasks at specific indices based on array task ID
# SELECTED_TASKS=()
# if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
#     SELECTED_TASKS+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
# fi
# if [ $((SLURM_ARRAY_TASK_ID + 8)) -lt ${#TASK_NAMES[@]} ]; then
#     SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
# fi
# if [ $((SLURM_ARRAY_TASK_ID + 16)) -lt ${#TASK_NAMES[@]} ]; then
#     SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")
# fi
# TASK_NAMES=("${SELECTED_TASKS[@]}")

echo "[i] Running tasks: ${TASK_NAMES[@]}"
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/testoutput/groot/robocasa/$CKPT_NAME/$CKPT_STEP/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    $PYTHON_BIN $BASE_DIR/Isaac-GR00T/scripts/robocasa_service.py --client \
        --port 897$SLURM_ARRAY_TASK_ID \
        --env_name $TASK_NAME \
        --video_dir $OUTPUT_DIR \
        --max_episode_steps 750 \
        --n_episodes 2 \
        --generative_textures \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done


# Wait on just the main.py processes
for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

# Kill serve_policy once those tasks finish
kill "$SERVE_PID"
echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."
