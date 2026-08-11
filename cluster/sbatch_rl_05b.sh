#!/bin/bash
#SBATCH --job-name=mor_rl_05b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3080:2
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nfs/home/ibagautdinov/MiniOneRec/logs/rl-%j.out

# MiniOneRec RL (GRPO): стартует с SFT-чекпоинта.
# Запуск: sbatch --export=ALL,MODEL_PATH=/nfs/home/ibagautdinov/MiniOneRec/output/sft_qwen05b_... sbatch_rl_05b.sh
# Смок:   sbatch --time=02:00:00 --export=ALL,MODEL_PATH=...,EPOCHS=0.01 sbatch_rl_05b.sh
set -x
echo "=== node $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source ~/anaconda3/etc/profile.d/conda.sh
conda activate minionerec
export CUDA_HOME=$CONDA_PREFIX
export TRITON_CACHE_DIR=/tmp/triton_ibagautdinov
export WANDB_MODE=offline
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
cd ~/MiniOneRec
echo "=== git: $(git rev-parse --short HEAD) @ $(git branch --show-current), dirty: $(git status --porcelain | wc -l) files"

: "${MODEL_PATH:?Set MODEL_PATH to the SFT checkpoint dir}"
category=Industrial_and_Scientific
train_file=$(ls ./data/Amazon/train/${category}*.csv)
eval_file=$(ls ./data/Amazon/valid/${category}*11.csv)
info_file=$(ls ./data/Amazon/info/${category}*.txt)
run_name=rl_qwen05b_${category}_${SLURM_JOB_ID}

accelerate launch \
    --config_file ./config/zero2_opt.yaml \
    --num_processes 2 --main_process_port 29531 \
    rl.py \
    --model_path ${MODEL_PATH} \
    --train_batch_size ${TRAIN_BS:-16} \
    --eval_batch_size ${EVAL_BS:-32} \
    --num_train_epochs ${EPOCHS:-2} \
    --gradient_accumulation_steps ${GRAD_ACC:-8} \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --info_file ${info_file} \
    --category ${category} \
    --sample_train ${SAMPLE_TRAIN:-False} \
    --eval_step 0.0999 \
    --reward_type ranking \
    --num_generations 16 \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --add_gt False \
    --beta 1e-3 \
    --dapo False \
    --output_dir /mnt/tank/scratch/ibagautdinov/minionerec_runs/${run_name} \
    --wandb_run_name ${run_name} \
    --sid_index_path ./data/Amazon/index/${category}.index.json \
    --item_meta_path ./data/Amazon/index/${category}.item.json

cp /nfs/home/ibagautdinov/MiniOneRec/logs/rl-${SLURM_JOB_ID}.out \
   /mnt/tank/scratch/ibagautdinov/minionerec_runs/${run_name}/slurm.log 2>/dev/null || true
echo "=== finished $(date)"
