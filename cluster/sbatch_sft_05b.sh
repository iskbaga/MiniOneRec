#!/bin/bash
#SBATCH --job-name=mor_sft_05b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3080:2
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nfs/home/ibagautdinov/MiniOneRec/logs/sft-%j.out

# MiniOneRec SFT: Qwen2.5-0.5B на midas (2x RTX 3080 Ti 12GB, Ampere = bf16 OK)
# Смок-тест:  sbatch --time=00:40:00 --export=ALL,SAMPLE=2048,MICRO_BS=4 sbatch_sft_05b.sh
# Полный ран: sbatch --export=ALL sbatch_sft_05b.sh
set -xeo pipefail
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

category=Industrial_and_Scientific
train_file=$(ls ./data/Amazon/train/${category}*11.csv)
eval_file=$(ls ./data/Amazon/valid/${category}*11.csv)
run_name=sft_qwen05b_${category}_${SLURM_JOB_ID}

torchrun --nproc_per_node 2 --master_port 29517 sft.py \
    --base_model /nfs/home/ibagautdinov/models/Qwen2.5-0.5B \
    --batch_size 1024 \
    --micro_batch_size ${MICRO_BS:-8} \
    --sample=${SAMPLE:--1} \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --output_dir /mnt/tank/scratch/ibagautdinov/minionerec_runs/${run_name} \
    --wandb_project minionerec \
    --wandb_run_name ${run_name} \
    --category ${category} \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/${category}.index.json \
    --item_meta_path ./data/Amazon/index/${category}.item.json \
    --freeze_LLM False

cp /nfs/home/ibagautdinov/MiniOneRec/logs/sft-${SLURM_JOB_ID}.out \
   /mnt/tank/scratch/ibagautdinov/minionerec_runs/${run_name}/slurm.log 2>/dev/null || true
echo "=== finished $(date)"
