#!/bin/bash
#SBATCH --job-name=mor_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3080:2
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nfs/home/ibagautdinov/MiniOneRec/logs/eval-%j.out

# Оффлайн-оценка HR@K / NDCG@K constrained beam search (num_beams 50), 2 GPU.
# Запуск: sbatch --export=ALL,EXP_NAME=/nfs/home/ibagautdinov/MiniOneRec/output/... sbatch_eval.sh
set -x
echo "=== node $(hostname), started $(date)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate minionerec
export CUDA_HOME=$CONDA_PREFIX
export TRITON_CACHE_DIR=/tmp/triton_ibagautdinov
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
cd ~/MiniOneRec

: "${EXP_NAME:?Set EXP_NAME to the model dir to evaluate}"
category=Industrial_and_Scientific
exp_name_clean=$(basename "$EXP_NAME")
test_file=$(ls ./data/Amazon/test/${category}*11.csv | head -1)
info_file=$(ls ./data/Amazon/info/${category}*.txt | head -1)
temp_dir="./temp/${category}-${exp_name_clean}-${SLURM_JOB_ID}"
mkdir -p "$temp_dir"

python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1"

for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
        --base_model "$EXP_NAME" \
        --info_file "$info_file" \
        --category ${category} \
        --test_data_path "$temp_dir/${i}.csv" \
        --result_json_data "$temp_dir/${i}.json" \
        --batch_size ${EVAL_BS:-4} \
        --num_beams 50 \
        --max_new_tokens 256 \
        --temperature 1.0 \
        --guidance_scale 1.0 \
        --length_penalty 0.0 &
done
wait

output_dir="./results/${exp_name_clean}"
mkdir -p "$output_dir"
actual_cuda_list=$(ls "$temp_dir"/*.json | sed 's/.*\///g; s/\.json//g' | tr '\n' ',' | sed 's/,$//')
python ./merge.py --input_path "$temp_dir" --output_path "$output_dir/final_result_${category}.json" --cuda_list "$actual_cuda_list"
python ./calc.py --path "$output_dir/final_result_${category}.json" --item_path "$info_file"
echo "=== finished $(date)"
