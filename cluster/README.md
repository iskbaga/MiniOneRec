# Запуск на кластере ИТМО (ctlab)

Бейзлайн: Qwen2.5-0.5B (base) на Amazon18 Industrial_and_Scientific, нода **midas**
(2× RTX 3080 Ti 12GB — единственная нода с Ampere/bf16; bf16 захардкожен в sft.py/rl.py).

## Пути на кластере

- Репо: `~/MiniOneRec` (этот форк, ветка `itmo-baseline`)
- Окружение: conda `minionerec` (python 3.11, torch 2.6.0+cu124, transformers 4.57.1,
  trl 0.24.0, deepspeed 0.18.0 + cuda-nvcc 12.4 — без nvcc deepspeed падает на импорте)
- Модель: `/mnt/tank/scratch/ibagautdinov/models/Qwen2.5-0.5B` (симлинк `~/models`)
- Чекпоинты: `/mnt/tank/scratch/ibagautdinov/minionerec_runs/` (home-квота 30GB — не туда)
- Логи: `~/MiniOneRec/logs/`

## Workflow

Изменения — локально → commit/push в форк → на кластере `git pull`.
Сабмит только со sphinx: `ssh -J ctlab.itmo.ru ibagautdinov@sphinx`.

```bash
cd ~/MiniOneRec

# SFT: смок / полный
sbatch --time=00:40:00 --export=ALL,SAMPLE=2048,MICRO_BS=4 cluster/sbatch_sft_05b.sh
sbatch --export=ALL cluster/sbatch_sft_05b.sh

# RL (нужен SFT-чекпоинт; ВАЖНО: именно .../final_checkpoint —
# только там сохраняется токенизатор с SID-токенами, корень output_dir — без него)
sbatch --export=ALL,MODEL_PATH=/mnt/tank/scratch/ibagautdinov/minionerec_runs/sft_.../final_checkpoint cluster/sbatch_rl_05b.sh

# Оценка (HR@K / NDCG@K, constrained beam search 50) — тоже от final_checkpoint
sbatch --export=ALL,EXP_NAME=/mnt/tank/scratch/ibagautdinov/minionerec_runs/.../final_checkpoint cluster/sbatch_eval.sh
```

## Грабли

- gateway `horse` без AVX — тяжёлый python только на compute-нодах (srun/sbatch);
- `--mem=24G` максимум для midas (RealMemory=32014MB);
- не задавать `#SBATCH --chdir` (ломает сабмит), `--output` — абсолютный путь, каталог заранее;
- в `rl.py` `train_batch_size` — per-device; глобальный батч генерации кратен `num_generations`;
- модель — base, не Instruct (Instruct ломает constrained decoding, см. README апстрима 2026-01-04).
