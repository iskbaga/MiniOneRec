# Как вести эксперименты

Регламент для работы над статьёй (улучшение RL в генеративных рекомендерах) на базе
этого форка MiniOneRec и кластера ctlab ИТМО. Цель: любой отчётный результат можно
воспроизвести по записи в [runs.md](runs.md) — код, данные, гиперпараметры, чекпоинт.

## Единица работы: ран

**Один ран = один slurm job = один git-коммит.** Всё связывается через `run_name`:

```
<стадия>_<модель>_<категория>_<SLURM_JOB_ID>
например: sft_qwen05b_Industrial_and_Scientific_797177
```

По `run_name` находятся все артефакты:

| Что | Где |
|---|---|
| Чекпоинт | `/mnt/tank/scratch/ibagautdinov/minionerec_runs/<run_name>/` |
| Slurm-лог | `~/MiniOneRec/logs/<стадия>-<jobid>.out` + копия `slurm.log` рядом с чекпоинтом |
| Git-состояние | первая строка лога: `=== git: <hash> @ <ветка>, dirty: N files` |
| Кривые обучения | `~/MiniOneRec/wandb/offline-run-*` (run_name совпадает) |
| Метрики eval | `results/<run_name>/final_result_*.json` — коммитятся в репо |
| Запись в журнале | строка в `docs/runs.md` |

Правило: **`dirty: 0`** в логе. Если в логе dirty > 0 — ран не считается отчётным
(код не зафиксирован в git). Сначала commit+push, потом sbatch.

## Версионирование: git-схема

- **`itmo-baseline`** — базовая ветка: неизменённый код апстрима + наша инфраструктура
  (`cluster/`, `docs/`). Сюда мержится только инфра и фиксы, НЕ исследовательские правки.
- **`exp/<имя>`** — ветка на каждое направление исследования, от `itmo-baseline`:
  `exp/reward-shaping`, `exp/adv-estimation`, `exp/dapo` … Все правки rl.py /
  minionerec_trainer.py живут в своей ветке. Имя ветки попадает в лог рана.
- Новые гиперпараметры — только через CLI-флаги (fire) с дефолтом = поведение бейзлайна,
  никаких молчаливых хардкодов: тогда один и тот же коммит запускает и бейзлайн, и вариант.
- Синхронизация с апстримом: `git fetch upstream && git rebase upstream/main itmo-baseline`,
  затем rebase exp-веток. Апстрим активно чинят (см. их Announcements) — заглядывать.
- Теги на «отчётные» состояния: `paper/<событие>` (например `paper/baseline-v1`) — то, что
  попадает в таблицы статьи.

## Цикл эксперимента

```bash
# 1. локально: ветка, правка, коммит, пуш
git checkout -b exp/my-idea itmo-baseline
...правки...
git commit -am "exp: ..." && git push -u origin exp/my-idea

# 2. на кластере: подтянуть и запустить
ssh -J ctlab.itmo.ru ibagautdinov@sphinx
cd ~/MiniOneRec && git fetch && git checkout exp/my-idea && git pull
sbatch --export=ALL,MODEL_PATH=<sft-чекпоинт> cluster/sbatch_rl_05b.sh

# 3. после завершения: eval
sbatch --export=ALL,EXP_NAME=/mnt/tank/scratch/ibagautdinov/minionerec_runs/<run_name> cluster/sbatch_eval.sh

# 4. результаты в git
#    метрики печатаются calc.py в конец eval-лога; json лежит в results/<run_name>/
git add results/<run_name> docs/runs.md && git commit -m "results: <run_name>" && git push
```

Пока джоба в очереди или бежит — **не менять рабочее дерево `~/MiniOneRec` на кластере**
(python-файлы читаются в момент старта, csv/json — по ходу). Нужно запустить два варианта
параллельно — два клона: `/mnt/tank/scratch/ibagautdinov/worktrees/<exp>` (git clone форка,
checkout нужной ветки, поправить пути в sbatch через `cd`).

## Смок-тест перед полным раном

Любое изменение кода сначала гоняется коротко (минуты, тот же скрипт):

```bash
# SFT-смок: 2048 сэмплов
sbatch --time=00:40:00 --export=ALL,SAMPLE=2048,MICRO_BS=4 cluster/sbatch_sft_05b.sh
# RL-смок: сабсет + короткий лимит
sbatch --time=01:00:00 --export=ALL,MODEL_PATH=...,SAMPLE_TRAIN=True cluster/sbatch_rl_05b.sh
```

Полный ран — только после чистого смока. Экономит очередь и нервы.

## Сиды и достоверность цифр

- Разовые прогоны годятся для итераций; **в статью — mean±std по ≥3 сидам** (лучше 6:
  0 1 7 42 100 999, как в прошлом проекте). Сид передаётся через `--seed` (sft) /
  добавить флаг в rl при необходимости.
- На этом кластере подтверждён **cuDNN non-determinism между архитектурами GPU**: один сид
  даёт разные метрики на Ampere (midas) и Pascal/Turing (разница до 0.5σ). Все сравнимые
  раны — на одном типе GPU (`--gres=gpu:3080:2` = midas). Нода печатается в начале лога.
- Сравнение вариантов RL — всегда от **одного и того же SFT-чекпоинта** (фиксировать его
  путь в runs.md).
- CC-метрика в eval (доля невалидных генераций) должна быть ~0 — иначе constrained
  decoding сломан и метрики бессмысленны (известная проблема с Instruct-моделями).

## wandb (offline)

Кластер без wandb-логина — раны пишутся offline в `~/MiniOneRec/wandb/`.
Посмотреть кривые: `rsync` папки `offline-run-*` на ноут → `wandb sync <dir>` (зальёт в
облако) или открыть локально. Для быстрой проверки хватает loss-строк в slurm-логе.

## Гигиена ресурсов

- Чекпоинты — только в scratch (`minionerec_runs/`), home-квота 30GB почти занята
  окружениями. Периодически удалять чекпоинты неудачных/промежуточных ранов
  (метрики и логи при этом остаются в git).
- midas один на всех (2×3080 Ti) — не держать интерактивные srun-сессии без дела,
  длинные раны сабмитить на ночь. Лимит очереди 14 дней, дефолт ≤2 GPU.
