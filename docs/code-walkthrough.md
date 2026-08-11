# MiniOneRec: разбор кода

Конспект по состоянию форка на ветке `itmo-baseline` (август 2026). Ссылки — `файл:строка`.
Цель — быстро ориентироваться при модификации RL-части. Сводка граблей — в конце.

## 1. Общая картина

Пайплайн: **SID-конструкция** (rq/, оффлайн, готовые SID уже в `data/Amazon/index/`) →
**SFT** (`sft.py`) → **RL** (`rl.py` + `minionerec_trainer.py`) → **eval**
(`split.py` → `evaluate.py` × N GPU → `merge.py` → `calc.py`).

Айтем = 3 SID-токена `<a_X><b_Y><c_Z>` (3 уровня RQ, кодбуки по 256), добавляются в словарь
Qwen как обычные (не special) токены. Генерация всегда ограничена trie валидных SID.

**Ключевой вывод по RL для статьи:** `ReReTrainer` — замороженный форк GRPOTrainer из trl
~0.14-0.15 (унаследован от `transformers.Trainer`, не от trl!). Одна итерация на батч →
ratio≡1, клиппинга нет вообще. Фактически это **on-policy REINFORCE с групповым
whitening-baseline + k3-KL к ref-политике + constrained beam sampling**. «DAPO»/«GSPO»
в коде — только схемы агрегации лосса; clip-higher, resample-until-nonzero-std и прочее
из статей — отсутствуют. Это большое пространство для улучшений.

## 2. Данные

### Форматы

- **CSV** (`data/Amazon/{train,valid,test}/`): колонки `user_id, history_item_title,
  item_title, history_item_id, item_id, history_item_sid, item_sid`. Списковые поля —
  python-repr строки, парсятся `eval()` (data.py:136 и др.).
- **info** (`data/Amazon/info/*.txt`): TSV `sid \t title \t item_id` — каталог валидных SID,
  из него строится trie для constrained decoding и словарь метрик calc.py.
- **index.json**: `{item_id: ["<a_236>","<b_231>","<c_226>"]}` — источник SID-токенов
  для расширения словаря (sft.py:39).
- **item.json**: `{item_id: {title, description, brand, categories}}`; description —
  строка-repr списка, отсюда `eval`-развороты.

История: готовится на препроцессинге — скользящие окна, **≤10 последних айтемов**
(amazon18_data_process.py:261). Сплит **не** leave-last-out: все примеры глобально
сортируются по timestamp таргета и режутся 8:1:1 хронологически
(amazon18_data_process.py:277-296) — юзер может быть во всех сплитах.

### Датасеты (data.py)

Общее: шаблон `### User Input:\n{input}\n\n### Response:\n{output}` (data.py:80-84);
labels: промпт замаскирован −100, учится только таргет; обрезка `[-max_len:]` — срезает
**начало** промпта (инструкцию), таргет живёт всегда (data.py:457-465).

SFT-микс (конкатенация в sft.py:191-205):
| Класс | Задача | Промпт (суть) |
|---|---|---|
| `SidSFTDataset` (data.py:397) | SID-история → SID следующего | `The user has interacted with items <sid>, <sid> … predict the next possible item` |
| `SidItemFeatDataset` (data.py:678) | выравнивание sid↔title | `Which item has the title: …?` / `What is the title of item "<sid>"?` |
| `FusionSeqRecDataset` (data.py:1126) | SID-история → **title** следующего | `…recommend the next item… Tell me the title` |

RL-микс (rl.py:85-102): `SidDataset` (тот же seq-промпт, но `{"prompt","completion"}`) +
`RLTitle2SidDataset` (title/description→SID по всему каталогу) + `RLSeqTitle2SidDataset`
(title-история→SID, sample=10000). Ещё 4 датасета закомментированы — готовые варианты
для расширения микса.

Ground truth в RL восстанавливается **по строке промпта** через два словаря:
`prompt2history` → `history2target` (rl.py:117-131) — любое расхождение промпта байт-в-байт
даёт KeyError, одинаковая история с разными таргетами перезаписывается (грабля).

## 3. SFT (sft.py)

Флоу `train()` (sft.py:90-267):
1. `TokenExtender` читает index.json, собирает **отсортированный** список SID-токенов
   (детерминированные id), `tokenizer.add_tokens` + `model.resize_token_embeddings`
   (sft.py:149-159). Явной инициализации новых эмбеддингов нет → поведение зависит от
   версии transformers (у нас 4.57 → mean-resizing).
2. Датасеты → материализация в HF Dataset в RAM (sft.py:217-221), shuffle seed 42.
3. TrainingArguments (sft.py:225-253): lr 3e-4, **linear** scheduler + `warmup_steps=20`
   (кастомный cosine в коде — мёртвый), bf16=True, eval/save каждые 5% шагов,
   `save_total_limit=1`, `load_best_model_at_end` по eval_loss,
   `EarlyStoppingCallback(patience=3)`. Grad accum = `batch_size/micro/world_size`.
4. Сохранение (sft.py:262-267): в корень `output_dir` — лучшая модель **без токенизатора**;
   в `output_dir/final_checkpoint/` — модель+токенизатор с SID-токенами.
   **Для eval/RL использовать только `final_checkpoint`.**

Режимы: `train_from_scratch` — та же архитектура со случайной инициализацией;
`freeze_LLM` — учить только эмбеддинги новых токенов, но **сломан**: `original_vocab_size`
не определён → NameError (sft.py:169) — чинить, если понадобится.

## 4. Constrained decoding (LogitProcessor.py + построение trie)

- Trie кодируется hash-словарём: ключ = `'-'.join(token_ids)` префикса, значение —
  множество допустимых следующих токенов. Строится дважды одинаково: evaluate.py:72-119
  и minionerec_trainer.py:529-572. Корневой ключ — токены `### Response:\n` (ровно 3 токена
  у Qwen — `prefix_index=3`; «gpt2» в имени модели → 4). После полного SID разрешён
  только EOS.
- `ConstrainedLogitsProcessor` (LogitProcessor.py:24-73): на шаге 0 ключ — последние 3
  токена **промпта** (поэтому каждый промпт обязан кончаться на `### Response:\n`),
  дальше — последние `count` сгенерированных. Пустое множество допустимых → warning +
  принудительный EOS (растёт CC-метрика). **`self.count` не сбрасывается** — процессор
  строго одноразовый, на каждый `generate` нужен новый экземпляр.

## 5. Eval (evaluate.py → calc.py)

- `evaluate.py`: `EvalSidDataset` (внимание: формулировка промпта чуть отличается от
  train `SidSFTDataset` — data.py:623 vs 416), чистый beam search
  `num_beams=50, num_return_sequences=50` + constrained processor, left-padding вручную.
  Выход: в каждый пример пишется `predict` = 50 SID-строк по убыванию beam score.
- `calc.py`: HR@k и NDCG@k для k ∈ {1,3,5,10,20,50}; ранг = позиция первого совпадения
  с `output`; `NDCG@k = 1/log2(rank+2)` (один релевантный айтем). **CC — абсолютное число
  невалидных генераций** по всем бимам (не доля); при исправном constrained decoding ≈ 0,
  ненулевой CC = сломан trie/шаблон (частая причина — Instruct-модель вместо base).
- `split.py`/`merge.py` — нарезка теста по GPU и склейка json-результатов, завязаны на
  имена `{cuda_id}.csv|.json`.

## 6. RL (rl.py + minionerec_trainer.py)

### Модели

Policy грузится дважды (rl.py:136 — лишняя, только ради device/эмбеддингов;
minionerec_trainer.py:271 — рабочая), ref-модель — замороженная копия policy
(:286-288). Итого ~3 копии весов на процесс — для 0.5B терпимо, для 1.5B+ чинить
(убрать первую загрузку).

### Награды (rl.py:156-258)

Вектор ранговых штрафов (rl.py:156-157):
```
ndcg_rewards[i] = −(1/log2(i+2)) / Σ_{j=0..G−1} 1/log2(j+2),  Σ = −1
```
- `rule_reward` (rl.py:186): бинарный exact-match с таргетом → 1.0 / 0.0.
- `ndcg_rule_reward` (rl.py:160): в группе из G=16 бимов (best-first): совпадение → 0;
  промах на позиции p → `ndcg_rewards[p]` (штраф тем больше, чем выше бим);
  **если GT вообще не попал в группу — вся группа получает 0** (сигнал «подними GT
  выше», а не «всё плохо»). Ранговый смысл валиден только при `beam_search=True`.
- `reward_type=ranking` (бейзлайн) = `rule + ndcg_rule` с весами 1:1.
  Ещё есть `semantic` (косинус ada-эмбеддингов) и `sasrec` (CF-скор) — требуют внешних
  артефактов.

### Генерация (minionerec_trainer.py:779-856)

`beam_search=True` (бейзлайн): промпты дедуплицируются, один `generate` с
`num_beams=G, num_return_sequences=G, do_sample=True` — **стохастический beam sampling**
по trie; все 16 гипотез — разные валидные SID, отсортированы по score. Без beam_search —
обычный ancestral sampling G копий. `dynamic_sampling` (только без beam): oversample
1.5×G + курирование группы (все копии GT + частотные негативы), но с ре-токенизацией
текстов (грабля №6). `add_gt`: подмена последнего ролаута группы на GT — off-policy
без коррекции + кривая арифметика (грабля №5).

### Advantage и loss

- Групповой whitening (minionerec_trainer.py:965-972): `A_i = (r_i − mean_G)/(std_G+1e-4)`,
  gather по всем процессам, потом срез своего куска. Группа с константной наградой
  (например, все нули) → A=0 → градиент только от KL.
- `compute_loss` (:1035-1073): `per_token_loss = −(exp(logπ−logπ.detach())·A − β·KL)`,
  ratio≡1 (одна итерация). KL — k3-оценщик (:1049). Ветки: default — среднее по токенам
  последовательности, потом по батчу; `dapo=True` — token-level нормализация по батчу;
  `gspo=True` — sequence-level ratio. Клиппинга нет ни в одной ветке.
- `sync_ref_model=True` (бейзлайн): EMA-синк ref←policy каждые 512 шагов, α=0.6
  (дефолты trl; поля `ref_model_sync_steps`/`ref_model_mixup_alpha` в GRPOConfig).

### Ключевые флаги (дефолт rl.py / бейзлайн rl.sh)

| Флаг | Значения | Смысл |
|---|---|---|
| `reward_type` | rule / **ranking** | состав наград (см. выше) |
| `num_generations` | 16 | G: группа GRPO = число бимов |
| `beam_search` | False / **True** | constrained beam sampling vs сэмплинг |
| `beta` | 0.04 / **1e-3** | вес KL (β=0 не отключает ref-forward) |
| `sync_ref_model` | False / **True** | EMA-синк ref-модели |
| `learning_rate` | 1e-6 / **1e-5** | + cosine, warmup_ratio 0.03, max_grad_norm 0.3 |
| `sample_train` | False | RL на хвостовых 80% train (работает только если «sft» в model_path!) |
| `add_gt`, `dynamic_sampling`, `dapo`, `gspo` | False | см. выше |
| `mask_all_zero` | False | **мёртвый флаг** — объявлен, никуда не передаётся |
| `test_during_training` | True / **False** | HR/NDCG на train-батчах; ломает stateful processor — держать False |

Батч-семантика: `train_batch_size` — **per-device**; глобальный батч кратен G проверяется,
но ветки beam/add_gt молча требуют кратности **per-device** батча G (64/16 ✓ у авторов;
16/16 ✓ у нас на midas).

### Optim: `paged_adamw_32bit` (bitsandbytes), max_completion_length=128, max_prompt_length=512 (режет длинные title-истории слева, молча).

## 7. Карта точек вмешательства (для статьи)

| Что менять | Где |
|---|---|
| Reward shaping | rl.py:156-245 (функции), :249-258 (композиция); веса — `GRPOConfig(reward_weights=…)`; вызов: minionerec_trainer.py:935-956. Сигнатура: `(prompts, completions, **kwargs) → list[B*G]`, группы по G подряд, best-first |
| Advantage | minionerec_trainer.py:965-972 — единственное место. LOO-baseline, отказ от /std, маскирование нулевых групп (та самая ниша `mask_all_zero`). Сырые reward'ы уже пробрасываются в compute_loss (:981,:1031), но не используются — готовый крючок |
| Sampling | ген-конфиги :479-504; ветвление beam/dynamic/plain :779-856; курирование групп `select_completion` :820-841; trie :529-572 + LogitProcessor.py (перс. ограничения — через `batch_id`, сейчас trie глобальный) |
| KL / ref | k3 :1049, β :1054; создание ref :283-292; ref-logps :897-906; EMA-синк :520-522 |
| Loss / off-policy | :1053-1064; для клиппинга/multi-iter — сохранить logπ на момент генерации в `_prepare_inputs` (:1024-1032) и считать ratio против них |
| Метрики | :985-1003, :1067-1071; test HR/NDCG :736-775 |
| Промпт-микс RL | rl.py:85-102 (+4 закомментированных датасета) |

## 8. Сводка граблей

**Ломают эксперимент:**
1. Eval/RL от корня `output_dir` — токенизатор без SID-токенов; только `final_checkpoint/`.
2. Instruct-модель вместо base → CC≫0, метрики мусор (README апстрима, 2026-01-04).
3. `test_during_training=True` — общий stateful processor на два generate → форс-EOS
   во втором (поэтому в бейзлайне False).
4. `temperature≠1.0` — конфликт двух TemperatureLogitsWarper (ValueError/двойное
   применение). Менять температуру — только убрав один из них.
5. per-device батч не кратен G — тихое перемешивание групп в beam/add_gt ветках.

**Искажают результаты молча:**
6. `dynamic_sampling` ре-токенизирует тексты — logπ считается не по тем токенам.
7. `add_gt`: `repeat` считается от числа уникальных таргетов, не G; BOS в середине.
8. `ndcg_rule_reward` при сэмплинге (без beam) — ранговый штраф превращается в шум.
9. SID-коллизии: есть айтемы с одинаковыми 3 токенами (без 4-го уровня дизамбигуации),
   словари sid↔title молча перезаписываются.
10. Train/eval промпты сформулированы по-разному (data.py:416 vs :623).
11. `history2target` перезаписывается при одинаковой истории с разными таргетами.
12. Обрезка `[-max_len:]` и `max_prompt_length=512` режут промпт слева молча.

**Просто знать:**
13. `freeze_LLM=True` — NameError (`original_vocab_size` не определён, sft.py:169).
14. `mask_all_zero` — мёртвый; `--item_k` в препроцессинге не используется;
    vLLM-путь в трейнере мёртв (NameError при use_vllm=True).
15. `eval()` на CSV-полях повсюду; `report_to=None` в HF = «все интеграции»
    (wandb работает, у нас offline).
16. Поля trl 0.24 `loss_type/epsilon/scale_rewards/num_iterations` трейнером
    игнорируются — их настройка ничего не меняет.
17. Policy в rl.py грузится дважды + ref = ~3 копии весов на процесс.
