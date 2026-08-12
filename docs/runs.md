# Журнал ранов

Одна строка на ран. Правила ведения — в [experiments.md](experiments.md).
Метрики — из `calc.py` (HR@K/NDCG@K на тесте, CC = доля невалидных генераций).

| Дата | run_name | Ветка @ коммит | Стадия | База | Ключевое | HR@5 | HR@10 | N@5 | N@10 | CC | Вывод |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-11 | sft_qwen05b_Industrial_and_Scientific_797175 | itmo-baseline @ f7d4486 | SFT-смок | Qwen2.5-0.5B | SAMPLE=2048, проверка пайплайна | — | — | — | — | — | ✅ пайплайн жив (26.5 мин, train_loss 1.80) |
| 2026-08-11 | sft_qwen05b_Industrial_and_Scientific_797177 | itmo-baseline @ f7d4486 | SFT | Qwen2.5-0.5B | полный: 36k seq, bs1024, micro8, lr3e-4, 10 эп.+ES | 0.0955 | 0.1244 | 0.0759 | 0.0853 | 0 | ✅ 1ч40м, ES эп. 5.5, eval_loss 1.663; eval=797178 (beam 50); HR@1=0.0556, HR@20=0.1604, HR@50=0.2250 |
| 2026-08-11 | rl_qwen05b_Industrial_and_Scientific_797179 | itmo-baseline @ dbfc3b8 | RL-смок | SFT 797177 | EPOCHS=0.01 | — | — | — | — | — | ❌ импорт-краш: sklearn нет в requirements апстрима (доставлен в env); sacct показал COMPLETED — после этого в скрипты добавлен set -e |
| 2026-08-11 | rl_qwen05b_Industrial_and_Scientific_797180 | itmo-baseline @ dbfc3b8 | RL-смок | SFT 797177 | EPOCHS=0.01, eval_step 0.0999 | — | — | — | — | — | ⏱ TIMEOUT 2ч, но механика ✅: 33/33 шагов, reward/KL живые, 16 уникальных валидных бимов; шаг ~7–8с, eval ~8.7 мин × 10 раз съел время; в финале IndexError (разбирается) |
