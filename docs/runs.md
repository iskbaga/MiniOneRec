# Журнал ранов

Одна строка на ран. Правила ведения — в [experiments.md](experiments.md).
Метрики — из `calc.py` (HR@K/NDCG@K на тесте, CC = доля невалидных генераций).

| Дата | run_name | Ветка @ коммит | Стадия | База | Ключевое | HR@5 | HR@10 | N@5 | N@10 | CC | Вывод |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-11 | sft_qwen05b_Industrial_and_Scientific_797175 | itmo-baseline @ f7d4486 | SFT-смок | Qwen2.5-0.5B | SAMPLE=2048, проверка пайплайна | — | — | — | — | — | ✅ пайплайн жив (26.5 мин, train_loss 1.80) |
| 2026-08-11 | sft_qwen05b_Industrial_and_Scientific_797177 | itmo-baseline @ f7d4486 | SFT | Qwen2.5-0.5B | полный: 36k seq, bs1024, micro8, lr3e-4, 10 эп.+ES | | | | | | 🕐 в работе |
