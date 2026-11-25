# Experiment log

This file is the run-by-run record of the project. `description.md` and `description_en.md` summarize the protocol and the current overall conclusions; this log records individual studies, their artifacts, and their limits. “Confirmed” means the result is supported by a retained CSV and/or an explicit run command in the project history. Exploratory studies with missing artifacts are identified as such; their exact scores must not be reconstructed from memory.

## Confirmed studies

### E1. Main cross-dataset LOOCV, frozen encoder, K=10

- **Question:** How do the comparison/adaptation methods perform across six held-out visual domains?
- **Protocol:** CUB, Cars, Dogs, Flowers, NABirds, Aircraft; five seeds; 50 target classes; 10 support and 10 query images per class; five source datasets train/use the shared encoder. The source checkpoints were cached, so the five seeds measure support/query and adaptation variability, not independent encoder training.
- **Methods:** ProtoNet, Linear, RelationNet, RelationGCN, BN-adapt, Nonspatial, Hyper-Pure, Hyper, CosClf, Fine-tune, Hyper-Sym.
- **Artifact:** [`results/final_loocv_k10.csv`](results/final_loocv_k10.csv)
- **Conclusion:** Hyper-Sym has the highest six-fold mean (0.8663); RelationGCN leads on Flowers. Fold-level results and limitations are in `description.md` §5.

### E2. Support-size comparison, frozen encoder, K=5/10/15

- **Question:** How does the number of support examples affect performance?
- **Protocol:** Same six datasets and 11 methods as E1, five seeds, 10 query images per class. K=5 and K=15 were launched as separate `cross_validate.py` runs. K=10 is the corrected final table from E1.
- **Artifacts:** [`results/final_loocv_k5.csv`](results/final_loocv_k5.csv), [`results/final_loocv_k10.csv`](results/final_loocv_k10.csv), [`results/final_loocv_k15.csv`](results/final_loocv_k15.csv)
- **Conclusion:** Hyper-Sym has the highest fold-averaged score at each K (0.8135, 0.8663, 0.8821); RelationGCN wins the Flowers fold at each K. These are descriptive results over six folds and five adaptation seeds.

### E3. Dedicated Hyper-Sym LOOCV, K=10

- **Command recorded by the project owner:** six datasets; only `hyper-sym`; seeds 1–5; K=10; M-per-class=10; `l4`; 2 generated kernels; 16 output channels; side learning rate `3e-4`; 5 adaptation epochs; same-domain negative ratio 0.5.
- **Originally requested output:** `results/loocv_hyper_sym.csv` (not retained after cleanup). The final consolidated comparison is in `results/final_loocv_k10.csv` and has the same Hyper-Sym method/protocol settings.
- **Conclusion:** Hyper-Sym’s per-fold results are compared against RelationGCN and other methods in E1/E2. The standalone CSV is not available to verify row-for-row identity with the consolidated run.

### E4. Fine-tuning combinations, K=10

- **Question:** Does unfreezing part of the encoder before applying each target adaptation help?
- **Protocol:** Six folds, five seeds; fine-tune `layer4` and the embedding head, up to 15 epochs with early stopping and an internal support validation split; then evaluate the selected comparison head. This is a separate protocol from E1.
- **Artifact:** [`results/final_loocv_k10_finetune.csv`](results/final_loocv_k10_finetune.csv)
- **Conclusion:** Hyper-Sym mean is 0.8577, below its frozen-encoder K=10 mean of 0.8663. Per-fold winners differ; see `description.md` §6. The separate unsuccessful end-to-end attempt is not part of this CSV.

### E5. Corrected ProtoNet K=10 table

- **Purpose:** Preserve the corrected ProtoNet values after discovering that an earlier ProtoNet column duplicated the baseline.
- **Artifact:** [`results/final_protonet_k10.csv`](results/final_protonet_k10.csv); corrected ProtoNet fields are also present in `results/final_loocv_k10.csv`.
- **Conclusion:** Use the corrected values in the final K=10 table. Do not use the superseded ProtoNet values.

### E6. Dense versus depthwise RelationGCN-style adapter, K=10

- **Question:** Could the earlier stronger RelationGCN results be due to a higher-capacity dense channel-mixing generator?
- **Protocol:** Six folds, five seeds, shared cached encoders and matching support/query samples; dense `relationgcn-full` trained for 15 epochs and compared with the corrected depthwise adapter and Hyper-Sym.
- **Artifact:** [`results/final_relationgcn_full_k10.csv`](results/final_relationgcn_full_k10.csv)
- **Conclusion:** Dense is higher than depthwise on all six folds and has a mean of 0.8689 versus 0.8322; Hyper-Sym is 0.8663. The 0.26 percentage-point mean margin over Hyper-Sym is too small to call decisive from five seeds. Dense uses roughly 153.4M generator parameters and is much slower/more memory-intensive. This is a reconstruction of the earlier design, not proof that it exactly matches the historical implementation.

## Exploratory studies and incomplete records

### X1. Hyper feature-level selection

The project contains `scripts/sweep_hyper.py --phase level`, which evaluates Hyper/Hyper-Pure at `l1`–`l4`. The project owner recalls that multiple checks led to choosing `layer4` for Hyper-Sym, while also recalling that `layer3` may have looked preferable. The corresponding sweep output was not retained during cleanup, so this log cannot confirm which architecture, seeds, exact scores, or selection criterion produced the final choice. The final cross-dataset configuration does use `l4`; see E1/E3. **Do not treat the sweep script’s existence as evidence of its results.** Re-run a controlled layer ablation and retain its raw CSV before making a stronger claim.

### X2. Hyper and side-block hyperparameter sweeps

Scripts define sweeps over architecture type, feature level, number/width of generated kernels, generator mode, learning rate, adaptation epochs, and regularization/held-out early stopping. They print aggregate results, but no raw result tables for these exploratory sweeps remain in `results/`. Execution of every phase and its selected configuration cannot be verified from the retained files. The final settings are documented in E1–E3, but should not be presented as independently reproducible sweep winners without recovering the original logs or rerunning the sweeps.

### X3. Negative-pair ratio, in-domain checks, and older LOOCV runs

Earlier artifacts existed for negative same-domain ratios 0.2/0.5/0.8, in-domain CUB/Cars, and initial K=5/10/15 and Hyper-Sym runs. Those CSVs and associated logs were removed when the results folder was cleaned. No numeric conclusions from them are included here. The current final LOOCV CSVs are the source of truth for the reported cross-domain comparisons.

### X4. End-to-end fine-tuning experiment

The project owner reports a separate unsuccessful attempt to unfreeze and train the encoder end-to-end. It was not recorded in the retained result tables and is excluded from all conclusions above. No score or detailed failure diagnosis is available in this log.

## Reporting rules

1. Keep the exact launch command, code revision, environment, seed list, and raw output CSV for every new run.
2. Mark smoke tests and failed runs separately; never pool them with five-seed final results.
3. Record whether source encoders were retrained or loaded from cache.
4. Treat the six folds as a small domain sample and the five seeds as adaptation/sampling variability when encoders are fixed.
5. When comparing architectures or feature levels, use identical folds, support/query draws, seeds, and training budgets where possible.

---

# Журнал экспериментов

Этот файл содержит записи по отдельным экспериментам. В `description.md` и `description_en.md` приведены общий протокол и сводные выводы; здесь фиксируются конкретные запуски, файлы результатов и ограничения. «Подтверждён» означает, что результат поддерживается сохранённым CSV и/или точной командой запуска из истории проекта. Для исследований без исходных таблиц это явно указано: точные оценки нельзя восстанавливать по памяти.

## Подтверждённые эксперименты

### E1. Основной междоменный LOOCV, замороженный энкодер, K=10

- **Вопрос:** Как методы сравнения и адаптации работают на шести отложенных визуальных доменах?
- **Протокол:** CUB, Cars, Dogs, Flowers, NABirds, Aircraft; пять seed; 50 целевых классов; 10 support- и 10 query-изображений на класс; общий энкодер обучался/использовался на остальных пяти датасетах. Source-чекпойнты были взяты из кэша, поэтому пять seed измеряют вариацию support/query и адаптации, а не независимого обучения энкодера.
- **Методы:** ProtoNet, Linear, RelationNet, RelationGCN, BN-adapt, Nonspatial, Hyper-Pure, Hyper, CosClf, Fine-tune, Hyper-Sym.
- **Результаты:** [`results/final_loocv_k10.csv`](results/final_loocv_k10.csv)
- **Вывод:** У Hyper-Sym наибольшее среднее по шести фолдам — 0.8663; RelationGCN лидирует на Flowers. Подробные значения по фолдам и ограничения приведены в `description.md`, §5.

### E2. Сравнение числа support-примеров, замороженный энкодер, K=5/10/15

- **Вопрос:** Как число примеров для адаптации влияет на качество?
- **Протокол:** Те же шесть датасетов и 11 методов, что в E1; пять seed и 10 query-изображений на класс. K=5 и K=15 запускались отдельными командами `cross_validate.py`; K=10 — исправленная финальная таблица из E1.
- **Результаты:** [`results/final_loocv_k5.csv`](results/final_loocv_k5.csv), [`results/final_loocv_k10.csv`](results/final_loocv_k10.csv), [`results/final_loocv_k15.csv`](results/final_loocv_k15.csv)
- **Вывод:** У Hyper-Sym наибольшее среднее по фолдам для каждого K: 0.8135, 0.8663 и 0.8821. RelationGCN выигрывает фолд Flowers при каждом K. Это описательные результаты по шести фолдам и пяти seed адаптации.

### E3. Отдельный LOOCV Hyper-Sym, K=10

- **Команда, зафиксированная владельцем проекта:** шесть датасетов; только `hyper-sym`; seed 1–5; K=10; M-per-class=10; `l4`; 2 генерируемых ядра; 16 выходных каналов; learning rate side-блока `3e-4`; 5 эпох адаптации; доля внутридоменных негативных пар 0.5.
- **Изначальный файл:** `results/loocv_hyper_sym.csv` (удалён при очистке). Финальное сводное сравнение находится в `results/final_loocv_k10.csv` и содержит Hyper-Sym с теми же настройками и протоколом.
- **Вывод:** Результаты Hyper-Sym по фолдам сопоставлены с RelationGCN и другими методами в E1/E2. Отдельного CSV больше нет, поэтому построчное совпадение со сводным запуском проверить нельзя.

### E4. Комбинации с fine-tuning, K=10

- **Вопрос:** Помогает ли размораживание части энкодера перед target-адаптацией?
- **Протокол:** Шесть фолдов, пять seed; fine-tuning `layer4` и embedding head, до 15 эпох с early stopping и внутренним разбиением support для валидации; затем обучалась/применялась выбранная голова сравнения. Протокол отличается от E1.
- **Результаты:** [`results/final_loocv_k10_finetune.csv`](results/final_loocv_k10_finetune.csv)
- **Вывод:** Среднее Hyper-Sym равно 0.8577, ниже результата с замороженным энкодером при K=10 (0.8663). Лучший метод зависит от фолда; см. `description.md`, §6. Отдельная неудачная end-to-end попытка в этот CSV не входит.

### E5. Исправленная таблица ProtoNet, K=10

- **Цель:** Сохранить корректные значения ProtoNet после обнаружения, что прежняя колонка ProtoNet дублировала baseline.
- **Результаты:** [`results/final_protonet_k10.csv`](results/final_protonet_k10.csv); исправленные поля также включены в `results/final_loocv_k10.csv`.
- **Вывод:** Для итогов K=10 использовать исправленные значения. Устаревшие значения ProtoNet не использовать.

### E6. Dense и depthwise варианты RelationGCN-подобного адаптера, K=10

- **Вопрос:** Могли ли более сильные прежние результаты RelationGCN объясняться более ёмким dense-генератором со смешиванием каналов?
- **Протокол:** Шесть фолдов, пять seed, общие кэшированные энкодеры и одинаковые support/query-разбиения; dense `relationgcn-full` обучался 15 эпох и сравнивался с исправленным depthwise-адаптером и Hyper-Sym.
- **Результаты:** [`results/final_relationgcn_full_k10.csv`](results/final_relationgcn_full_k10.csv)
- **Вывод:** Dense-вариант выше depthwise на всех шести фолдах: среднее 0.8689 против 0.8322; у Hyper-Sym — 0.8663. Преимущество над Hyper-Sym в 0.26 процентного пункта слишком мало, чтобы считать его убедительным при пяти seed. У dense-генератора около 153.4 млн параметров, он значительно медленнее и требует больше памяти. Это реконструкция прежней идеи, а не доказательство точного совпадения с исторической реализацией.

## Исследовательские запуски с неполными записями

### X1. Выбор уровня признаков Hyper

В проекте есть `scripts/sweep_hyper.py --phase level`, который сравнивает Hyper/Hyper-Pure на `l1`–`l4`. Владелец проекта помнит, что после нескольких проверок для Hyper-Sym выбрали `layer4`, хотя ему также казалось, что мог лучше выглядеть `layer3`. Таблица sweep-а при очистке не сохранилась, поэтому здесь нельзя подтвердить, какая именно архитектура и выборка seed использовались, каковы были точные оценки и критерий выбора. В финальной междоменной конфигурации используется `l4`; см. E1/E3. **Наличие sweep-скрипта само по себе не доказывает результаты sweep-а.** Для строгого вывода нужно повторить контролируемую абляцию уровней и сохранить исходный CSV.

### X2. Гиперпараметрические sweep-ы Hyper и side-блоков

Скрипты задают перебор типа архитектуры, уровня признаков, числа/ширины ядер, способа генерации, learning rate, числа эпох, регуляризации и early stopping на отложенной части support. Они печатают агрегаты, но исходные таблицы этих исследований в `results/` не сохранились. Нельзя проверить, выполнялась ли каждая фаза и по каким результатам выбирали финальную конфигурацию. Настройки E1–E3 задокументированы, однако без исходных логов их не следует называть подтверждёнными победителями sweep-а.

### X3. Доля негативных пар, внутридоменные проверки и старые LOOCV

Ранее существовали артефакты для долей внутридоменных негативов 0.2/0.5/0.8, in-domain CUB/Cars и первоначальных прогонов K=5/10/15 и Hyper-Sym. Эти CSV и связанные логи удалены при очистке. Численные выводы по ним здесь не приводятся. Для текущего междоменного сравнения источником являются финальные LOOCV CSV.

### X4. End-to-end fine-tuning

Владелец проекта сообщил об отдельной неудачной попытке разморозить и обучить энкодер end-to-end. Она не записана в сохранившихся таблицах и не входит в выводы выше. Оценка и подробная диагностика не сохранились.

## Правила дальнейшей отчётности

1. Для каждого запуска сохранять точную команду, ревизию кода, окружение, список seed и исходный CSV.
2. Smoke-тесты и неудачные запуски записывать отдельно; не объединять их с финальными результатами по пяти seed.
3. Фиксировать, переобучались ли source-энкодеры или загружались из кэша.
4. Интерпретировать шесть фолдов как небольшую выборку доменов, а пять seed при фиксированных энкодерах — как вариацию адаптации и выборки.
5. При сравнении архитектур и уровней признаков по возможности использовать одинаковые фолды, support/query-разбиения, seed и бюджеты обучения.
