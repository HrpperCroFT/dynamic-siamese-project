# Dynamic Siamese Project

Research code for cross-domain few-shot image similarity. A shared image encoder is paired with comparison heads, including reference-conditioned convolution kernels. The experimental protocol and current findings are documented in [`description.md`](description.md) (Russian) and [`description_en.md`](description_en.md) (English).

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Run project commands from the repository root using `.venv\Scripts\python.exe`.

## Datasets

The downloader handles datasets with accessible public archives and then checks each dataset with the project's loader:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py
.\.venv\Scripts\python.exe scripts\download_datasets.py --only dogs,flowers
.\.venv\Scripts\python.exe scripts\download_datasets.py --check-only
```

CUB, Stanford Cars, and NABirds require a separately obtained archive or acceptance of the dataset's distribution terms. The downloader prints the source and expected location for these datasets. Put extracted data under `data/` using the directory names in `src/dynamic_siamese/datasets_multi.py`. Local datasets and model checkpoints are excluded from Git.

## Experiments

Run the full configured experiment suite with:

```powershell
.\.venv\Scripts\python.exe scripts\run_all_experiments.py --arches extended --seeds 1,2,3,4,5 --force
```

For a direct leave-one-dataset-out run:

```powershell
.\.venv\Scripts\python.exe scripts\cross_validate.py --datasets cub,cars,dogs,flowers,nabirds,aircraft --arches hyper-sym --seeds 1,2,3,4,5 --K 10 --M-per-class 10 --output results/my_run.csv
```

Scripts under `scripts/` run experiments and set up datasets. One-off analysis utilities are in `scripts/analysis/`. Quick synthetic and CUB examples are in `examples/`.

## Results

The `results/` directory contains the final LOOCV tables described in the experiment documentation. Smoke tests, temporary logs, and superseded outputs should not be treated as final measurements.

---

# Проект Dynamic Siamese

Исследовательский код для few-shot сравнения изображений между доменами. Общий энкодер изображений используется вместе с головами сравнения, в том числе с генерацией свёрточных ядер по изображению-референсу. Протокол экспериментов и текущие выводы описаны в [`description.md`](description.md) (русский) и [`description_en.md`](description_en.md) (английский).

## Установка (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Запускайте команды из корня репозитория через `.venv\Scripts\python.exe`.

## Датасеты

Загрузчик скачивает датасеты с доступными публичными архивами, а затем проверяет, что каждый набор читается загрузчиком проекта:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py
.\.venv\Scripts\python.exe scripts\download_datasets.py --only dogs,flowers
.\.venv\Scripts\python.exe scripts\download_datasets.py --check-only
```

Для CUB, Stanford Cars и NABirds архив нужно получить отдельно или принять условия распространения. Скрипт выводит источник и ожидаемый путь для этих наборов. Распакуйте данные в `data/`, используя имена каталогов из `src/dynamic_siamese/datasets_multi.py`. Локальные датасеты и веса моделей исключены из Git.

## Эксперименты

Запуск полного настроенного набора экспериментов:

```powershell
.\.venv\Scripts\python.exe scripts\run_all_experiments.py --arches extended --seeds 1,2,3,4,5 --force
```

Пример прямого leave-one-dataset-out запуска:

```powershell
.\.venv\Scripts\python.exe scripts\cross_validate.py --datasets cub,cars,dogs,flowers,nabirds,aircraft --arches hyper-sym --seeds 1,2,3,4,5 --K 10 --M-per-class 10 --output results/my_run.csv
```

Сценарии экспериментов и настройки датасетов находятся в `scripts/`, разовые утилиты анализа — в `scripts/analysis/`. Примеры быстрых запусков на синтетических данных и CUB находятся в `examples/`.

## Результаты

В каталоге `results/` лежат финальные LOOCV-таблицы, описанные в документации экспериментов. Smoke-тесты, временные логи и устаревшие результаты не следует считать итоговыми измерениями.
