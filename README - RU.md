# Диагностика рака груди - Классификация с машинным обучением

Проект классификации рака груди с несколькими моделями и API.

## Быстрый старт

```bash
git clone <repository-url>
cd breast-cancer-detection
pip install -r requirements.txt

# Запуск пайплайна
python scripts/main_pipeline.py

# Запуск API
python api.py

# Запуск тестов
python -m pytest tests/

Структура проекта
text

breast-cancer-detection/
├── api.py                    # FastAPI
├── requirements.txt          # Зависимости
├── README.md                 # Документация
├── .gitignore
├── data/data.csv            # Датасет
├── tests/                   # Тесты
│   ├── test_api.py
│   └── test_unit.py
├── scripts/                 # Основной код
│   ├── __init__.py
│   ├── config.py
│   ├── data_processor.py
│   ├── evaluator.py
│   ├── fuzzy_enhancer.py
│   ├── main_pipeline.py
│   ├── model_trainer.py
│   ├── utils.py
│   └── visualizer.py
└── .github/workflows/ci.yml

Основные компоненты

Обработка данных (scripts/data_processor.py)

    Загрузка и очистка датасета

    Обработка пропущенных значений и выбросов

    Масштабирование и PCA

Обучение моделей (scripts/model_trainer.py)

    Обучение 4 моделей: Decision Tree, Naive Bayes, Perceptron, KNN

Фаззи-улучшение (scripts/fuzzy_enhancer.py)

    Кластеризация Fuzzy C-Means

    Улучшение пространства признаков

    Улучшение производительности модели

Оценка (scripts/evaluator.py)

    Расчёт accuracy, precision, recall, F1

    Матрицы ошибок

    Сравнение моделей

API (api.py)

    FastAPI

    Endpointы: /predict, /health, /models

    Предсказания в реальном времени

Использование API
bash

python api.py

# Документация
# http://localhost:8000/docs

# Пример
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [/* 30 features */]}'

Режимы пайплайна
bash

python scripts/main_pipeline.py <mode>

    basic — простая обработка

    preprocessed — полная предобработка

    fuzzy — с фаззи-улучшением

    full — полный анализ

    all — все режимы

Зависимости

    scikit-learn, pandas, numpy

    fastapi, uvicorn, pydantic

    matplotlib, seaborn

    scikit-fuzzy

    pytest

Полный список в requirements.txt.
Датасет

Wisconsin Breast Cancer Dataset:

    569 образцов, 30 признаков

    Бинарная классификация: доброкачественная vs злокачественная

    Признаки: radius, texture, perimeter, area и др.

CI/CD

Автоматическое тестирование с GitHub Actions:

    Юнит-тесты и тесты API

    Линтинг (black, isort, flake8)

    Проверка структуры

    Запуск на Python 3.9, 3.10, 3.11
