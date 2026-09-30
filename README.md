# Breast Cancer Detection - Machine Learning Classification

Breast cancer classification project with multiple models and API.

## Quick Start

```bash
git clone <repository-url>
cd breast-cancer-detection
pip install -r requirements.txt

# Run pipeline
python scripts/main_pipeline.py

# Run API
python api.py

# Run tests
python -m pytest tests/

Project Structure
text

breast-cancer-detection/
├── api.py                    # FastAPI
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── .gitignore
├── data/data.csv            # Dataset
├── tests/                   # Tests
│   ├── test_api.py
│   └── test_unit.py
├── scripts/                 # Main code
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

Main Components

Data Processing (scripts/data_processor.py)

    Loads and cleans dataset

    Handles missing values and outliers

    Scaling and PCA

Model Training (scripts/model_trainer.py)

    Trains 4 models: Decision Tree, Naive Bayes, Perceptron, KNN

Fuzzy Enhancement (scripts/fuzzy_enhancer.py)

    Fuzzy C-Means clustering

    Feature space enhancement

    Model performance improvement

Evaluation (scripts/evaluator.py)

    Calculates accuracy, precision, recall, F1

    Confusion matrices

    Model comparison

API (api.py)

    FastAPI

    Endpoints: /predict, /health, /models

    Real-time prediction

API Usage
bash

python api.py

# Documentation
# http://localhost:8000/docs

# Example
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [/* 30 features */]}'

Pipeline Modes
bash

python scripts/main_pipeline.py <mode>

    basic — simple processing

    preprocessed — full preprocessing

    fuzzy — with fuzzy enhancement

    full — complete analysis

    all — all modes

Dependencies

    scikit-learn, pandas, numpy

    fastapi, uvicorn, pydantic

    matplotlib, seaborn

    scikit-fuzzy

    pytest

Full list in requirements.txt.
Dataset

Wisconsin Breast Cancer Dataset:

    569 samples, 30 features

    Binary classification: Benign vs Malignant

    Features: radius, texture, perimeter, area, etc.

CI/CD

Automated testing with GitHub Actions:

    Unit and API tests

    Linting (black, isort, flake8)

    Structure validation

    Runs on Python 3.9, 3.10, 3.11
