# Breast Cancer Detection - ML Classification

Machine learning pipeline for breast cancer classification with multiple models and API.

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd breast-cancer-detection
pip install -r requirements.txt

# Run pipeline
python scripts/main_pipeline.py

# Start API
python api.py

# Run tests
python -m pytest tests/

Project Structure
text

breast-cancer-detection/
- api.py                    # FastAPI application
- requirements.txt          # Dependencies
- README.md                 # Documentation
- .gitignore               # Git ignore file
- data/data.csv            # Dataset
- tests/                   # Test files
 - test_api.py
 - test_unit.py
- scripts/                 # Main code
 - __init__.py
 - config.py
 - data_processor.py
 - evaluator.py
 - fuzzy_enhancer.py
 - main_pipeline - esm miporsad.py
 - main_pipeline.py
 - model_trainer.py
 - utils.py
 - visualizer.py
- .github/workflows/ci.yml # CI/CD

Main Components
1. Data Processing (scripts/data_processor.py)

    Loads and cleans breast cancer dataset

    Handles missing values and outliers

    Scales features and applies PCA

2. Model Training (scripts/model_trainer.py)

    Trains 4 classifiers:

        Decision Tree

        Naive Bayes

        Perceptron

        K-Nearest Neighbors

3. Fuzzy Enhancement (scripts/fuzzy_enhancer.py)

    Fuzzy C-Means clustering

    Feature space enhancement

    Model performance improvement

4. Evaluation (scripts/evaluator.py)

    Calculates accuracy, precision, recall, F1

    Generates confusion matrices

    Compares model performance

5. REST API (api.py)

    FastAPI web service

    Endpoints: /predict, /health, /models

    Real-time predictions

API Usage
bash

# Start server
python api.py

# Access documentation
# http://localhost:8000/docs

# Example prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [/* 30 features */]}'

Pipeline Modes

Run with: python scripts/main_pipeline.py <mode>

    basic - Simple processing

    preprocessed - Full preprocessing

    fuzzy - With fuzzy enhancement

    full - Complete analysis

    all - Run all modes

Dependencies

Main packages:

    scikit-learn, pandas, numpy

    fastapi, uvicorn, pydantic

    matplotlib, seaborn

    scikit-fuzzy

    pytest

See requirements.txt for complete list.
Dataset

Wisconsin Breast Cancer Dataset:

    569 samples, 30 features

    Binary classification: Benign vs Malignant

    Features: radius, texture, perimeter, area, etc.

CI/CD

Automated testing with GitHub Actions:

    Unit and API tests

    Code linting (black, isort, flake8)

    Structure validation

    Runs on Python 3.9, 3.10, 3.11
