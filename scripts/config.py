"""
Central configuration for Breast Cancer Classification project
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

PCA_VARIANCE = 0.95

FUZZY_N_CLUSTERS = 2
FUZZY_M = 2
FUZZY_ERROR = 0.005
FUZZY_MAXITER = 1000

MODELS_CONFIG = {
    "Decision Tree": {
        "class": "DecisionTreeClassifier",
        "params": {"random_state": RANDOM_STATE}
    },
    "Naive Bayes": {
        "class": "GaussianNB",
        "params": {}
    },
    "Perceptron": {
        "class": "Perceptron",
        "params": {"random_state": RANDOM_STATE}
    },
    "KNN": {
        "class": "KNeighborsClassifier",
        "params": {}
    }
}