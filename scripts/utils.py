import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve
)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import MODELS_DIR
import warnings
warnings.filterwarnings('ignore')

def save_model(model, model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    safe_model_name = model_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    file_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.pkl")
    
    try:
        joblib.dump(model, file_path)
        print(f"Model {model_name} saved to: {file_path}")
        return file_path
    except (IOError, OSError) as e:
        print(f"Error saving model {model_name}: {e}")
        return None
    except Exception as e:
        print(f"Unknown error saving model {model_name}: {type(e).__name__}")
        return None

def load_model(model_name):
    safe_model_name = model_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    file_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    try:
        model = joblib.load(file_path)
        print(f"Model {model_name} loaded from: {file_path}")
        return model
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unknown error loading model: {type(e).__name__}")

def calculate_metrics(y_true, y_pred, y_prob=None):
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, 
                                                          output_dict=True, 
                                                          zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                metrics["fpr"] = fpr
                metrics["tpr"] = tpr
            except ValueError as e:
                print(f" Error calculating ROC-AUC: {e}")
                metrics["roc_auc"] = None
                metrics["fpr"] = None
                metrics["tpr"] = None
            except Exception as e:
                print(f" Unknown error calculating ROC-AUC: {type(e).__name__}")
                metrics["roc_auc"] = None
                metrics["fpr"] = None
                metrics["tpr"] = None
        
        return metrics
        
    except ValueError as e:
        print(f"Value error calculating metrics: {e}")
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": None,
            "classification_report": None
        }

def print_metrics(metrics, model_name):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    try:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if metrics.get("roc_auc") is not None:
            print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
            
        print(f"\nConfusion Matrix:")
        if metrics['confusion_matrix'] is not None:
            print(metrics['confusion_matrix'])
        else:
            print("Not available")
            
    except KeyError as e:
        print(f" Metric {e} not found")
    except Exception as e:
        print(f" Error displaying metrics: {type(e).__name__}")

def ensure_directories():
    from config import MODELS_DIR, RESULTS_DIR, PLOTS_DIR, DATA_DIR
    
    directories = [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, DATA_DIR]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory confirmed: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
    
    return True