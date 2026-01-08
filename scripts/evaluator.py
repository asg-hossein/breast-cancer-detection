import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import RESULTS_DIR
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    
    def __init__(self, results):
        self.results = results
        self.summary_df = None
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Results folder: {RESULTS_DIR}")
    
    def _extract_metric_value(self, metrics, metric_name):
        metric_aliases = {
            'accuracy': ['accuracy', 'Accuracy', 'acc'],
            'precision': ['precision', 'Precision', 'prec'],
            'recall': ['recall', 'Recall', 'rec'],
            'f1_score': ['f1_score', 'f1', 'F1', 'f1-score', 'F1-Score'],
            'roc_auc': ['roc_auc', 'auc', 'AUC', 'AUC-ROC']
        }
        
        if metric_name in metric_aliases:
            for alias in metric_aliases[metric_name]:
                if alias in metrics:
                    try:
                        return float(metrics[alias])
                    except (ValueError, TypeError):
                        return np.nan
        
        return np.nan
    
    def create_summary_dataframe(self):
        summary_data = []
        
        for model_name, result in self.results.items():
            try:
                if isinstance(result, dict):
                    if "metrics" in result:
                        metrics = result["metrics"]
                    elif any(key in result for key in ['accuracy', 'precision', 'recall', 'f1_score']):
                        metrics = result
                    else:
                        metrics = {}
                else:
                    metrics = {}
                
                row = {
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-Score": np.nan,
                    "AUC-ROC": np.nan,
                    "Train_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if metrics:
                    row["Accuracy"] = self._extract_metric_value(metrics, 'accuracy')
                    row["Precision"] = self._extract_metric_value(metrics, 'precision')
                    row["Recall"] = self._extract_metric_value(metrics, 'recall')
                    row["F1-Score"] = self._extract_metric_value(metrics, 'f1_score')
                    row["AUC-ROC"] = self._extract_metric_value(metrics, 'roc_auc')
                
                display_row = row.copy()
                for key in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
                    if pd.isna(display_row[key]):
                        display_row[key] = "N/A"
                    else:
                        display_row[key] = f"{display_row[key]:.4f}"
                
                summary_data.append(display_row)
                print(f"Results for {model_name} added")
                
            except Exception as e:
                error_row = {
                    "Model": model_name,
                    "Accuracy": "ERROR",
                    "Precision": "ERROR",
                    "Recall": "ERROR",
                    "F1-Score": "ERROR",
                    "AUC-ROC": "ERROR",
                    "Train_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Error": str(e)[:50]
                }
                summary_data.append(error_row)
                print(f" Error processing results for {model_name}: {e}")
        
        self.summary_df = pd.DataFrame(summary_data)
        return self.summary_df
    
    def save_results_to_csv(self, filename="classification_results.csv"):
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            self.summary_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"CSV results saved: {filepath}")
            return filepath
        except (IOError, OSError) as e:
            print(f"CSV save error: {e}")
            return None
    
    def save_detailed_results(self, filename="detailed_results.json"):
        detailed_results = {}
        
        for model_name, result in self.results.items():
            try:
                if isinstance(result, dict) and "metrics" in result:
                    metrics = result["metrics"]
                else:
                    metrics = {}
                
                detailed_results[model_name] = {}
                
                metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                for key in metric_keys:
                    value = self._extract_metric_value(metrics, key)
                    if not pd.isna(value):
                        detailed_results[model_name][key] = value
                
                if "confusion_matrix" in metrics:
                    try:
                        detailed_results[model_name]["confusion_matrix"] = (
                            metrics["confusion_matrix"].tolist()
                        )
                    except AttributeError:
                        detailed_results[model_name]["confusion_matrix"] = (
                            metrics["confusion_matrix"]
                        )
                
                if "classification_report" in metrics:
                    detailed_results[model_name]["classification_report"] = (
                        metrics["classification_report"]
                    )
                    
            except Exception as e:
                detailed_results[model_name] = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            print(f"JSON results saved: {filepath}")
            return filepath
        except (IOError, OSError) as e:
            print(f"JSON save error: {e}")
            return None
    
    def print_comparison_table(self):
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        print("\n" + "="*80)
        print("Model Comparison Table")
        print("="*80)
        
        try:
            print(self.summary_df.to_string(index=False))
        except Exception as e:
            print(f"Error displaying table: {e}")
        
        print("="*80)
    
    def get_best_model_info(self):
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        try:
            valid_models = self.summary_df[
                ~self.summary_df['Accuracy'].isin(['N/A', 'ERROR'])
            ].copy()
            
            if len(valid_models) == 0:
                print("⚠️ No valid models found")
                return None
            
            def to_float(x):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 0.0
            
            valid_models['Accuracy_num'] = valid_models['Accuracy'].apply(to_float)
            
            best_idx = valid_models['Accuracy_num'].idxmax()
            
            return {
                "model_name": valid_models.loc[best_idx, 'Model'],
                "accuracy": valid_models.loc[best_idx, 'Accuracy'],
                "accuracy_num": valid_models.loc[best_idx, 'Accuracy_num']
            }
            
        except Exception as e:
            print(f" Error finding best model: {e}")
            return None