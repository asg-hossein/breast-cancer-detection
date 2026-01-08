import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import PLOTS_DIR
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

class DataVisualizer:
    
    def __init__(self, data_processor=None):
        self.data_processor = data_processor
        self.figsize_large = (16, 12)
        self.figsize_medium = (12, 8)
        self.figsize_small = (8, 6)
        
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"Plots folder: {PLOTS_DIR}")
    
    def plot_correlation_heatmap(self, data, figsize=None, save=True):
        if figsize is None:
            figsize = self.figsize_large
        
        try:
            corr_matrix = data.corr()
            
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Correlation plot saved: {filepath}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating correlation plot: {e}")
            plt.close()
    
    def plot_feature_histograms(self, X_scaled, feature_names, figsize=None, save=True):
        if figsize is None:
            num_features = len(feature_names)
            if num_features <= 10:
                figsize = (15, 10)
            elif num_features <= 20:
                figsize = (18, 14)
            else:
                figsize = (20, 25)
        
        try:
            scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
            num_features = len(feature_names)
            
            max_cols = 5
            cols = min(max_cols, num_features)
            rows = (num_features + cols - 1) // cols
            
            if rows > 1:
                figsize = (figsize[0], min(figsize[1], 5 * rows))
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            
            if num_features == 1:
                axes = np.array([axes])
            
            axes = axes.flatten()
            
            for i, (col, ax) in enumerate(zip(scaled_df.columns, axes)):
                try:
                    sns.histplot(scaled_df[col], bins=30, kde=True, 
                                color='skyblue', ax=ax)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error in {col}', 
                           ha='center', va='center', fontsize=10)
                    ax.set_title(col + " (Error)", fontsize=10)
            
            for i in range(len(scaled_df.columns), len(axes)):
                axes[i].axis('off')
            
            plt.suptitle('Standardized Feature Histograms (Z-score)', 
                        fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(PLOTS_DIR, "feature_histograms.png")
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"📊 Feature histograms saved: {filepath}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating histograms: {e}")
            plt.close()
    
    def plot_class_distribution(self, y, labels=['Benign (B)', 'Malignant (M)'], save=True):
        plt.figure(figsize=self.figsize_small)
        
        try:
            counts = pd.Series(y).value_counts().sort_index()
            colors = ['lightgreen', 'lightcoral']
            
            bars = plt.bar(range(len(counts)), counts.values, 
                          color=colors, edgecolor='black')
            
            plt.title('Distribution of Benign and Malignant Samples', fontsize=14)
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(range(len(counts)), labels)
            
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{count}', ha='center', va='bottom', fontsize=12)
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(PLOTS_DIR, "class_distribution.png")
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Class distribution plot saved: {filepath}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating class distribution plot: {e}")
            plt.close()


class ModelVisualizer:
    
    def __init__(self, results):
        self.results = results
        self.figsize_large = (16, 12)
        self.figsize_medium = (12, 8)
        self.figsize_small = (8, 6)
        
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"Plots folder: {PLOTS_DIR}")
    
    def plot_roc_curves(self, figsize=None, save=True):
        if figsize is None:
            figsize = self.figsize_medium
        
        plt.figure(figsize=figsize)
        
        try:
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
            
            has_valid_roc = False
            
            for model_name, result in self.results.items():
                metrics = result.get("metrics", {})
                if (metrics.get("fpr") is not None and 
                    metrics.get("tpr") is not None):
                    
                    fpr = metrics["fpr"]
                    tpr = metrics["tpr"]
                    auc = metrics.get("roc_auc", 0)
                    
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{model_name} (AUC = {auc:.3f})')
                    has_valid_roc = True
            
            if not has_valid_roc:
                plt.text(0.5, 0.5, 'ROC data not available', 
                        ha='center', va='center', fontsize=12)
                print(" ROC data not available for any model")
            
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves for Different Models', fontsize=16, pad=20)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save and has_valid_roc:
                filepath = os.path.join(PLOTS_DIR, "roc_curves.png")
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"ROC plot saved: {filepath}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating ROC plot: {e}")
            plt.close()
    
    def plot_confusion_matrices(self, save=True):
        labels = ['Benign', 'Malignant']
        
        for model_name, result in self.results.items():
            try:
                cm = result.get("metrics", {}).get("confusion_matrix")
                
                if cm is None:
                    print(f" Confusion matrix for {model_name} not available")
                    continue
                
                plt.figure(figsize=self.figsize_small)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           cbar_kws={'label': 'Number of Samples'})
                
                plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=15)
                plt.xlabel('Prediction', fontsize=12)
                plt.ylabel('True Values', fontsize=12)
                
                plt.tight_layout()
                
                if save:
                    safe_name = (model_name.replace(' ', '_')
                                .replace('/', '_')
                                .replace('\\', '_'))
                    filepath = os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png")
                    plt.savefig(filepath, bbox_inches='tight', dpi=300)
                    print(f"   Confusion matrix {model_name} saved")
                
                plt.show()
                plt.close()
                
            except Exception as e:
                print(f"Error generating confusion matrix {model_name}: {e}")
                plt.close()
    
    def plot_model_comparison(self, figsize=None, save=True):
        if figsize is None:
            figsize = self.figsize_medium
        
        plt.figure(figsize=figsize)
        
        try:
            metrics_data = []
            for model_name, result in self.results.items():
                metrics = result.get("metrics", {})
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0)
                })
            
            df = pd.DataFrame(metrics_data)
            df_melted = df.melt(id_vars=['Model'], 
                               var_name='Metric', 
                               value_name='Value')
            
            sns.barplot(x='Model', y='Value', hue='Metric', 
                       data=df_melted, palette='viridis')
            
            plt.title('Model Performance Comparison', fontsize=16, pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.ylim(0, 1.05)
            plt.legend(title='Metric', title_fontsize=12, 
                      fontsize=10, loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            
            ax = plt.gca()
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', 
                           fontsize=9)
            
            plt.xticks(rotation=15)
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(PLOTS_DIR, "model_comparison.png")
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Model comparison plot saved: {filepath}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating comparison plot: {e}")
            plt.close()
    
    def plot_all_model_charts(self):
        print("\nGenerating model plots...")
        
        try:
            self.plot_roc_curves()
            self.plot_confusion_matrices()
            self.plot_model_comparison()
            
            print("All model plots generated and saved")
            
        except Exception as e:
            print(f"Error generating model plots: {e}")