import numpy as np
from skfuzzy.cluster import cmeans
from sklearn.base import clone
from sklearn.decomposition import PCA
from config import FUZZY_N_CLUSTERS, FUZZY_M, FUZZY_ERROR, FUZZY_MAXITER
from utils import calculate_metrics, print_metrics, save_model
import warnings
warnings.filterwarnings('ignore')

class FuzzyEnhancer:
    
    def __init__(self, n_clusters=FUZZY_N_CLUSTERS, m=FUZZY_M, 
                 error=FUZZY_ERROR, maxiter=FUZZY_MAXITER):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.membership_train = None
        self.membership_test = None
        self.fpc_score = None
        
    def _normalize_to_membership(self, data):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        row_sums = normalized.sum(axis=1, keepdims=True)
        return normalized / row_sums
    
    def apply_fuzzy_cmeans(self, X, fallback_method='pca'):
        print("Applying Fuzzy C-Means...")
        
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) is less than number of clusters ({self.n_clusters})"
            )
        
        try:
            cntr, u, u0, d, jm, p, fpc = cmeans(
                data=X.T,
                c=self.n_clusters,
                m=self.m,
                error=self.error,
                maxiter=self.maxiter,
                init=None,
                seed=42
            )
            
            self.fpc_score = fpc
            
            print(f"Fuzzy C-Means executed successfully")
            print(f"   Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
            print(f"   Number of iterations: {p}")
            
            if fpc < 0.3:
                print(" Warning: Clustering quality is low")
            
            return u.T
            
        except ValueError as e:
            print(f"Fuzzy C-Means error: {e}")
            
            if fallback_method == 'pca':
                print("Using PCA as fallback method...")
                return self._apply_pca_fallback(X)
            else:
                raise ValueError(f"Fuzzy C-Means failed: {str(e)[:100]}")
                
        except ImportError as e:
            print(f"scikit-fuzzy library not installed: {e}")
            print("Please run: pip install scikit-fuzzy")
            raise
            
        except Exception as e:
            print(f"Unknown error in Fuzzy C-Means: {type(e).__name__}")
            if fallback_method == 'pca':
                return self._apply_pca_fallback(X)
            else:
                raise RuntimeError(f"Fuzzy C-Means execution error: {str(e)[:100]}")
    
    def _apply_pca_fallback(self, X):
        print("🔧 Running PCA to generate alternative features...")
        
        try:
            pca = PCA(n_components=self.n_clusters, random_state=42)
            X_pca = pca.fit_transform(X)
            
            membership = self._normalize_to_membership(X_pca)
            
            explained_variance = sum(pca.explained_variance_ratio_)
            print(f"PCA executed (preserved variance: {explained_variance:.2%})")
            print(f" Note: Using PCA instead of Fuzzy C-Means")
            
            return membership
            
        except Exception as e:
            raise RuntimeError(f"PCA also failed: {str(e)[:100]}")
    
    def enhance_with_fuzzy_features(self, X_train, X_test, y_train, y_test, base_model):
        print(f"\n{'='*60}")
        print(f"Model Enhancement with Fuzzy Features")
        print(f"{'='*60}")
        
        print("\nCalculating membership degrees...")
        self.membership_train = self.apply_fuzzy_cmeans(X_train)
        self.membership_test = self.apply_fuzzy_cmeans(X_test)
        
        print("\nAdding fuzzy features to data...")
        X_train_enhanced = np.hstack([X_train, self.membership_train])
        X_test_enhanced = np.hstack([X_test, self.membership_test])
        
        print(f"   New training data dimensions: {X_train_enhanced.shape}")
        print(f"   New test data dimensions: {X_test_enhanced.shape}")
        
        print("\nTraining enhanced model...")
        enhanced_model = clone(base_model)
        enhanced_model.fit(X_train_enhanced, y_train)
        
        print("\nEvaluating model...")
        y_pred = enhanced_model.predict(X_test_enhanced)
        
        y_prob = None
        if hasattr(enhanced_model, "predict_proba"):
            try:
                y_prob = enhanced_model.predict_proba(X_test_enhanced)[:, 1]
            except (AttributeError, IndexError) as e:
                print(f" Error calculating probability: {e}")
                y_prob = None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        print(f"Model enhancement completed")
        
        return enhanced_model, metrics
    
    def compare_with_base_model(self, base_model_name, base_accuracy, 
                               enhanced_accuracy, enhanced_metrics):
        print("\n" + "="*60)
        print("Base Model vs Enhanced Model Comparison")
        print("="*60)
        
        print(f"\nBase Model ({base_model_name}):")
        print(f"   Accuracy: {base_accuracy:.4f}")
        
        print(f"\nEnhanced Model (Fuzzy-Enhanced):")
        print(f"   Accuracy: {enhanced_accuracy:.4f}")
        
        improvement = enhanced_accuracy - base_accuracy
        
        print(f"\n{'='*40}")
        if improvement > 0:
            print(f"Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
            print(f"Result: Fuzzy model performed better")
        elif improvement < 0:
            print(f" Reduction: {improvement:.4f} ({abs(improvement)*100:.2f}%)")
            print(f"Result: Base model performed better")
        else:
            print(f"No change")
            print(f"Result: Both models have the same performance")
        print(f"{'='*40}")
        
        print_metrics(enhanced_metrics, "Fuzzy Enhanced Model")
        
        return improvement
    
    def save_enhanced_model(self, model, model_name="fuzzy_enhanced"):
        print(f"\nSaving enhanced model: {model_name}")
        return save_model(model, model_name)
    
    def get_fuzzy_quality_report(self):
        if self.fpc_score is None:
            return "Fuzzy C-Means not executed yet"
        
        report = {
            "fpc_score": self.fpc_score,
            "quality": "Excellent" if self.fpc_score > 0.7 else 
                      "Good" if self.fpc_score > 0.5 else 
                      "Medium" if self.fpc_score > 0.3 else 
                      "Weak",
            "n_clusters": self.n_clusters,
            "fuzziness": self.m
        }
        
        return report