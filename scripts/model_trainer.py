from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from config import MODELS_CONFIG, RANDOM_STATE
from utils import save_model, calculate_metrics, print_metrics
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
    def initialize_models(self, models_to_use=None):
        print("Initializing models...")
        
        if models_to_use is None:
            models_to_use = list(MODELS_CONFIG.keys())
        
        model_classes = {
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GaussianNB": GaussianNB,
            "Perceptron": Perceptron,
            "KNeighborsClassifier": KNeighborsClassifier
        }
        
        for model_name in models_to_use:
            if model_name in MODELS_CONFIG:
                config = MODELS_CONFIG[model_name]
                model_class = model_classes[config["class"]]
                model = model_class(**config["params"])
                self.models[model_name] = model
                print(f"   {model_name} added")
        
        print(f"Number of models: {len(self.models)}")
        return self.models
    
    def train_all_models(self, X_train, y_train, X_test, y_test, save_models=True):
        print("\nStarting model training...")
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                    except (AttributeError, IndexError, ValueError) as e:
                        print(f"Error calculating probability for {name}: {e}")
                        y_prob = None
                
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                
                self.results[name] = {
                    "model": model,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "metrics": metrics
                }
                
                print_metrics(metrics, name)
                
                if save_models:
                    save_model(model, name)
                    
            except ValueError as e:
                print(f"Value error in training {name}: {e}")
                continue
            except RuntimeError as e:
                print(f"Runtime error in training {name}: {e}")
                continue
            except Exception as e:
                print(f"Unknown error in training {name}: {type(e).__name__}: {e}")
                continue
        
        self._find_best_model()
        
        return self.results
    
    def _find_best_model(self):
        if not self.results:
            print(" No results available for comparison")
            return None
        
        try:
            best_name = max(self.results.keys(), 
                           key=lambda k: self.results[k]["metrics"]["accuracy"])
            
            self.best_model_name = best_name
            self.best_model = self.results[best_name]["model"]
            
            print(f"\nBest model: {best_name}")
            print(f"   Accuracy: {self.results[best_name]['metrics']['accuracy']:.4f}")
            
            return best_name
            
        except (KeyError, ValueError) as e:
            print(f" Error finding best model: {e}")
            return None
    
    def get_best_model(self):
        if self.best_model is None:
            self._find_best_model()
        return self.best_model, self.best_model_name
    
    def train_specific_model(self, model_name, X_train, y_train, X_test, y_test):
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        print(f"Training model {model_name}...")
        model = self.models[model_name]
        
        try:
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except (AttributeError, IndexError, ValueError) as e:
                    print(f"Error calculating probability: {e}")
                    y_prob = None
            
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            self.results[model_name] = {
                "model": model,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "metrics": metrics
            }
            
            print_metrics(metrics, model_name)
            save_model(model, model_name)
            
            return metrics
            
        except ValueError as e:
            print(f"Value error in training {model_name}: {e}")
            return None
        except Exception as e:
            print(f"Unknown error in training {model_name}: {type(e).__name__}: {e}")
            return None
    
    def clone_model(self, model):
        return clone(model)