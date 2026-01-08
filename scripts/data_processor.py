import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE, PCA_VARIANCE
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    
    def __init__(self, file_path=DATA_PATH):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pca = None
        self.imputer = SimpleImputer(strategy='mean')
        self.outliers_removed = 0
        
    def load_data(self):
        print("Loading data...")
        
        try:
            self.data = pd.read_csv(self.file_path)
            
            if 'id' in self.data.columns:
                self.data = self.data.drop('id', axis=1)
                print("   id column removed")
            
            if 'diagnosis' in self.data.columns:
                le = LabelEncoder()
                self.data['diagnosis'] = le.fit_transform(self.data['diagnosis'])
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"   Diagnosis encoded: {mapping}")
            
            print(f"Data loaded. Dimensions: {self.data.shape}")
            return self.data
            
        except FileNotFoundError:
            print(f"Data file not found: {self.file_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self):
        print("Checking for missing values...")
        
        try:
            missing_count = self.data.isnull().sum().sum()
            
            if missing_count > 0:
                print(f"    {missing_count} missing values found")
                
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                self.data[numeric_cols] = self.imputer.fit_transform(self.data[numeric_cols])
                
                print("   Missing values filled with mean")
            else:
                print("   No missing values found")
            
            return self.data
            
        except Exception as e:
            print(f"Error handling missing values: {e}")
            raise
    
    def _calculate_outliers_mask(self, columns):
        mask = pd.Series([True] * len(self.data))
        
        for col in columns:
            try:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_mask = (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
                mask = mask & col_mask
                
            except Exception as e:
                print(f" Error calculating outliers for {col}: {e}")
                continue
        
        return mask
    
    def remove_outliers_iqr(self, columns=None):
        print("Removing outliers (IQR)...")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != 'diagnosis']
        
        try:
            original_len = len(self.data)
            mask = self._calculate_outliers_mask(columns)
            
            self.data = self.data[mask].reset_index(drop=True)
            self.outliers_removed = original_len - len(self.data)
            
            print(f"   {self.outliers_removed} outlier samples removed")
            print(f"   Remaining samples: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            print(f"Error removing outliers: {e}")
            return self.data
    
    def split_data(self, remove_outliers=True, stratify=True):
        print("Splitting data...")
        
        try:
            self.X = self.data.drop('diagnosis', axis=1)
            self.y = self.data['diagnosis']
            
            if remove_outliers:
                self.remove_outliers_iqr()
            
            stratify_col = self.y if stratify else None
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE, 
                stratify=stratify_col
            )
            
            print(f"Data split:")
            print(f"   Training: {self.X_train.shape}")
            print(f"   Test: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except ValueError as e:
            print(f"Value error in data split: {e}")
            raise
        except Exception as e:
            print(f"Unknown error in data split: {type(e).__name__}")
            raise
    
    def scale_data(self):
        print("Scaling data...")
        
        try:
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            print("Data scaled")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            print(f"Error scaling data: {e}")
            raise
    
    def apply_pca(self, X_train_scaled, X_test_scaled, variance=PCA_VARIANCE):
        print(f"Applying PCA (preserving {variance*100}% variance)...")
        
        try:
            self.pca = PCA(n_components=variance, random_state=RANDOM_STATE)
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
            
            explained_variance = sum(self.pca.explained_variance_ratio_)
            n_components = X_train_pca.shape[1]
            
            print(f"PCA applied. New dimensions: {n_components}")
            print(f"   Preserved variance: {explained_variance:.4f}")
            
            return X_train_pca, X_test_pca
            
        except ValueError as e:
            print(f"Value error in PCA: {e}")
            raise
        except Exception as e:
            print(f"Unknown error in PCA: {type(e).__name__}")
            raise
    
    def get_processed_data(self, use_pca=False, remove_outliers=True):
        try:
            self.load_data()
            self.handle_missing_values()
            
            self.split_data(remove_outliers=remove_outliers)
            
            X_train_scaled, X_test_scaled = self.scale_data()
            
            if use_pca:
                X_train_final, X_test_final = self.apply_pca(X_train_scaled, X_test_scaled)
            else:
                X_train_final, X_test_final = X_train_scaled, X_test_scaled
            
            print(f"\nData processing completed:")
            print(f"    PCA: {'Applied' if use_pca else 'Not applied'}")
            print(f"    Outliers removed: {'Applied' if remove_outliers else 'Not applied'}")
            if remove_outliers:
                print(f"    Number of outliers removed: {self.outliers_removed}")
            
            return X_train_final, X_test_final, self.y_train, self.y_test, self.scaler
            
        except Exception as e:
            print(f"Error processing data: {e}")
            raise