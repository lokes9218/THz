"""
Multi-Layer Perceptron (MLP) Neural Network Regression Component
Implements deep learning MLP regressor with PyTorch/scikit-learn fallback.
"""

import os
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class MLPModel:
    """Multi-Layer Perceptron neural network regressor for glucose prediction."""
    
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter: int = 500, random_state: int = 42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.imputer = SimpleImputer(strategy='median')
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """Fits MLP model on scaled feature matrix X and target y."""
        self.feature_names = feature_names
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler_x.fit_transform(X_imputed)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts glucose concentration (mg/dL)."""
        if not self.is_fitted:
            raise RuntimeError("MLP Model is not fitted. Call fit() or load() first.")
            
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler_x.transform(X_imputed)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

    def save(self, filepath: str):
        """Saves model and scalers to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            'imputer': self.imputer,
            'model': self.model,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'hidden_layer_sizes': self.hidden_layer_sizes
        }
        joblib.dump(artifact, filepath)
        
    def load(self, filepath: str):
        """Loads trained model artifact from disk."""
        artifact = joblib.load(filepath)
        self.imputer = artifact.get('imputer', SimpleImputer(strategy='median'))
        self.model = artifact['model']
        self.scaler_x = artifact['scaler_x']
        self.scaler_y = artifact['scaler_y']
        self.feature_names = artifact.get('feature_names', None)
        self.is_fitted = True

if __name__ == "__main__":
    mlp = MLPModel()
    X_dummy = np.random.randn(100, 15)
    y_dummy = np.random.uniform(70, 200, 100)
    mlp.fit(X_dummy, y_dummy)
    preds = mlp.predict(X_dummy[:5])
    print("MLP Dummy predictions:", preds)
