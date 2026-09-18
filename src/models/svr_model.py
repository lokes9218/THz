"""
Support Vector Regression (SVR) Model Component
Trains SVR regressor with RBF kernel and exports model + preprocessing scalers.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class SVRModel:
    """SVR model for non-invasive glucose estimation."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 100.0, gamma: str = 'scale', epsilon: float = 0.1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.imputer = SimpleImputer(strategy='median')
        self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """Fits SVR model on scaled input feature matrix X and target y."""
        self.feature_names = feature_names
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler_x.fit_transform(X_imputed)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts glucose concentration (mg/dL)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() first.")
            
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
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'epsilon': self.epsilon
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
    svr = SVRModel()
    X_dummy = np.random.randn(100, 15)
    y_dummy = np.random.uniform(70, 200, 100)
    svr.fit(X_dummy, y_dummy)
    preds = svr.predict(X_dummy[:5])
    print("SVR Dummy predictions:", preds)