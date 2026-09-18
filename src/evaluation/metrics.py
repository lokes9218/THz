"""
Metrics Evaluation Module
Calculates MAE, RMSE, R2, MAPE, Median Absolute Error, and Maximum Absolute Error dynamically.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MetricsCalculator:
    """Calculates evaluation metrics from real reference vs predicted glucose vectors."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculates standard regression performance metrics."""
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        r2 = float(r2_score(y_t, y_p))
        
        # Mean Absolute Percentage Error (MAPE)
        mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(y_t, 1.0))) * 100.0)
        
        med_ae = float(np.median(np.abs(y_t - y_p)))
        max_ae = float(np.max(np.abs(y_t - y_p)))
        
        residuals = y_p - y_t
        residual_std = float(np.std(residuals))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': med_ae,
            'max_ae': max_ae,
            'residual_std': residual_std,
            'num_samples': len(y_t)
        }

if __name__ == "__main__":
    y_real = np.array([100.0, 120.0, 150.0, 180.0])
    y_hat = np.array([102.0, 118.0, 153.0, 175.0])
    m = MetricsCalculator.calculate_all_metrics(y_real, y_hat)
    print("Metrics calculated dynamically:", m)
