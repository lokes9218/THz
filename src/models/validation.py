"""
Model Validation and Performance Metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model_Validator:
    """Validate model performance"""
    
    def __init__(self, y_true, y_pred):
        """Initialize validator"""
        self.y_true = y_true
        self.y_pred = y_pred
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)
        mape = np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def plot_prediction_vs_reference(self):
        """Plot predicted vs reference glucose"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_true, self.y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Reference Glucose (mg/dL)', fontsize=12)
        plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)
        plt.title('Glucose Prediction Performance', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self):
        """Plot error distribution"""
        errors = self.y_pred - self.y_true
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (mg/dL)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Glucose Prediction Error Distribution', fontsize=14)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label='Mean Error')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def print_report(self):
        """Print comprehensive validation report"""
        metrics = self.calculate_metrics()
        
        print("=" * 60)
        print("MODEL VALIDATION REPORT")
        print("=" * 60)
        print(f"\nMean Absolute Error (MAE):     {metrics['MAE']:.2f} mg/dL")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f} mg/dL")
        print(f"R² Score:                       {metrics['R2']:.4f}")
        print(f"Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")
        print(f"\nSample count: {len(self.y_true)}")
        print(f"Prediction range: {self.y_pred.min():.1f} - {self.y_pred.max():.1f} mg/dL")
        print(f"Reference range:  {self.y_true.min():.1f} - {self.y_true.max():.1f} mg/dL")
        print("=" * 60)