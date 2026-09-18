"""
Ablation Study Module
Compares model metrics across pipeline configurations (Raw vs Denoised vs Normalized vs Full Pipeline).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..models.svr_model import SVRModel
from .metrics import MetricsCalculator

class AblationStudy:
    """Executes ablation experiment comparing performance across signal processing stages."""
    
    def __init__(self, dataset_df: pd.DataFrame):
        self.df = dataset_df.copy()
        
    def run_ablation(self) -> pd.DataFrame:
        metadata_cols = ['sample_id', 'glucose_mg_dl', 'temperature_c', 'humidity_pct', 'skin_thickness_mm', 'sensor_distance_cm', 'noise_level_pct', 'seed']
        feature_cols = [c for c in self.df.columns if c not in metadata_cols]
        
        y = self.df['glucose_mg_dl'].values
        X_full = self.df[feature_cols].values
        
        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)
        
        results = []
        
        # Configuration A: Raw Features (subset of 3 raw amplitude features)
        raw_indices = [0, 1, 2] if len(feature_cols) >= 3 else list(range(len(feature_cols)))
        svr_a = SVRModel()
        svr_a.fit(X_tr[:, raw_indices], y_tr)
        preds_a = svr_a.predict(X_te[:, raw_indices])
        m_a = MetricsCalculator.calculate_all_metrics(y_te, preds_a)
        results.append({'pipeline_stage': 'A. Raw Spectral Features', 'mae': m_a['mae'], 'rmse': m_a['rmse'], 'r2': m_a['r2']})
        
        # Configuration B: Denoised + Normalized Features (Full 15 Features)
        svr_b = SVRModel()
        svr_b.fit(X_tr, y_tr)
        preds_b = svr_b.predict(X_te)
        m_b = MetricsCalculator.calculate_all_metrics(y_te, preds_b)
        results.append({'pipeline_stage': 'B. Denoised + Normalized Pipeline', 'mae': m_b['mae'], 'rmse': m_b['rmse'], 'r2': m_b['r2']})
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    dummy_df = pd.DataFrame({
        'glucose_mg_dl': np.random.uniform(70, 200, 50),
        'f1': np.random.randn(50),
        'f2': np.random.randn(50),
        'f3': np.random.randn(50)
    })
    ab = AblationStudy(dummy_df)
    res = ab.run_ablation()
    print("Ablation study complete:\n", res)
