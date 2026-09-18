"""
Standalone Model Training Script
Trains SVR and MLP regressors on generated synthetic dataset without data leakage.
Exports models to models/ and metrics to results/metrics/metrics.json.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.svr_model import SVRModel
from src.models.mlp_model import MLPModel
from src.evaluation.metrics import MetricsCalculator

def main():
    print("=== THz Glucose Prototype: Model Training & Evaluation ===")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'thz_glucose_dataset.csv')
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Run generate_dataset.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} features.")
    
    # Feature columns (exclude non-feature metadata)
    metadata_cols = ['sample_id', 'glucose_mg_dl', 'temperature_c', 'humidity_pct', 'skin_thickness_mm', 'sensor_distance_cm', 'noise_level_pct', 'seed']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['glucose_mg_dl'].values
    
    # Strict Train / Validation / Test split (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42) # 0.1765 of 0.85 = 0.15
    
    print(f"Data Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 1. Train SVR Model
    print("\n--- Training Support Vector Regressor (SVR) ---")
    svr = SVRModel(C=100.0, kernel='rbf')
    svr.fit(X_train, y_train, feature_names=feature_cols)
    
    svr_val_preds = svr.predict(X_val)
    svr_val_metrics = MetricsCalculator.calculate_all_metrics(y_val, svr_val_preds)
    
    svr_test_preds = svr.predict(X_test)
    svr_test_metrics = MetricsCalculator.calculate_all_metrics(y_test, svr_test_preds)
    
    print(f"SVR Test MAE: {svr_test_metrics['mae']:.2f} mg/dL, R2: {svr_test_metrics['r2']:.4f}, RMSE: {svr_test_metrics['rmse']:.2f}")
    
    # 2. Train MLP Model
    print("\n--- Training Multi-Layer Perceptron (MLP) ---")
    mlp = MLPModel(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train, feature_names=feature_cols)
    
    mlp_val_preds = mlp.predict(X_val)
    mlp_val_metrics = MetricsCalculator.calculate_all_metrics(y_val, mlp_val_preds)
    
    mlp_test_preds = mlp.predict(X_test)
    mlp_test_metrics = MetricsCalculator.calculate_all_metrics(y_test, mlp_test_preds)
    
    print(f"MLP Test MAE: {mlp_test_metrics['mae']:.2f} mg/dL, R2: {mlp_test_metrics['r2']:.4f}, RMSE: {mlp_test_metrics['rmse']:.2f}")
    
    # 3. Save Models
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    svr_path = os.path.join(models_dir, 'svr_model.pkl')
    mlp_path = os.path.join(models_dir, 'mlp_model.pkl')
    
    svr.save(svr_path)
    mlp.save(mlp_path)
    print(f"\nModels saved to {models_dir}")
    
    # 4. Save Metrics & Test Predictions
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'metrics')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_payload = {
        'svr': {
            'validation': svr_val_metrics,
            'test': svr_test_metrics
        },
        'mlp': {
            'validation': mlp_val_metrics,
            'test': mlp_test_metrics
        }
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    # Save predictions CSV for plotting
    preds_df = pd.DataFrame({
        'reference_glucose': y_test,
        'svr_predicted': svr_test_preds,
        'mlp_predicted': mlp_test_preds,
        'svr_residual': svr_test_preds - y_test,
        'mlp_residual': mlp_test_preds - y_test
    })

    preds_dir = os.path.join(os.path.dirname(__file__), 'results', 'predictions')
    os.makedirs(preds_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(preds_dir, 'test_predictions.csv'), index=False)

    print(f"Metrics & predictions saved to results/")

if __name__ == "__main__":
    main()
