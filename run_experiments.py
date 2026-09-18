"""
Standalone Experiments & Ablation Execution CLI Script
Executes noise sensitivity, temperature sensitivity, ablation studies, and generates research report.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.svr_model import SVRModel
from src.evaluation.robustness import RobustnessEvaluator
from src.evaluation.ablation import AblationStudy
from src.utils.report_generator import ReportGenerator

def main():
    print("=== THz Glucose Prototype: Experiments & Ablation Suite ===")
    
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'thz_glucose_dataset.csv')
    if not os.path.exists(csv_path):
        print("Dataset not found. Run generate_dataset.py first.")
        return
        
    df = pd.read_csv(csv_path)
    metadata_cols = ['sample_id', 'glucose_mg_dl', 'temperature_c', 'humidity_pct', 'skin_thickness_mm', 'sensor_distance_cm', 'noise_level_pct', 'seed']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Load trained model
    svr_path = os.path.join(os.path.dirname(__file__), 'models', 'svr_model.pkl')
    if not os.path.exists(svr_path):
        print("Model not found. Run train_models.py first.")
        return
        
    svr = SVRModel()
    svr.load(svr_path)
    
    exp_dir = os.path.join(os.path.dirname(__file__), 'results', 'experiments')
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. Robustness Experiments
    print("\n--- Running Noise Sensitivity Experiment ---")
    evaluator = RobustnessEvaluator(svr, feature_cols)
    df_noise = evaluator.evaluate_noise_sensitivity(noise_levels=[0.0, 1.0, 2.0, 3.0, 5.0], num_trials=30)
    df_noise.to_csv(os.path.join(exp_dir, 'noise_robustness.csv'), index=False)
    print("Noise Sensitivity Results:\n", df_noise)
    
    print("\n--- Running Temperature Sensitivity Experiment ---")
    df_temp = evaluator.evaluate_temperature_sensitivity(temperatures=[20.0, 25.0, 30.0, 35.0, 40.0], num_trials=30)
    df_temp.to_csv(os.path.join(exp_dir, 'temperature_robustness.csv'), index=False)
    print("Temperature Sensitivity Results:\n", df_temp)
    
    # 2. Ablation Study
    print("\n--- Running Pipeline Ablation Study ---")
    ablation = AblationStudy(df)
    df_ablation = ablation.run_ablation()
    df_ablation.to_csv(os.path.join(exp_dir, 'ablation_study.csv'), index=False)
    print("Ablation Study Results:\n", df_ablation)
    
    # 3. HTML Report Generation
    print("\n--- Generating Experiment Report ---")
    reports_dir = os.path.join(os.path.dirname(__file__), 'results', 'reports')
    metrics_path = os.path.join(os.path.dirname(__file__), 'results', 'metrics', 'metrics.json')
    rg = ReportGenerator(reports_dir)
    report_file = rg.generate_html_report(metrics_path)
    print(f"Report generated successfully at {report_file}")

if __name__ == "__main__":
    main()
