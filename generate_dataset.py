"""
Standalone Dataset Generation Script
Executes synthetic dataset generator and saves data/datasets/thz_glucose_dataset.csv with train/val/test splits.
"""

import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.physics.dataset_generator import SyntheticDatasetGenerator

def main():
    print("=== THz Glucose Prototype: Dataset Generator ===")
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_samples = config['dataset'].get('num_samples', 2000)
        seed = config['simulation'].get('seed', 42)
    else:
        num_samples = 2000
        seed = 42
        
    generator = SyntheticDatasetGenerator(num_samples=num_samples, seed=seed)
    df = generator.generate()
    
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'datasets')
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, 'thz_glucose_dataset.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset saved successfully to {csv_path}")
    print(f"  Total Samples: {len(df)}")
    print(f"  Total Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()