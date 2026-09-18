"""
Synthetic Dataset Generator Module
Executes complete forward physics and signal processing pipeline to generate synthetic datasets.
"""

import os
import numpy as np
import pandas as pd
from ..physics.spectrum_generator import SpectrumGenerator
from ..signal_processing.processor import SignalProcessor

class SyntheticDatasetGenerator:
    """Generates synthetic dataset from physics simulation without data leakage."""
    
    def __init__(self, num_samples: int = 2000, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self.signal_processor = SignalProcessor()
        
    def generate(self) -> pd.DataFrame:
        """
        Generates dataset rows:
        Independent variables: glucose, temperature, humidity, skin_thickness, sensor_distance, noise_level
        Extracted features: 15 physics/signal features
        """
        np.random.seed(self.seed)
        
        # Controlled distributions for physiological & environmental parameters
        glucose_vals = np.random.uniform(70.0, 200.0, self.num_samples)
        temp_vals = np.random.uniform(25.0, 35.0, self.num_samples)
        humidity_vals = np.random.uniform(30.0, 80.0, self.num_samples)
        thickness_vals = np.random.uniform(1.2, 2.4, self.num_samples)
        distance_vals = np.random.uniform(3.0, 8.0, self.num_samples)
        noise_vals = np.random.uniform(0.5, 3.0, self.num_samples)
        
        rows = []
        
        print(f"Generating {self.num_samples} physics-simulated dataset samples...")
        
        for idx in range(self.num_samples):
            g = glucose_vals[idx]
            t = temp_vals[idx]
            h = humidity_vals[idx]
            th = thickness_vals[idx]
            d = distance_vals[idx]
            n = noise_vals[idx]
            sample_seed = self.seed + idx
            
            sim = SpectrumGenerator(glucose_mg_dl=g,
                                    temperature_c=t,
                                    humidity_pct=h,
                                    skin_thickness_mm=th,
                                    sensor_distance_cm=d,
                                    noise_level_pct=n,
                                    seed=sample_seed)
            
            res = sim.generate_full_simulation()
            
            # Signal processing
            proc = self.signal_processor.process_spectrum(
                res['frequencies_thz'],
                res['measured_noisy_amplitude'],
                res['measured_noisy_phase']
            )
            
            row = {
                'sample_id': idx,
                'glucose_mg_dl': g,
                'temperature_c': t,
                'humidity_pct': h,
                'skin_thickness_mm': th,
                'sensor_distance_cm': d,
                'noise_level_pct': n,
                'seed': sample_seed
            }
            
            # Add 15 extracted features
            row.update(proc['features'])
            rows.append(row)
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{self.num_samples} samples.")
                
        df = pd.DataFrame(rows)
        return df

if __name__ == "__main__":
    gen = SyntheticDatasetGenerator(num_samples=100, seed=42)
    df_out = gen.generate()
    print("Dataset generated successfully. Shape:", df_out.shape)