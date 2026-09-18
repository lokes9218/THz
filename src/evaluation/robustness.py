"""
Robustness and Sensitivity Analysis Module
Evaluates model performance degradation under variation of noise, temperature, humidity, and distance.
"""

import numpy as np
import pandas as pd
from ..physics.spectrum_generator import SpectrumGenerator
from ..signal_processing.processor import SignalProcessor

class RobustnessEvaluator:
    """Runs systematic parameter sweep experiments to measure pipeline stability."""
    
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.processor = SignalProcessor()
        
    def evaluate_noise_sensitivity(self, noise_levels=[0.0, 1.0, 2.0, 3.0, 5.0], num_trials=50) -> pd.DataFrame:
        """Evaluates MAE degradation across noise levels."""
        results = []
        for n in noise_levels:
            maes = []
            for i in range(num_trials):
                g = float(np.random.uniform(70, 200))
                sim = SpectrumGenerator(glucose_mg_dl=g, noise_level_pct=n, seed=100+i)
                res = sim.generate_full_simulation()
                proc = self.processor.process_spectrum(res['frequencies_thz'], res['measured_noisy_amplitude'], res['measured_noisy_phase'])
                
                feat_vec = np.array([[proc['features'][k] for k in self.feature_names]])
                pred = float(self.model.predict(feat_vec)[0])
                maes.append(abs(pred - g))
                
            results.append({
                'noise_level_pct': n,
                'mean_mae': float(np.mean(maes)),
                'std_mae': float(np.std(maes))
            })
        return pd.DataFrame(results)

    def evaluate_temperature_sensitivity(self, temperatures=[20.0, 25.0, 30.0, 35.0, 40.0], num_trials=50) -> pd.DataFrame:
        """Evaluates MAE under skin temperature variation."""
        results = []
        for t in temperatures:
            maes = []
            for i in range(num_trials):
                g = float(np.random.uniform(70, 200))
                sim = SpectrumGenerator(glucose_mg_dl=g, temperature_c=t, seed=200+i)
                res = sim.generate_full_simulation()
                proc = self.processor.process_spectrum(res['frequencies_thz'], res['measured_noisy_amplitude'], res['measured_noisy_phase'])
                
                feat_vec = np.array([[proc['features'][k] for k in self.feature_names]])
                pred = float(self.model.predict(feat_vec)[0])
                maes.append(abs(pred - g))
                
            results.append({
                'temperature_c': t,
                'mean_mae': float(np.mean(maes)),
                'std_mae': float(np.std(maes))
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    from ..models.svr_model import SVRModel
    dummy_svr = SVRModel()
    dummy_svr.is_fitted = True
    dummy_svr.predict = lambda x: np.array([120.0] * len(x))
    evaluator = RobustnessEvaluator(dummy_svr, ['peak_amplitude'])
    df_noise = evaluator.evaluate_noise_sensitivity(noise_levels=[0.5, 1.5], num_trials=5)
    print("Robustness noise sweep complete:\n", df_noise)
