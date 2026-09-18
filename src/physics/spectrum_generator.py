"""
Integrated Physics Spectrum Generator Component
Connects Virtual Patient, THz Source, Multilayer Tissue, Reflection Calculator, Environment, and Receiver.
"""

import numpy as np
from .thz_source import THzSource
from .tissue_model import MultilayerTissueModel
from .reflection import ReflectionCalculator
from .environment import EnvironmentalModel
from .receiver import THzReceiver

class SpectrumGenerator:
    """Master generator linking the complete forward physics simulation pipeline."""
    
    def __init__(self, 
                 glucose_mg_dl: float = 120.0,
                 temperature_c: float = 32.0,
                 humidity_pct: float = 50.0,
                 skin_thickness_mm: float = 1.8,
                 sensor_distance_cm: float = 5.0,
                 noise_level_pct: float = 1.5,
                 start_freq_thz: float = 0.1,
                 end_freq_thz: float = 2.0,
                 num_points: int = 200,
                 seed: int = 42):
        
        self.glucose = float(glucose_mg_dl)
        self.temperature = float(temperature_c)
        self.humidity = float(humidity_pct)
        self.skin_thickness = float(skin_thickness_mm)
        self.sensor_distance = float(sensor_distance_cm)
        self.noise_level = float(noise_level_pct)
        self.seed = seed
        
        # Instantiate sub-models
        self.source = THzSource(start_freq_thz, end_freq_thz, num_points, seed=seed)
        
        # Scale skin thickness relative to nominal (1.8 mm nominal)
        thickness_scale = self.skin_thickness / 1.8
        self.tissue = MultilayerTissueModel(glucose_mg_dl=self.glucose, 
                                            skin_thickness_scale=thickness_scale, 
                                            temperature_c=self.temperature)
        
        self.reflection_calc = ReflectionCalculator(self.tissue, self.source)
        self.environment = EnvironmentalModel(temperature_c=self.temperature, 
                                              humidity_pct=self.humidity, 
                                              distance_cm=self.sensor_distance)
        self.receiver = THzReceiver(noise_level_pct=self.noise_level, seed=seed)
        
    def generate_full_simulation(self) -> dict:
        """Executes full forward physics pipeline."""
        incident = self.source.generate_incident_spectrum()
        ref_results = self.reflection_calc.calculate_reflection()
        
        # Apply environmental atmospheric attenuation
        clean_reflected_complex = self.environment.apply_environment(
            ref_results['frequencies_thz'], 
            ref_results['reflection_complex']
        )
        
        # Receiver acquisition with detector noise
        rx_results = self.receiver.acquire(ref_results['frequencies_thz'], clean_reflected_complex)
        
        return {
            'patient_parameters': {
                'glucose_mg_dl': self.glucose,
                'temperature_c': self.temperature,
                'humidity_pct': self.humidity,
                'skin_thickness_mm': self.skin_thickness,
                'sensor_distance_cm': self.sensor_distance,
                'noise_level_pct': self.noise_level,
                'seed': self.seed
            },
            'frequencies_thz': self.source.frequencies_thz,
            'incident_amplitude': incident['amplitude'],
            'ideal_reflection_amplitude': ref_results['amplitude'],
            'ideal_reflection_phase': ref_results['phase'],
            'env_attenuated_amplitude': np.abs(clean_reflected_complex),
            'measured_noisy_amplitude': rx_results['amplitude_noisy'],
            'measured_noisy_phase': rx_results['phase_noisy'],
            'measured_complex': rx_results['noisy_complex'],
            'ideal_complex': ref_results['reflection_complex']
        }

if __name__ == "__main__":
    sim = SpectrumGenerator(glucose_mg_dl=140.0)
    out = sim.generate_full_simulation()
    print("Full forward simulation complete. Frequencies:", len(out['frequencies_thz']))