"""
Unit tests for physics engine modules using standard library unittest.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics.thz_source import THzSource
from src.physics.tissue_model import MultilayerTissueModel
from src.physics.reflection import ReflectionCalculator
from src.physics.environment import EnvironmentalModel
from src.physics.receiver import THzReceiver
from src.physics.spectrum_generator import SpectrumGenerator

class TestPhysicsEngine(unittest.TestCase):
    
    def test_thz_source_generation(self):
        source = THzSource(start_freq_thz=0.1, end_freq_thz=2.0, num_points=100)
        spec = source.generate_incident_spectrum()
        self.assertEqual(len(spec['frequencies_thz']), 100)
        self.assertEqual(spec['frequencies_thz'][0], 0.1)
        self.assertEqual(spec['frequencies_thz'][-1], 2.0)
        self.assertEqual(len(spec['complex_field']), 100)

    def test_multilayer_tissue_model_glucose_sensitivity(self):
        tissue_low = MultilayerTissueModel(glucose_mg_dl=70.0)
        tissue_high = MultilayerTissueModel(glucose_mg_dl=200.0)
        
        freqs = np.array([0.5, 1.0, 1.5])
        eps_dermis_low = tissue_low.get_layer_complex_permittivity('dermis', freqs)
        eps_dermis_high = tissue_high.get_layer_complex_permittivity('dermis', freqs)
        
        self.assertFalse(np.array_equal(eps_dermis_low, eps_dermis_high))
        self.assertGreater(np.real(eps_dermis_high[0]), np.real(eps_dermis_low[0]))

    def test_reflection_calculator_physics(self):
        source = THzSource(start_freq_thz=0.1, end_freq_thz=2.0, num_points=50)
        tissue = MultilayerTissueModel(glucose_mg_dl=120.0)
        calc = ReflectionCalculator(tissue, source)
        
        res = calc.calculate_reflection()
        self.assertIn('amplitude', res)
        self.assertIn('phase', res)
        self.assertEqual(len(res['amplitude']), 50)
        self.assertTrue(np.all(res['amplitude'] >= 0.0) and np.all(res['amplitude'] <= 1.0))

    def test_environmental_model(self):
        env = EnvironmentalModel(temperature_c=25.0, humidity_pct=50.0, distance_cm=5.0)
        freqs = np.linspace(0.1, 2.0, 50)
        atten = env.get_attenuation_factor(freqs)
        self.assertTrue(np.all(atten > 0.0) and np.all(atten <= 1.0))

    def test_receiver_noise_reproducibility(self):
        rx1 = THzReceiver(noise_level_pct=2.0, seed=42)
        rx2 = THzReceiver(noise_level_pct=2.0, seed=42)
        
        freqs = np.linspace(0.1, 2.0, 50)
        clean = np.ones(50, dtype=complex) * 0.5
        
        res1 = rx1.acquire(freqs, clean)
        res2 = rx2.acquire(freqs, clean)
        
        self.assertTrue(np.allclose(res1['amplitude_noisy'], res2['amplitude_noisy']))

    def test_full_spectrum_generator(self):
        sim = SpectrumGenerator(glucose_mg_dl=130.0, seed=42)
        out = sim.generate_full_simulation()
        self.assertIn('measured_noisy_amplitude', out)
        self.assertIn('measured_noisy_phase', out)
        self.assertEqual(len(out['measured_noisy_amplitude']), 200)

if __name__ == '__main__':
    unittest.main()
