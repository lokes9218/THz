"""
Unit tests for signal processing pipeline (denoising, baseline, normalization, features, master processor).
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signal_processing.denoising import SignalDenoiser
from src.signal_processing.baseline import BaselineCorrector
from src.signal_processing.normalization import SignalNormalizer
from src.signal_processing.features import FeatureExtractor
from src.signal_processing.processor import SignalProcessor

class TestSignalProcessing(unittest.TestCase):
    
    def test_denoiser_output(self):
        denoiser = SignalDenoiser(method='savgol', savgol_window=15, savgol_poly=2)
        np.random.seed(42)
        f = np.linspace(0.1, 2.0, 100)
        noisy = np.sin(f) + np.random.normal(0, 0.1, 100)
        
        denoised = denoiser.denoise_amplitude(noisy)
        self.assertEqual(len(denoised), 100)
        self.assertTrue(np.all(denoised >= 0.0))

    def test_baseline_corrector(self):
        corrector = BaselineCorrector(poly_order=2)
        f = np.linspace(0.1, 2.0, 100)
        quadratic_drift = 0.5 * f**2
        signal = quadratic_drift + 0.1 * np.exp(-f)
        
        res = corrector.correct_baseline(f, signal)
        self.assertIn('baseline', res)
        self.assertIn('corrected', res)
        self.assertEqual(len(res['corrected']), 100)

    def test_normalization_minmax(self):
        normalizer = SignalNormalizer(method='minmax')
        vec = np.array([2.5, 7.1, 15.3, -3.0, 8.8])
        norm = normalizer.normalize(vec)
        self.assertAlmostEqual(np.min(norm), 0.0)
        self.assertAlmostEqual(np.max(norm), 1.0)

    def test_feature_extractor_dictionary(self):
        fe = FeatureExtractor()
        f = np.linspace(0.1, 2.0, 100)
        amp = np.sin(f) + 1.5
        phase = 0.5 * f
        
        feats = fe.extract_features(f, amp, phase)
        self.assertEqual(len(feats), 15)
        self.assertIn('peak_amplitude', feats)
        self.assertIn('spectral_area', feats)
        self.assertIn('spectral_entropy', feats)

    def test_master_signal_processor(self):
        sp = SignalProcessor()
        f = np.linspace(0.1, 2.0, 100)
        amp = np.sin(f) + 1.0 + np.random.normal(0, 0.05, 100)
        phase = 0.2 * f
        
        res = sp.process_spectrum(f, amp, phase)
        self.assertIn('normalized_amplitude', res)
        self.assertIn('features', res)
        self.assertEqual(len(res['normalized_amplitude']), 100)

if __name__ == '__main__':
    unittest.main()
