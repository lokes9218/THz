"""
Unit tests for 3D computational holography & inverse scattering reconstruction pipeline.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.holography.spatial_scan import SpatialScanner
from src.holography.reconstruction import AngularSpectrumReconstructor
from src.holography.inverse_scattering import LinearizedInverseScattering

class TestHolography(unittest.TestCase):
    
    def test_spatial_scanner_grid(self):
        scanner = SpatialScanner(grid_size_x=12, grid_size_y=12, spatial_step_mm=0.5)
        data = scanner.acquire_spatial_grid(glucose_mg_dl=130.0, base_seed=42)
        
        self.assertIn('spatial_field', data)
        self.assertEqual(data['spatial_field'].shape[0], 12)
        self.assertEqual(data['spatial_field'].shape[1], 12)

    def test_angular_spectrum_reconstruction_3d_volume(self):
        scanner = SpatialScanner(grid_size_x=10, grid_size_y=10)
        data = scanner.acquire_spatial_grid(glucose_mg_dl=120.0, base_seed=42)
        
        reconstructor = AngularSpectrumReconstructor(depth_max_mm=2.0, depth_steps=10)
        rec = reconstructor.reconstruct_volume(data)
        
        self.assertIn('volume_3d', rec)
        self.assertEqual(rec['volume_3d'].shape, (10, 10, 10))

    def test_linearized_inverse_scattering(self):
        scanner = SpatialScanner(grid_size_x=10, grid_size_y=10)
        data = scanner.acquire_spatial_grid(glucose_mg_dl=140.0, base_seed=42)
        
        inv = LinearizedInverseScattering(depth_max_mm=2.0, depth_steps=10)
        res = inv.estimate_dielectric_contrast(data)
        
        self.assertIn('contrast_volume_3d', res)
        self.assertEqual(res['contrast_volume_3d'].shape, (10, 10, 10))
        self.assertEqual(len(res['mean_contrast_by_depth']), 10)

if __name__ == '__main__':
    unittest.main()
