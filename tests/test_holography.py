"""
Unit tests for 3D computational holography reconstruction pipeline.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.holography.spatial_scan import SpatialScanner
from src.holography.reconstruction import AngularSpectrumReconstructor

class TestHolography(unittest.TestCase):
    
    def test_spatial_scanner_grid(self):
        scanner = SpatialScanner(grid_size_x=12, grid_size_y=12, spatial_step_mm=0.5)
        data = scanner.acquire_spatial_grid(glucose_mg_dl=130.0, base_seed=42)
        
        self.assertIn('spatial_field', data)
        self.assertEqual(data['spatial_field'].shape[0], 12)
        self.assertEqual(data['spatial_field'].shape[1], 12)
        self.assertEqual(len(data['x_coords_mm']), 12)
        self.assertEqual(len(data['y_coords_mm']), 12)

    def test_angular_spectrum_reconstruction_3d_volume(self):
        scanner = SpatialScanner(grid_size_x=10, grid_size_y=10)
        data = scanner.acquire_spatial_grid(glucose_mg_dl=120.0, base_seed=42)
        
        reconstructor = AngularSpectrumReconstructor(depth_max_mm=2.0, depth_steps=10)
        rec = reconstructor.reconstruct_volume(data)
        
        self.assertIn('volume_3d', rec)
        self.assertIn('slice_xy', rec)
        self.assertIn('slice_xz', rec)
        self.assertIn('slice_yz', rec)
        self.assertIn('mip', rec)
        
        self.assertEqual(rec['volume_3d'].shape, (10, 10, 10))
        self.assertTrue(np.all(rec['volume_3d'] >= 0.0))

if __name__ == '__main__':
    unittest.main()
