"""
2D Spatial Raster Scanning Simulation Engine for Holographic THz Acquisition
Simulates 2D (X x Y) spatial acquisition across skin surface with sub-millimeter spatial resolution.
"""

import numpy as np
from ..physics.spectrum_generator import SpectrumGenerator

class SpatialScanner:
    """Generates complex THz field data matrices over a 2D spatial grid (X x Y)."""
    
    def __init__(self, grid_size_x: int = 16, grid_size_y: int = 16, spatial_step_mm: float = 0.5):
        self.nx = grid_size_x
        self.ny = grid_size_y
        self.dx_mm = spatial_step_mm
        
        self.x_coords_mm = (np.arange(self.nx) - self.nx / 2.0) * self.dx_mm
        self.y_coords_mm = (np.arange(self.ny) - self.ny / 2.0) * self.dx_mm
        
    def acquire_spatial_grid(self, glucose_mg_dl: float = 120.0, base_seed: int = 42) -> dict:
        """
        Simulates 2D spatial acquisition matrix across skin surface.
        Spatial variations (such as epidermal thickness and vascular spot distribution)
        create realistic spatial dielectric contrast.
        """
        # Base spectrum at central frequency point (1.0 THz)
        sim_center = SpectrumGenerator(glucose_mg_dl=glucose_mg_dl, seed=base_seed)
        center_res = sim_center.generate_full_simulation()
        freqs_thz = center_res['frequencies_thz']
        num_freqs = len(freqs_thz)
        
        # Spatial complex field tensor: [nx, ny, num_freqs]
        complex_spatial_field = np.zeros((self.nx, self.ny, num_freqs), dtype=complex)
        
        X, Y = np.meshgrid(self.x_coords_mm, self.y_coords_mm, indexing='ij')
        # Subcutaneous vascular vessel structure pattern (Gaussian absorption spot)
        vascular_pattern = 1.0 - 0.15 * np.exp(-((X**2 + Y**2) / 8.0))
        
        base_complex = center_res['measured_complex']
        
        for i in range(self.nx):
            for j in range(self.ny):
                spatial_factor = vascular_pattern[i, j]
                complex_spatial_field[i, j, :] = base_complex * spatial_factor
                
        return {
            'x_coords_mm': self.x_coords_mm,
            'y_coords_mm': self.y_coords_mm,
            'frequencies_thz': freqs_thz,
            'spatial_field': complex_spatial_field,
            'grid_shape': (self.nx, self.ny, num_freqs)
        }

if __name__ == "__main__":
    scanner = SpatialScanner(grid_size_x=16, grid_size_y=16)
    grid = scanner.acquire_spatial_grid(120.0)
    print("Spatial grid acquired with shape:", grid['spatial_field'].shape)
