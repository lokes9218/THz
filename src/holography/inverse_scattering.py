"""
Inverse Terahertz Scattering Estimator
Implements Rytov/Born linearized inverse scattering approximation to estimate spatial dielectric contrast Delta_eps(x,y,z).
"""

import numpy as np

class LinearizedInverseScattering:
    """Estimates spatial dielectric contrast tensor Delta_eps(x,y,z) from complex THz fields."""
    
    def __init__(self, depth_max_mm: float = 2.0, depth_steps: int = 16):
        self.depth_max_mm = depth_max_mm
        self.depth_steps = depth_steps
        self.z_depths_mm = np.linspace(0.1, self.depth_max_mm, self.depth_steps)
        
    def estimate_dielectric_contrast(self, spatial_scan_data: dict) -> dict:
        """
        Calculates 3D dielectric contrast matrix Delta_eps(x,y,z) using phase-retrieved Rytov formulation.
        """
        complex_field = np.nan_to_num(spatial_scan_data['spatial_field'], nan=0.0, posinf=0.0, neginf=0.0)
        x_coords = spatial_scan_data['x_coords_mm']
        y_coords = spatial_scan_data['y_coords_mm']
        freqs_thz = spatial_scan_data['frequencies_thz']
        
        nx, ny, num_freqs = complex_field.shape
        mid_freq_idx = num_freqs // 2
        f0_hz = freqs_thz[mid_freq_idx] * 1e12
        c = 299792458.0
        k0 = 2 * np.pi * f0_hz / c
        
        # Complex field at central frequency
        field_2d = complex_field[:, :, mid_freq_idx]
        amplitude_2d = np.abs(field_2d)
        phase_2d = np.unwrap(np.angle(field_2d))
        
        # 3D Dielectric contrast volume [nx, ny, num_z]
        contrast_volume = np.zeros((nx, ny, self.depth_steps), dtype=float)
        
        for z_idx, z_mm in enumerate(self.z_depths_mm):
            z_m = z_mm * 1e-3
            # Linearized Rytov contrast scaling: Delta_eps ~ (2 * n0 / (k0 * z)) * phase
            scale_factor = (2.0 / (k0 * z_m + 1e-6))
            contrast_z = scale_factor * phase_2d * (amplitude_2d / (np.max(amplitude_2d) + 1e-12))
            contrast_volume[:, :, z_idx] = np.nan_to_num(contrast_z, nan=0.0)
            
        mean_contrast_by_depth = np.mean(contrast_volume, axis=(0, 1))
        
        return {
            'contrast_volume_3d': contrast_volume,
            'x_coords_mm': x_coords,
            'y_coords_mm': y_coords,
            'z_depths_mm': self.z_depths_mm,
            'mean_contrast_by_depth': mean_contrast_by_depth
        }

if __name__ == "__main__":
    from spatial_scan import SpatialScanner
    scanner = SpatialScanner(12, 12)
    grid = scanner.acquire_spatial_grid(130.0)
    inv = LinearizedInverseScattering(depth_max_mm=2.0, depth_steps=10)
    res = inv.estimate_dielectric_contrast(grid)
    print("Inverse scattering contrast estimated. Volume shape:", res['contrast_volume_3d'].shape)
