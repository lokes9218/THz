"""
3D Computational Holography Reconstruction Engine using Angular Spectrum Method (ASM)
Reconstructs 3D volumetric tissue intensity field U(x,y,z) from 2D spatial complex THz measurements.
"""

import numpy as np

class AngularSpectrumReconstructor:
    """
    Implements physical Angular Spectrum Method (ASM) diffraction propagation equation:
    U(x,y,z) = F^-1 { F{U(x,y,0)} * exp(j * k_z * z) }
    where k_z = sqrt(k0^2 - k_x^2 - k_y^2).
    """
    
    def __init__(self, depth_max_mm: float = 2.0, depth_steps: int = 20):
        self.z_max_mm = depth_max_mm
        self.num_z = depth_steps
        self.z_depths_mm = np.linspace(0.0, self.z_max_mm, self.num_z)
        
    def reconstruct_volume(self, spatial_scan_data: dict) -> dict:
        """
        Reconstructs 3D volumetric intensity voxel tensor [Nx, Ny, Nz].
        """
        complex_field = np.nan_to_num(spatial_scan_data['spatial_field'], nan=0.0, posinf=0.0, neginf=0.0)
        x_coords = spatial_scan_data['x_coords_mm']
        y_coords = spatial_scan_data['y_coords_mm']
        freqs_thz = spatial_scan_data['frequencies_thz']
        
        nx, ny, num_freqs = complex_field.shape
        dx = (x_coords[1] - x_coords[0]) * 1e-3  # convert mm to m
        dy = (y_coords[1] - y_coords[0]) * 1e-3
        
        mid_freq_idx = num_freqs // 2
        f0 = freqs_thz[mid_freq_idx] * 1e12  # Hz
        c = 299792458.0
        wavelength = c / f0
        k0 = 2 * np.pi / wavelength
        
        U_surface = complex_field[:, :, mid_freq_idx]
        
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        kz_sq = k0**2 - KX**2 - KY**2
        kz = np.where(kz_sq >= 0, np.sqrt(kz_sq), 1j * np.sqrt(np.abs(kz_sq)))
        
        A_0 = np.fft.fft2(U_surface)
        volume_3d = np.zeros((nx, ny, self.num_z), dtype=float)
        
        for z_idx, z_mm in enumerate(self.z_depths_mm):
            z_m = z_mm * 1e-3
            H_z = np.exp(1j * kz * z_m)
            A_z = A_0 * H_z
            U_z = np.fft.ifft2(A_z)
            intensity_z = np.abs(U_z) ** 2
            volume_3d[:, :, z_idx] = np.nan_to_num(intensity_z, nan=0.0)
            
        slice_xy = volume_3d[:, :, self.num_z // 2]
        slice_xz = volume_3d[:, ny // 2, :]
        slice_yz = volume_3d[nx // 2, :, :]
        mip = np.max(volume_3d, axis=2)
        
        return {
            'volume_3d': volume_3d,
            'x_coords_mm': x_coords,
            'y_coords_mm': y_coords,
            'z_depths_mm': self.z_depths_mm,
            'slice_xy': slice_xy,
            'slice_xz': slice_xz,
            'slice_yz': slice_yz,
            'mip': mip
        }

if __name__ == "__main__":
    from spatial_scan import SpatialScanner
    scanner = SpatialScanner(16, 16)
    data = scanner.acquire_spatial_grid(120.0)
    reconstructor = AngularSpectrumReconstructor(depth_max_mm=2.0, depth_steps=16)
    vol = reconstructor.reconstruct_volume(data)
    print("3D Hologram Volume reconstructed. Shape:", vol['volume_3d'].shape)
