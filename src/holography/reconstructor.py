"""
3D Holographic Reconstruction
Converts 2D THz measurements to 3D volumes
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Holography_Reconstructor:
    """Reconstruct 3D tissue volume from THz data"""
    
    def __init__(self, frequencies, reflection_amplitude, reflection_phase):
        """
        Initialize reconstructor
        
        Parameters:
        - frequencies: THz frequency vector
        - reflection_amplitude: Complex reflection amplitude
        - reflection_phase: Reflection phase
        """
        self.frequencies = frequencies
        self.amplitude = reflection_amplitude
        self.phase = reflection_phase
    
    def virtual_scan(self, x_positions=None, y_positions=None):
        """
        Create virtual scanning aperture
        
        Parameters:
        - x_positions: X coordinates (mm)
        - y_positions: Y coordinates (mm)
        
        Returns:
        - scan_data: 3D array of complex fields
        """
        if x_positions is None:
            x_positions = np.linspace(-5, 5, 21)  # 21 points from -5 to +5 mm
        if y_positions is None:
            y_positions = np.linspace(-5, 5, 21)
        
        nx, ny, nf = len(x_positions), len(y_positions), len(self.frequencies)
        scan_data = np.zeros((nx, ny, nf), dtype=complex)
        
        # For each spatial point, simulate measurement
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                # Gaussian envelope simulating spatial localization
                envelope = np.exp(-(x**2 + y**2) / 10.0)
                # Complex field at this position
                complex_field = self.amplitude * envelope * np.exp(1j * self.phase)
                scan_data[i, j, :] = complex_field
        
        return scan_data, x_positions, y_positions
    
    def back_propagation(self, scan_data, propagation_distance_mm=2.0):
        """
        Numerical back-propagation to reconstruct depth
        
        Parameters:
        - scan_data: 3D scan data
        - propagation_distance_mm: Distance to propagate back
        
        Returns:
        - volume: 3D reconstructed volume
        """
        # Simple depth reconstruction using frequency dependence
        nx, ny, nf = scan_data.shape
        max_freq = self.frequencies[-1]
        
        # Create depth axis
        depths = np.linspace(0, propagation_distance_mm, nf)
        volume = np.zeros((nx, ny, nf))
        
        # For each depth, calculate intensity
        for d_idx, depth in enumerate(depths):
            # Higher frequencies penetrate less → stronger signal at surface
            # Lower frequencies penetrate more → stronger signal at depth
            if d_idx == 0:
                weight = 1.0
            else:
                weight = np.exp(-depth / max_freq)
            
            volume[:, :, d_idx] = np.abs(scan_data[:, :, d_idx]) * weight
        
        # Smooth volume
        volume = gaussian_filter(volume, sigma=0.5)
        
        return volume, depths
    
    def visualize_3d(self, volume, depths, save_path=None):
        """
        3D visualization of reconstructed volume
        
        Parameters:
        - volume: 3D reconstructed data
        - depths: Depth coordinates
        - save_path: Path to save image
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create voxel data
        x = np.arange(volume.shape[0])
        y = np.arange(volume.shape[1])
        z = np.arange(volume.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Voxel colors based on intensity
        colors = plt.cm.hot(volume / np.max(volume))
        
        ax.voxels(X, Y, Z, volume > np.max(volume) * 0.3, colors=colors, alpha=0.3)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Depth (mm)')
        ax.set_title('3D Holographic Reconstruction')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        