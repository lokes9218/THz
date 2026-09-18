"""
THz Source Engine
Generates deterministic THz frequency grids, incident spectral power distribution, and wave parameters.
"""

import numpy as np

class THzSource:
    """Configurable physical THz radiation source model (0.1 - 10.0 THz)."""
    
    def __init__(self, start_freq_thz: float = 0.1, end_freq_thz: float = 2.0, num_points: int = 200, seed: int = 42):
        self.start_freq = float(start_freq_thz)
        self.end_freq = float(end_freq_thz)
        self.num_points = int(num_points)
        self.seed = seed
        self.c = 299792458.0  # Speed of light in m/s
        
        # Frequency vector in THz and Hz
        self.frequencies_thz = np.linspace(self.start_freq, self.end_freq, self.num_points)
        self.frequencies_hz = self.frequencies_thz * 1e12
        self.angular_freq = 2 * np.pi * self.frequencies_hz
        
        # Wavelengths in micrometers (μm)
        self.wavelengths_um = (self.c / self.frequencies_hz) * 1e6
        
        # Free-space wave numbers k0 (rad/m)
        self.k0 = self.angular_freq / self.c
        
    def generate_incident_spectrum(self, amplitude: float = 1.0, phase_shift: float = 0.0) -> dict:
        """
        Generates incident THz electric field pulse spectral envelope E_inc(f).
        Uses a Gaussian-broadband spectral envelope centered at mid-band.
        """
        center_freq = (self.start_freq + self.end_freq) / 2.0
        bandwidth = (self.end_freq - self.start_freq) / 3.0
        
        # Gaussian spectral envelope
        spectral_envelope = amplitude * np.exp(-0.5 * ((self.frequencies_thz - center_freq) / bandwidth) ** 2)
        phase = np.full(self.num_points, phase_shift, dtype=float)
        
        complex_field = spectral_envelope * np.exp(1j * phase)
        
        return {
            'frequencies_thz': self.frequencies_thz,
            'complex_field': complex_field,
            'amplitude': np.abs(complex_field),
            'phase': phase,
            'wavelengths_um': self.wavelengths_um
        }

if __name__ == "__main__":
    source = THzSource()
    spec = source.generate_incident_spectrum()
    print(f"THzSource initialized: {len(spec['frequencies_thz'])} points from {source.start_freq} to {source.end_freq} THz.")