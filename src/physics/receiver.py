"""
Virtual THz Receiver Engine with Noise & Instrumental Response Simulation
"""

import numpy as np

class THzReceiver:
    """
    Virtual THz Receiver acquiring reflected THz field, introducing thermal noise,
    shot noise, and detector frequency-bandwidth response.
    """
    
    def __init__(self, noise_level_pct: float = 1.5, bandwidth_limit_thz: float = 2.0, seed: int = None):
        self.noise_level_pct = float(noise_level_pct)
        self.bandwidth_limit = float(bandwidth_limit_thz)
        self.seed = seed
        
    def acquire(self, frequencies_thz: np.ndarray, clean_complex_reflection: np.ndarray) -> dict:
        """
        Acquires clean complex reflection signal and adds deterministic/configurable noise.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        amplitude_clean = np.abs(clean_complex_reflection)
        phase_clean = np.unwrap(np.angle(clean_complex_reflection))
        
        # Detector bandwidth roll-off (low-pass filter response at high THz frequencies)
        detector_gain = 1.0 / (1.0 + (frequencies_thz / self.bandwidth_limit) ** 4)
        
        # Additive Gaussian Noise scaled by noise_level_pct
        sigma_noise = (self.noise_level_pct / 100.0) * np.mean(amplitude_clean)
        
        noise_real = np.random.normal(0, sigma_noise, size=len(frequencies_thz))
        noise_imag = np.random.normal(0, sigma_noise, size=len(frequencies_thz))
        complex_noise = noise_real + 1j * noise_imag
        
        noisy_complex_reflection = (clean_complex_reflection * detector_gain) + complex_noise
        
        amplitude_noisy = np.abs(noisy_complex_reflection)
        phase_noisy = np.unwrap(np.angle(noisy_complex_reflection))
        
        return {
            'frequencies_thz': frequencies_thz,
            'clean_complex': clean_complex_reflection,
            'noisy_complex': noisy_complex_reflection,
            'amplitude_clean': amplitude_clean,
            'phase_clean': phase_clean,
            'amplitude_noisy': amplitude_noisy,
            'phase_noisy': phase_noisy,
            'noise_sigma': sigma_noise
        }

if __name__ == "__main__":
    rx = THzReceiver(noise_level_pct=2.0, seed=42)
    sample_freq = np.linspace(0.1, 2.0, 50)
    sample_field = np.ones(50, dtype=complex) * 0.5
    res = rx.acquire(sample_freq, sample_field)
    print("Receiver acquisition complete. Noisy amplitude mean:", np.mean(res['amplitude_noisy']))
