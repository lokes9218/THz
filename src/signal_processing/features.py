"""
Feature Extraction Module
Extracts 15 distinct physics-informed spectral and phase features from processed THz data.
"""

import numpy as np

class FeatureExtractor:
    """Extracts scalar feature vectors from 1D THz amplitude and phase spectra."""
    
    def extract_features(self, frequencies_thz: np.ndarray, amplitude: np.ndarray, phase: np.ndarray) -> dict:
        """
        Extracts 15 domain-informed features:
        1. Peak Amplitude
        2. Peak Frequency (THz)
        3. Minimum Amplitude
        4. Mean Amplitude
        5. Amplitude Standard Deviation
        6. Spectral Area (Integral)
        7. Spectral Bandwidth (FWHM)
        8. Phase Mean
        9. Phase Variance
        10. Phase Slope (Delay parameter dphi/df)
        11. Amplitude at 0.5 THz
        12. Amplitude at 1.0 THz
        13. Amplitude at 1.5 THz
        14. Spectral Entropy (Shannon Entropy)
        15. Absorption Power Ratio (Power Integral)
        """
        amp_max = float(np.max(amplitude))
        peak_idx = int(np.argmax(amplitude))
        peak_freq = float(frequencies_thz[peak_idx])
        
        amp_min = float(np.min(amplitude))
        amp_mean = float(np.mean(amplitude))
        amp_std = float(np.std(amplitude))
        
        # Spectral Area (Trapezoidal integration)
        spectral_area = float(np.trapezoid(amplitude, frequencies_thz))
        
        # Spectral Bandwidth (Full Width at Half Maximum - FWHM)
        half_max = amp_max / 2.0
        above_half = np.where(amplitude >= half_max)[0]
        if len(above_half) > 1:
            bandwidth = float(frequencies_thz[above_half[-1]] - frequencies_thz[above_half[0]])
        else:
            bandwidth = 0.0
            
        phase_mean = float(np.mean(phase))
        phase_var = float(np.var(phase))
        
        # Phase Slope via Linear Fit (dphi / df)
        if len(frequencies_thz) > 1:
            phase_slope = float(np.polyfit(frequencies_thz, phase, 1)[0])
        else:
            phase_slope = 0.0
            
        # Frequency band specific amplitudes
        idx_05 = int(np.argmin(np.abs(frequencies_thz - 0.5)))
        idx_10 = int(np.argmin(np.abs(frequencies_thz - 1.0)))
        idx_15 = int(np.argmin(np.abs(frequencies_thz - 1.5)))
        
        amp_0_5thz = float(amplitude[idx_05])
        amp_1_0thz = float(amplitude[idx_10])
        amp_1_5thz = float(amplitude[idx_15])
        
        # Spectral Shannon Entropy
        amp_norm = amplitude / (np.sum(amplitude) + 1e-12)
        amp_norm_pos = amp_norm[amp_norm > 0]
        spectral_entropy = float(-np.sum(amp_norm_pos * np.log2(amp_norm_pos)))
        
        # Absorption Power Ratio (Sum of squared amplitude)
        power_ratio = float(np.sum(amplitude ** 2))
        
        return {
            'peak_amplitude': amp_max,
            'peak_frequency_thz': peak_freq,
            'min_amplitude': amp_min,
            'mean_amplitude': amp_mean,
            'amp_std': amp_std,
            'spectral_area': spectral_area,
            'bandwidth_fwhm': bandwidth,
            'phase_mean': phase_mean,
            'phase_variance': phase_var,
            'phase_slope': phase_slope,
            'amp_0_5thz': amp_0_5thz,
            'amp_1_0thz': amp_1_0thz,
            'amp_1_5thz': amp_1_5thz,
            'spectral_entropy': spectral_entropy,
            'power_ratio': power_ratio
        }

if __name__ == "__main__":
    fe = FeatureExtractor()
    f = np.linspace(0.1, 2.0, 100)
    a = np.exp(-((f - 1.0)**2) / 0.1)
    p = 2.0 * f
    feats = fe.extract_features(f, a, p)
    print("Extracted 15 features successfully. Peak amp:", feats['peak_amplitude'])
