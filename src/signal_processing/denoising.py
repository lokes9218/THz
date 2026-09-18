"""
Signal Denoising Module
Implements Savitzky-Golay filtering and Wavelet Transform Denoising with strict NaN protection.
"""

import numpy as np
from scipy.signal import savgol_filter

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

class SignalDenoiser:
    """Provides noise suppression algorithms for THz spectral measurements."""
    
    def __init__(self, method: str = 'savgol', savgol_window: int = 15, savgol_poly: int = 3, wavelet_name: str = 'db4'):
        self.method = method
        self.window = savgol_window if savgol_window % 2 != 0 else savgol_window + 1
        self.poly = savgol_poly
        self.wavelet = wavelet_name
        
    def denoise_amplitude(self, noisy_amplitude: np.ndarray) -> np.ndarray:
        """
        Denoises 1D spectral amplitude signal using configured algorithm.
        """
        noisy_clean = np.nan_to_num(noisy_amplitude, nan=0.0, posinf=1.0, neginf=0.0)
        
        if len(noisy_clean) < self.window:
            return noisy_clean
            
        if self.method == 'savgol':
            try:
                clean_sig = savgol_filter(noisy_clean, window_length=self.window, polyorder=self.poly)
                return np.nan_to_num(np.clip(clean_sig, 0.0, None), nan=0.0)
            except Exception:
                return noisy_clean
            
        elif self.method == 'wavelet' and HAS_PYWT:
            try:
                coeffs = pywt.wavedec(noisy_clean, self.wavelet, level=2)
                sigma = (1.0 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
                threshold = sigma * np.sqrt(2.0 * np.log(len(noisy_clean)))
                
                denoised_coeffs = [coeffs[0]]
                for c in coeffs[1:]:
                    denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
                    
                clean_sig = pywt.waverec(denoised_coeffs, self.wavelet)[:len(noisy_clean)]
                return np.nan_to_num(np.clip(clean_sig, 0.0, None), nan=0.0)
            except Exception:
                return noisy_clean
        else:
            try:
                clean_sig = savgol_filter(noisy_clean, window_length=min(11, len(noisy_clean)), polyorder=2)
                return np.nan_to_num(np.clip(clean_sig, 0.0, None), nan=0.0)
            except Exception:
                return noisy_clean

if __name__ == "__main__":
    denoiser = SignalDenoiser(method='savgol')
    t = np.linspace(0, 10, 100)
    sig = np.sin(t) + np.random.normal(0, 0.1, 100)
    clean = denoiser.denoise_amplitude(sig)
    print("Denoising complete. Clean signal std:", np.std(clean))
