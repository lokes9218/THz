"""
Spectral Normalization and Standardization Module
Implements Min-Max, Standard Normal Variate (SNV), and Vector Area Normalization.
"""

import numpy as np

class SignalNormalizer:
    """Normalizes THz reflection spectra to eliminate system gain variations."""
    
    def __init__(self, method: str = 'minmax'):
        self.method = method
        
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalizes 1D signal vector.
        """
        if len(signal) == 0:
            return signal
            
        if self.method == 'minmax':
            s_min = np.min(signal)
            s_max = np.max(signal)
            if abs(s_max - s_min) < 1e-12:
                return np.zeros_like(signal)
            return (signal - s_min) / (s_max - s_min)
            
        elif self.method == 'snv':
            # Standard Normal Variate: (x - mean) / std
            mean = np.mean(signal)
            std = np.std(signal)
            if std < 1e-12:
                return signal - mean
            return (signal - mean) / std
            
        elif self.method == 'area':
            # Area / L1 unit normalization
            area = np.sum(np.abs(signal))
            if area < 1e-12:
                return signal
            return signal / area
            
        else:
            return signal

if __name__ == "__main__":
    norm = SignalNormalizer('minmax')
    v = np.array([2.0, 5.0, 10.0, 3.0])
    print("Min-max normalized:", norm.normalize(v))
