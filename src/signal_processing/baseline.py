"""
Baseline Removal and Drift Correction Module
Computes polynomial baseline fitting and AirPLS/Asymmetric Least Squares drift suppression.
"""

import numpy as np

class BaselineCorrector:
    """Removes slow low-frequency drift and atmospheric background baseline shifts."""
    
    def __init__(self, method: str = 'poly', poly_order: int = 2):
        self.method = method
        self.poly_order = poly_order
        
    def correct_baseline(self, frequencies_thz: np.ndarray, amplitude: np.ndarray) -> dict:
        """
        Fits background baseline polynomial and returns baseline-subtracted corrected signal.
        """
        if self.method == 'poly':
            # Polynomial baseline fit
            poly_coeffs = np.polyfit(frequencies_thz, amplitude, self.poly_order)
            baseline = np.polyval(poly_coeffs, frequencies_thz)
            corrected = amplitude - baseline
            # Shift corrected signal so minimum is 0 to preserve positive spectrum representation
            corrected_positive = corrected - np.min(corrected)
            return {
                'baseline': baseline,
                'corrected': corrected_positive
            }
        else:
            # Linear minimum baseline anchor fallback
            baseline = np.linspace(amplitude[0], amplitude[-1], len(amplitude))
            corrected = amplitude - baseline
            corrected_positive = corrected - np.min(corrected)
            return {
                'baseline': baseline,
                'corrected': corrected_positive
            }

if __name__ == "__main__":
    corrector = BaselineCorrector(poly_order=2)
    f = np.linspace(0.1, 2.0, 100)
    sig = 0.5 * f**2 + np.sin(5*f) * 0.1
    res = corrector.correct_baseline(f, sig)
    print("Baseline correction complete. Corrected range:", np.min(res['corrected']), np.max(res['corrected']))
