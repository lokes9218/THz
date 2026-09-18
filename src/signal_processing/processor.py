"""
Master Signal Processor
Chains Denoising, Baseline Correction, Normalization, Feature Extraction, and PCA Dimensionality Reduction.
"""

import numpy as np
from sklearn.decomposition import PCA
from .denoising import SignalDenoiser
from .baseline import BaselineCorrector
from .normalization import SignalNormalizer
from .features import FeatureExtractor

class SignalProcessor:
    """Master pipeline executing raw -> noisy -> denoised -> baseline corrected -> normalized -> feature extraction -> PCA."""
    
    def __init__(self, 
                 denoise_method: str = 'savgol', 
                 baseline_order: int = 2, 
                 norm_method: str = 'minmax', 
                 pca_components: int = 5):
        
        self.denoiser = SignalDenoiser(method=denoise_method)
        self.baseline_corrector = BaselineCorrector(poly_order=baseline_order)
        self.normalizer = SignalNormalizer(method=norm_method)
        self.feature_extractor = FeatureExtractor()
        self.pca_components = pca_components
        self.pca_model = None
        
    def process_spectrum(self, frequencies_thz: np.ndarray, noisy_amplitude: np.ndarray, noisy_phase: np.ndarray) -> dict:
        """
        Executes sequential signal processing stages on a single THz spectrum.
        """
        # Step 1: Denoising
        denoised_amp = self.denoiser.denoise_amplitude(noisy_amplitude)
        
        # Step 2: Baseline Correction
        baseline_res = self.baseline_corrector.correct_baseline(frequencies_thz, denoised_amp)
        baseline_amp = baseline_res['baseline']
        corrected_amp = baseline_res['corrected']
        
        # Step 3: Normalization
        normalized_amp = self.normalizer.normalize(corrected_amp)
        
        # Step 4: Feature Extraction
        features = self.feature_extractor.extract_features(frequencies_thz, normalized_amp, noisy_phase)
        
        return {
            'frequencies_thz': frequencies_thz,
            'raw_noisy_amplitude': noisy_amplitude,
            'denoised_amplitude': denoised_amp,
            'baseline_amplitude': baseline_amp,
            'corrected_amplitude': corrected_amp,
            'normalized_amplitude': normalized_amp,
            'phase': noisy_phase,
            'features': features
        }

    def fit_transform_pca(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fits PCA model and transforms feature matrix."""
        n_comp = min(self.pca_components, feature_matrix.shape[1], feature_matrix.shape[0])
        self.pca_model = PCA(n_components=n_comp)
        return self.pca_model.fit_transform(feature_matrix)

if __name__ == "__main__":
    sp = SignalProcessor()
    f = np.linspace(0.1, 2.0, 100)
    sig = 0.5 * np.exp(-f) + np.random.normal(0, 0.05, 100)
    ph = 1.5 * f
    res = sp.process_spectrum(f, sig, ph)
    print("Master processing complete. Features count:", len(res['features']))