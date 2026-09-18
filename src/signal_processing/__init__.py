"""
Signal processing package initialization.
"""

from .denoising import SignalDenoiser
from .baseline import BaselineCorrector
from .normalization import SignalNormalizer
from .features import FeatureExtractor
from .processor import SignalProcessor

__all__ = [
    'SignalDenoiser',
    'BaselineCorrector',
    'SignalNormalizer',
    'FeatureExtractor',
    'SignalProcessor'
]
