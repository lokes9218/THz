"""
Physics package initialization
Exposes physical THz modeling classes.
"""

from .thz_source import THzSource
from .tissue_model import MultilayerTissueModel
from .reflection import ReflectionCalculator
from .environment import EnvironmentalModel
from .receiver import THzReceiver
from .spectrum_generator import SpectrumGenerator

__all__ = [
    "THzSource",
    "MultilayerTissueModel",
    "ReflectionCalculator",
    "EnvironmentalModel",
    "THzReceiver",
    "SpectrumGenerator"
]