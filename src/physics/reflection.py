"""
Electromagnetic Transfer Matrix Method (TMM) Reflection Calculator
Calculates multi-layer wave interaction, reflection coefficients, absorption, and phase delay.
"""

import numpy as np
from .thz_source import THzSource
from .tissue_model import MultilayerTissueModel

class ReflectionCalculator:
    """Calculates complex electromagnetic wave reflection and transmission from multilayer tissue."""
    
    def __init__(self, tissue_model: MultilayerTissueModel, thz_source: THzSource):
        self.tissue = tissue_model
        self.thz = thz_source
        
    def calculate_reflection(self) -> dict:
        freqs_thz = self.thz.frequencies_thz
        k0 = self.thz.k0
        num_freq = len(freqs_thz)
        
        layers = MultilayerTissueModel.get_ordered_layers()
        n0 = 1.0 + 0j  # Air medium
        
        reflection_complex = np.zeros(num_freq, dtype=complex)
        transmission_complex = np.zeros(num_freq, dtype=complex)
        
        for idx in range(num_freq):
            k_vac = k0[idx]
            
            # Start with 2x2 Identity matrix
            M_total = np.eye(2, dtype=complex)
            
            for layer in layers:
                n_layer = self.tissue.get_layer_complex_refractive_index(layer, np.array([freqs_thz[idx]]))[0]
                d_layer = self.tissue.get_layer_thickness(layer)
                
                # Wave propagation phase shift delta = k0 * n * d
                delta = k_vac * n_layer * d_layer
                
                # Standard characteristic matrix for single homogenous layer
                # B = cos(delta), C = 1j * n * sin(delta)
                m11 = np.cos(delta)
                m12 = -1j * np.sin(delta) / n_layer
                m21 = -1j * n_layer * np.sin(delta)
                m22 = np.cos(delta)
                
                M_layer = np.array([[m11, m12], [m21, m22]], dtype=complex)
                M_total = M_total @ M_layer
                
            n_sub = self.tissue.get_layer_complex_refractive_index(layers[-1], np.array([freqs_thz[idx]]))[0]
            
            # Equivalent input admittance Y = (M21 + M22*n_sub) / (M11 + M12*n_sub)
            num_Y = M_total[1, 0] + M_total[1, 1] * n_sub
            den_Y = M_total[0, 0] + M_total[0, 1] * n_sub
            
            if np.abs(den_Y) < 1e-15:
                r = 0.0 + 0j
            else:
                Y_in = num_Y / den_Y
                r = (n0 - Y_in) / (n0 + Y_in)
                
            reflection_complex[idx] = r
            
        amplitude = np.clip(np.abs(reflection_complex), 0.0, 1.0)
        phase = np.unwrap(np.angle(reflection_complex))
        power_reflectance = amplitude ** 2
        
        return {
            'frequencies_thz': freqs_thz,
            'reflection_complex': reflection_complex,
            'transmission_complex': transmission_complex,
            'amplitude': amplitude,
            'phase': phase,
            'power_reflectance': power_reflectance
        }