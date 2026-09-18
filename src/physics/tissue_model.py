"""
Multilayer Skin Tissue Model with Debye Dispersion and Glucose Permittivity Perturbation
"""

import numpy as np

class MultilayerTissueModel:
    """
    Models multilayer human skin (Stratum Corneum, Epidermis, Dermis, Subcutaneous tissue)
    using single/double Debye relaxation equations modified by glucose concentration.
    """
    
    def __init__(self, glucose_mg_dl: float = 120.0, skin_thickness_scale: float = 1.0, temperature_c: float = 32.0):
        self.glucose = float(glucose_mg_dl)
        self.thickness_scale = float(skin_thickness_scale)
        self.temperature_c = float(temperature_c)
        
        # Nominal layer parameters (thickness in meters, Debye properties)
        # 1. Stratum Corneum (thin dry layer)
        # 2. Epidermis (low water content)
        # 3. Dermis / Interstitial Fluid (high water content, glucose sensitive)
        # 4. Subcutaneous (fat layer)
        self.layer_specs = {
            'stratum_corneum': {
                'thickness_m': 0.02e-3 * self.thickness_scale,
                'eps_inf': 2.1,
                'delta_eps': 3.5,
                'tau_ps': 0.12,
                'glucose_sensitive': False
            },
            'epidermis': {
                'thickness_m': 0.08e-3 * self.thickness_scale,
                'eps_inf': 2.5,
                'delta_eps': 5.0,
                'tau_ps': 0.15,
                'glucose_sensitive': False
            },
            'dermis': {
                'thickness_m': 1.50e-3 * self.thickness_scale,
                'eps_inf': 3.2,
                'delta_eps': 25.0,
                'tau_ps': 8.5,
                'glucose_sensitive': True
            },
            'subcutaneous': {
                'thickness_m': 2.00e-3 * self.thickness_scale,
                'eps_inf': 2.2,
                'delta_eps': 4.0,
                'tau_ps': 0.20,
                'glucose_sensitive': False
            }
        }
        
    def get_layer_complex_permittivity(self, layer_name: str, frequencies_thz: np.ndarray) -> np.ndarray:
        """
        Calculates complex permittivity ε*(f) = ε'(f) - j*ε''(f) using the Debye model:
        ε*(ω) = ε_inf + Δε / (1 + j*ω*τ)
        Glucose perturbation is applied to the dermis (interstitial fluid/blood layer):
        Higher glucose concentration slightly alters free/bound water ratio,
        modifying Δε and introducing a loss shift.
        """
        if layer_name not in self.layer_specs:
            raise ValueError(f"Unknown tissue layer: {layer_name}")
            
        spec = self.layer_specs[layer_name]
        eps_inf = spec['eps_inf']
        delta_eps = spec['delta_eps']
        tau = spec['tau_ps'] * 1e-12  # Convert picoseconds to seconds
        
        # Temperature effect on relaxation time (approx. -2% per °C deviation from 32°C)
        temp_factor = 1.0 - 0.02 * (self.temperature_c - 32.0)
        tau = tau * temp_factor
        
        # Glucose perturbation on dermis layer
        if spec['glucose_sensitive']:
            # Baseline reference glucose is 100 mg/dL
            # Glucose perturbation: increases bound water, lowering static permittivity slightly, altering absorption
            glucose_diff = self.glucose - 100.0
            delta_eps += glucose_diff * 0.0015
            eps_inf += glucose_diff * 0.0005
            
        omega = 2.0 * np.pi * frequencies_thz * 1e12
        
        # Complex Debye equation: ε*(ω) = ε_inf + Δε / (1 + 1j * ω * τ)
        permittivity = eps_inf + (delta_eps / (1.0 + 1j * omega * tau))
        
        return permittivity

    def get_layer_complex_refractive_index(self, layer_name: str, frequencies_thz: np.ndarray) -> np.ndarray:
        """
        Calculates complex refractive index n*(f) = n(f) - j*k(f) = sqrt(ε*(f)).
        """
        eps_complex = self.get_layer_complex_permittivity(layer_name, frequencies_thz)
        n_complex = np.sqrt(eps_complex)
        # Ensure correct physical branch selection for sqrt (positive real part, positive imag absorption component)
        n = np.real(n_complex)
        k = np.abs(np.imag(n_complex))
        return n - 1j * k

    def get_layer_thickness(self, layer_name: str) -> float:
        """Returns layer thickness in meters."""
        return self.layer_specs[layer_name]['thickness_m']

    def get_ordered_layers() -> list:
        """Returns ordered list of tissue layers from outer to inner."""
        return ['stratum_corneum', 'epidermis', 'dermis', 'subcutaneous']

if __name__ == "__main__":
    tissue = MultilayerTissueModel(glucose_mg_dl=150.0)
    freqs = np.linspace(0.1, 2.0, 5)
    n_dermis = tissue.get_layer_complex_refractive_index('dermis', freqs)
    print("Dermis refractive index at sample frequencies:", n_dermis)