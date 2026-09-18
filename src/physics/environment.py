"""
Environmental Attenuation and Atmospheric Compensation Model
Computes THz attenuation due to ambient water vapor absorption, temperature, and sensor distance.
"""

import numpy as np

class EnvironmentalModel:
    """
    Models atmospheric THz propagation attenuation driven by temperature, humidity, and distance.
    Uses simplified atmospheric water vapor absorption models in the 0.1 - 2.0 THz window.
    """
    
    def __init__(self, temperature_c: float = 25.0, humidity_pct: float = 50.0, distance_cm: float = 5.0):
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.distance_m = float(distance_cm) / 100.0
        
    def calculate_water_vapor_density(self) -> float:
        """
        Calculates water vapor density (g/m³) using Arden Buck equation for saturation vapor pressure.
        """
        T = self.temperature_c
        # Saturation vapor pressure P_sat in hPa
        P_sat = 6.1121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
        # Actual vapor pressure P_v
        P_v = (self.humidity_pct / 100.0) * P_sat
        # Absolute humidity / vapor density rho_v in g/m³
        rho_v = (P_v * 216.7) / (T + 273.15)
        return rho_v

    def get_attenuation_factor(self, frequencies_thz: np.ndarray) -> np.ndarray:
        """
        Calculates frequency-dependent atmospheric attenuation coefficient alpha(f) in dB/m.
        In the 0.1 - 2.0 THz range, water vapor has characteristic resonance peaks (e.g. at 0.557 THz, 0.752 THz, 1.097 THz, 1.163 THz, 1.689 THz).
        """
        rho_v = self.calculate_water_vapor_density()
        
        # Continuum absorption base (scales with f^2 and vapor density)
        alpha_continuum = 0.05 * (frequencies_thz ** 2) * (rho_v / 10.0)  # dB/m
        
        # Resonant absorption line peaks (freq in THz, strength, width in THz)
        water_lines = [
            (0.557, 12.0, 0.03),
            (0.752, 6.0, 0.04),
            (1.097, 18.0, 0.05),
            (1.163, 10.0, 0.05),
            (1.689, 25.0, 0.06)
        ]
        
        alpha_resonance = np.zeros_like(frequencies_thz)
        for center_f, strength, width in water_lines:
            alpha_resonance += strength * (rho_v / 10.0) * (width**2) / ((frequencies_thz - center_f)**2 + width**2)
            
        alpha_total_db_per_m = alpha_continuum + alpha_resonance
        
        # Free-space path loss factor (distance scaling): E_atten = E0 * exp(-alpha_neper * 2*distance)
        # 1 Np = 8.686 dB -> alpha_neper = alpha_db / 8.686
        alpha_neper = alpha_total_db_per_m / 8.686
        
        # Round-trip air path (2 * distance_m)
        total_neper_loss = alpha_neper * (2.0 * self.distance_m)
        transmission_factor = np.exp(-total_neper_loss)
        
        return transmission_factor

    def apply_environment(self, frequencies_thz: np.ndarray, complex_spectrum: np.ndarray) -> np.ndarray:
        """Applies environmental attenuation to a complex THz field."""
        atten_factor = self.get_attenuation_factor(frequencies_thz)
        return complex_spectrum * atten_factor

if __name__ == "__main__":
    env = EnvironmentalModel(temperature_c=30.0, humidity_pct=70.0, distance_cm=10.0)
    freqs = np.linspace(0.1, 2.0, 100)
    atten = env.get_attenuation_factor(freqs)
    print(f"Environmental attenuation factor range at 10 cm: min={np.min(atten):.4f}, max={np.max(atten):.4f}")
