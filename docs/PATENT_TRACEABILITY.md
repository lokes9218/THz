# PATENT TRACEABILITY MATRIX

**Patent Application Title:** Blood Glucose Level Estimation System  
**Patent Application Number:** 202541105574  
**Filing Date:** October 31, 2025  

| Patent Component ID & Title | Description in Patent Specification | Software Implementation Module | Input Data | Output Data | Verification Test | Physical / Simulation Assumption |
|---|---|---|---|---|---|---|
| **Processing Engine 102** | Central orchestration, memory allocation, task execution | `src/core/engine.py` | Config object, seeds | Session execution state | `tests/test_engine.py` | Central Python execution loop. |
| **Terahertz Transmitter Unit 104** | Emits pulsed/CW non-ionizing waves ($0.1–10\text{ THz}$) | `src/physics/thz_source.py` | Freq range, num points, phase | Complex incident field $E_{\text{inc}}(f)$ | `tests/test_thz_source.py` | Configurable 0.1–10.0 THz spectral grid. |
| **Terahertz Receiver Unit 106** | Coherent detection, temporal gating, raw reflection capture | `src/receiver/virtual_receiver.py` | Reflected field $E_{\text{ref}}(f)$, noise | Raw complex spectrum (Ideal & Noisy) | `tests/test_receiver.py` | Additive Gaussian noise & low-pass bandwidth limit. |
| **Holographic Image Reconstruction Unit 108** | Phase-resolved imaging & inverse scattering $\rightarrow$ 3D tissue map | `src/holography/angular_spectrum.py`<br>`src/holography/volume.py` | 2D complex spatial field $H(x,y,f)$ | 3D volumetric intensity field $U(x,y,z)$ | `tests/test_holography.py` | Physical Angular Spectrum Method (ASM) back-propagation. |
| **Deep Learning Implementation Unit 110** | CNN/MLP/SVR regression on holographic & feature representation | `src/ml/svr.py`<br>`src/ml/neural_model.py` | 3D Hologram tensors + 15 features | Predicted glucose ($\text{mg/dL}$) | `tests/test_ml.py` | Trained SVR & MLP regressors with scaler persistence. |
| **Signal Processing & Calibration Unit 112** | Wavelet denoising, baseline correction, SNV/Min-Max normalization | `src/signal/denoising.py`<br>`src/signal/baseline.py`<br>`src/signal/normalization.py` | Noisy measured reflection spectrum | Cleaned, baseline-subtracted spectrum | `tests/test_signal.py` | Savitzky-Golay filtering + DWT VisuShrink soft thresholding. |
| **Environmental Sensors 114** | Monitors ambient temp ($10–40^\circ\text{C}$), humidity, distance | `src/physics/atmospheric_model.py`<br>`src/receiver/noise_model.py` | Temp (°C), Humidity (%), Distance (cm) | Frequency-dependent attenuation $\alpha(f)$ | `tests/test_environment.py` | Arden Buck water vapor pressure + THz line absorption. |
| **User Interface 116** | Visualizes 3D holograms, spectral plots, predictions | `app/app.py` & `app/pages/*.py` | Runtime session state tensors | 15-page interactive research workbench | Manual / E2E verification | Interactive Streamlit application interface. |
| **Power Supply Unit 118** | Power management and task scheduling | `src/core/power_manager.py` | Computational load profile | Power distribution efficiency state | `tests/test_power.py` | Power allocation management simulation. |
