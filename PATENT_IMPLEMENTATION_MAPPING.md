# PATENT IMPLEMENTATION MAPPING MATRIX

**Patent Application Title:** Blood Glucose Level Estimation System  
**Patent Application Number:** 202541105574  
**Filing Date:** October 31, 2025  

| Patent Component / Claim Number | Description in Patent Specification | Software Implementation Module | Input Data | Output Data | Verification Status | Physical / Simulation Assumption |
|---|---|---|---|---|---|---|
| **Claim 1 (System Architecture)** | Contactless THz sensing engine (0.1–10 THz) for blood glucose estimation. | `src/physics/thz_source.py`<br>`src/physics/spectrum_generator.py` | Frequency range (0.1–2.0 THz), amplitude, phase | Incident complex E-field envelope $E_{\text{inc}}(f)$ | **Implemented & Verified** | Default simulation band 0.1–2.0 THz across 200 frequency points. |
| **Claim 1 (Tissue Interaction)** | EM wave interaction with subcutaneous tissues & interstitial fluids. | `src/physics/tissue_model.py` | Glucose concentration (mg/dL), temperature (°C) | Layer-specific complex permittivity $\varepsilon^*(f)$ & refractive index $n^*(f)$ | **Implemented & Verified** | Single/double Debye relaxation model with linear perturbation on dermis layer. |
| **Claim 1 (Reflection Calculation)** | Wave interaction & multi-layer reflection coefficient calculation. | `src/physics/reflection.py` | $\varepsilon^*(f)$, layer thickness $d$ | Complex reflection coefficient $R(f)$, power reflectance $\|R(f)\|^2$, phase delay | **Implemented & Verified** | 2x2 Transfer Matrix Method (TMM) assuming optically flat tissue layers. |
| **Claim 2 (Denoising Pipeline)** | Wavelet transform filtering & noise suppression. | `src/signal_processing/denoising.py` | Noisy measured reflection amplitude | Denoised spectral signal | **Implemented & Verified** | Savitzky-Golay polynomial filter + VisuShrink Discrete Wavelet Transform (DWT). |
| **Claim 3 (Feature Extraction & PCA)** | Extracting spectral features & PCA dimensionality reduction. | `src/signal_processing/features.py`<br>`src/signal_processing/processor.py` | Processed amplitude & phase spectrum | 15 scalar features + 5 PCA components | **Implemented & Verified** | Features include peak amp, FWHM bandwidth, phase slope $d\phi/df$, and Shannon entropy. |
| **Claim 4 (Tissue Characterization)** | Regression-based MLPs for dielectric constant characterization. | `src/models/mlp_model.py` | Extracted 15-feature matrix | Estimated glucose concentration (mg/dL) | **Implemented & Verified** | MLP Regressor (64x32 hidden layers) trained on non-leaking split. |
| **Claim 5 (Deep Learning Prediction)** | Deep learning model / SVR on 3D holographic representations. | `src/models/svr_model.py` | Extracted feature matrix / 3D volumetric projections | Final blood glucose estimate | **Implemented & Verified** | Support Vector Regressor (SVR) with RBF kernel and StandardScaler scaling. |
| **Claim 8 (Resonance Mapping)** | Spectral resonance mapping across frequency channels. | `src/physics/environment.py` | Temperature (°C), relative humidity (%) | Atmospheric attenuation factor $\alpha(f)$ | **Implemented & Verified** | Arden Buck water vapor absorption + resonant line attenuation at 0.557 & 1.097 THz. |
| **Claim 10 (Method Workflow)** | Complete sequential pipeline from acquisition to 3D holography & prediction. | `app/app.py`<br>`src/holography/reconstruction.py` | Virtual patient profile | 3D volumetric intensity field $U(x,y,z)$, glucose prediction | **Implemented & Verified** | Physical Angular Spectrum Method (ASM) wave diffraction back-propagation. |

---

### Non-Claimed Boundary Statements
- **Clinical Validation:** This software prototype provides **computational simulation proof-of-concept**. It does NOT constitute clinical medical trial validation.
- **Hardware Interface:** Physical THz emitters/receivers are modeled via standard transfer function equations without physical USB/PCIe hardware drivers.
