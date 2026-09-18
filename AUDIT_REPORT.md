# AUDIT REPORT

**Project Name:** THz Non-Invasive Blood Glucose Level Estimation Software Prototype  
**Patent Application Number:** 202541105574  

---

### 1. Current Architecture Overview
The repository initially contained an elementary 8-page Streamlit application script (`app/app.py`) along with skeleton physics and model files in `src/`.

---

### 2. Existing Modules Inspection
- `app/app.py`: Streamlit frontend pages.
- `src/physics/tissue_model.py`: Basic skin layer thickness and nominal permittivity dictionary.
- `src/physics/thz_source.py`: Basic linear frequency vector generator.
- `src/physics/reflection.py`: Transfer Matrix Method skeleton.
- `src/signal_processing/processor.py`: Wavelet denoising placeholder.
- `src/holography/reconstructor.py`: Depth-weighted volume generator.
- `src/models/svr_model.py`: Basic scikit-learn SVR wrapper.

---

### 3. Identified Broken & Invalid Logic
1. **Mock Scientific Predictions:** Fake predictions generated using `glucose + np.random.normal(0, 5)` in `app.py`.
2. **Hardcoded Evaluation Metrics:** Static constants (`MAE = 5.23`, `RMSE = 7.14`, `R² = 0.9234`) displayed without dynamic evaluation.
3. **Incomplete Physics Modeling:** Missing complex Debye dielectric relaxation dispersion, atmospheric water vapor line attenuation, and physical sensor distance propagation.
4. **Placeholder Holography:** Depth-weighted spatial smoothing rather than physical Angular Spectrum Method (ASM) diffraction back-propagation.
5. **Data Leakage Risks:** Dataset generation lacked patient-level grouping.

---

### 4. Rebuild Strategy
- **Deleted:** All hardcoded metrics, fake prediction functions, and isolated demo buttons.
- **Rewritten / Extended:** Core physics engine (Debye dispersion, 2x2 TMM admittance formulation, atmospheric loss), signal processing pipeline, 3D ASM computational holography, trainable SVR and MLP regressors, and 15-page Streamlit workbench.
