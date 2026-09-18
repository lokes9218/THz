# PROTOTYPE ARCHITECTURE & SYSTEM DESIGN

This document describes the software architecture, modular directory structure, dataflow pipelines, and execution CLI scripts for the THz Non-Invasive Blood Glucose Prototype.

---

### 1. End-to-End Pipeline Architecture

```
Virtual Patient Profile (Glucose, Temp, Humidity, Thickness, Distance, Noise)
       ↓
THz Source Engine (0.1 - 2.0 THz Grid, Incident Field Envelope)
       ↓
Multilayer Tissue Model (Stratum Corneum, Epidermis, Dermis [Debye Model], Subcutaneous)
       ↓
Transfer Matrix Method (TMM) Reflection Engine (Complex Reflection & Phase)
       ↓
Atmospheric Environmental Attenuation & Virtual THz Receiver (Noise & Bandwidth)
       ↓
Signal Processing Pipeline (Savitzky-Golay Denoising, Baseline Correction, SNV/Min-Max Normalization)
       ↓
Feature Extraction Engine (15 Spectral, Phase, & Entropy Features)
       ↓
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 ↓                                                                               ↓
3D Angular Spectrum Computational Holography                             AI Machine Learning Engine
 (Spatial Scanning & ASM Wave Back-Propagation)                           (SVR & MLP Regressors)
 ↓                                                                               ↓
3D Volumetric Tissue Intensity Maps (XY, XZ, YZ)                          Glucose Estimation & Metrics
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         ↓
                     Robustness, Sensitivity & Ablation Analysis
                                         ↓
                 Automated Research Experiment Report Generator
```

---

### 2. Standalone Command-Line Interface (CLI) Executables

```bash
# 1. Generate Synthetic Dataset (2,000 samples)
python generate_dataset.py

# 2. Train SVR & MLP Regressors (Non-leaking splits, Scaler export)
python train_models.py

# 3. Execute Sensitivity Sweeps, Ablation Studies & HTML Report
python run_experiments.py

# 4. Launch 12-Page Streamlit Research Workbench Application
streamlit run app/app.py
```
