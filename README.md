# THz Non-Invasive Blood Glucose Estimation Prototype

**Patent Application Number:** 202541105574  
**Title:** Blood Glucose Level Estimation System  
**Filing Date:** October 31, 2025  

A fully reproducible, research-oriented **software-only computational simulation prototype** implementing terahertz (THz) spectroscopy, electromagnetic multilayer transfer matrix modeling, signal processing, 3D computational holography (Angular Spectrum Method), and machine learning regression for non-invasive blood glucose estimation.

---

## 🌟 Key Features

- **Virtual Patient Model:** Configurable glucose ($50 - 300 \text{ mg/dL}$), skin temperature ($20 - 40 ^\circ\text{C}$), humidity ($20 - 90\%$), skin thickness, sensor distance, and receiver noise.
- **Forward Physics Engine:** $0.1 - 2.0 \text{ THz}$ frequency grid, double Debye relaxation equations for dermal tissue permittivity, and planar Transfer Matrix Method (TMM).
- **Atmospheric & Receiver Model:** Arden Buck water vapor absorption, THz line attenuation (0.557, 1.097 THz), and thermal/shot receiver noise.
- **Signal Processing:** Savitzky-Golay filtering, baseline polynomial drift correction, Min-Max / SNV normalization, and 15 extracted spectral/phase features.
- **3D Computational Holography:** Physical Angular Spectrum Method (ASM) wave back-propagation generating 3D volumetric intensity fields $[N_x, N_y, N_z]$ and depth slices ($XY, XZ, YZ$).
- **AI Machine Learning:** Trainable Support Vector Regressor (SVR) and Multi-Layer Perceptron (MLP) trained on non-leaking dataset splits ($70\% / 15\% / 15\%$) with dynamic MAE, RMSE, $R^2$, and MAPE metric calculation.
- **Streamlit Workbench:** Interactive 12-page research workbench application.
- **Automated HTML Report:** One-click automated report generation compiling experiment summaries.

---

## 🛠️ Quick Start & CLI Usage

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Physics Dataset
```bash
python generate_dataset.py
```
Outputs `data/datasets/thz_glucose_dataset.csv` (2,000 samples).

### 3. Train Machine Learning Models
```bash
python train_models.py
```
Trains SVR and MLP regressors, exports models to `models/`, and saves test set metrics to `results/metrics/metrics.json`.

### 4. Run Sensitivity, Robustness & Ablation Experiments
```bash
python run_experiments.py
```
Executes noise sweeps ($0\% - 5\%$), temperature sweeps ($20 - 40^\circ\text{C}$), ablation studies, and generates `results/reports/experiment_report.html`.

### 5. Launch Interactive Streamlit Research Workbench
```bash
streamlit run app/app.py
```

---

## 📂 Project Structure

```
d:\THz_Glucose_Prototype\
├── config.yaml                          # Simulation parameters
├── requirements.txt                     # Dependencies
├── README.md                            # Master overview
├── ASSUMPTIONS.md                       # Physical equations documentation
├── LIMITATIONS.md                       # Simulation boundaries & disclaimers
├── PROTOTYPE_ARCHITECTURE.md            # System architecture design
├── PATENT_IMPLEMENTATION_MAPPING.md     # Patent Claim 1-10 mapping matrix
├── generate_dataset.py                  # Dataset generator CLI
├── train_models.py                      # SVR/MLP training CLI
├── run_experiments.py                   # Experiments CLI
├── app/
│   └── app.py                           # 12-page Streamlit Research Workbench
├── src/
│   ├── physics/                         # Forward physics & TMM engine
│   ├── signal_processing/               # Denoising & feature extraction
│   ├── holography/                      # 3D ASM wave back-propagation
│   ├── models/                          # SVR & MLP regressors
│   ├── evaluation/                      # Dynamic metrics & robustness
│   └── utils/                           # HTML report generator
└── tests/                               # Automated unit test suite
```

---

## 📄 License & Citation
Patent Application No. 202541105574, Vellore Institute of Technology (VIT) Chennai.
