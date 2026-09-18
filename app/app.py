"""
THz Non-Invasive Blood Glucose Estimation Prototype
Integrated 12-Page Research Workbench Streamlit Application
"""

import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# Insert repository root into python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics.spectrum_generator import SpectrumGenerator
from src.signal_processing.processor import SignalProcessor
from src.holography.spatial_scan import SpatialScanner
from src.holography.reconstruction import AngularSpectrumReconstructor
from src.models.svr_model import SVRModel
from src.models.mlp_model import MLPModel
from src.evaluation.metrics import MetricsCalculator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="THz Glucose Estimation Simulation Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
if 'patient_config' not in st.session_state:
    st.session_state['patient_config'] = {
        'glucose': 120.0,
        'temperature': 32.0,
        'humidity': 50.0,
        'skin_thickness': 1.8,
        'sensor_distance': 5.0,
        'noise_level': 1.5,
        'seed': 42
    }
    
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

if 'proc_results' not in st.session_state:
    st.session_state['proc_results'] = None

if 'hologram_results' not in st.session_state:
    st.session_state['hologram_results'] = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔬 THz Research Workbench")
st.sidebar.caption("Patent Implementation Software Simulation")

pages = [
    "1. System Overview & Patent Mapping",
    "2. Virtual Patient Setup",
    "3. THz Source & Tissue Simulation",
    "4. Electromagnetic Response (TMM)",
    "5. Signal Processing Pipeline",
    "6. Feature Extraction Table",
    "7. 3D Holographic Reconstruction",
    "8. Dataset Generator & Management",
    "9. AI Machine Learning Prediction",
    "10. Validation & Test Performance",
    "11. Robustness & Sensitivity Analysis",
    "12. Experiment Report & Assumptions"
]

selected_page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Scientific Notice:** Software-only computational simulation based on Transfer Matrix Method & Debye relaxation. No hardware required.")

# ==========================================
# PAGE 1: SYSTEM OVERVIEW
# ==========================================
if selected_page == pages[0]:
    st.title("🔬 THz Non-Invasive Blood Glucose Estimation Workbench")
    st.subheader("Patent Application Implementation: Fully Software Computational Prototype")
    
    st.markdown("""
    This application serves as the **software simulation prototype** implementing the published patent architecture for contactless non-invasive blood glucose estimation.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.container(border=True).markdown("""
        ### Core Scientific Pipeline
        1. **Virtual Patient Model**: Configurable glucose, temperature, humidity, thickness, noise.
        2. **Forward Physics Engine**: 0.1–2.0 THz frequency grid, Debye relaxation permittivity perturbation.
        3. **Electromagnetic TMM**: Complex Fresnel multi-layer wave interaction & phase delay calculation.
        4. **Virtual Receiver & Environment**: Atmospheric water vapor attenuation & thermal/shot noise.
        5. **Signal Processing**: Savitzky-Golay filtering, baseline drift correction, SNV/Min-Max normalization.
        6. **Feature Extraction**: 15 spectral, phase, and entropy parameters.
        7. **3D Holography**: Physical Angular Spectrum Method (ASM) wave back-propagation.
        8. **AI ML Regressors**: SVR & MLP models trained on non-leaking dataset splits.
        """)
        
    with col2:
        st.container(border=True).markdown("""
        ### Architectural Flow Diagram
        ```
        Virtual Patient Parameters
                  ↓
          THz Source Engine
                  ↓
        Multilayer Tissue (Debye Model)
                  ↓
       Transfer Matrix Method (TMM)
                  ↓
      Atmospheric Loss & Receiver Noise
                  ↓
      Signal Processing & Denoising
                  ↓
        Feature Extraction (15 Features)
           ┌──────┴──────┐
           ↓             ↓
      3D Holography   ML Regression
      (ASM Wave propagation)  (SVR / MLP)
           ↓             ↓
      3D Tissue Map   Glucose Estimate
           └──────┬──────┘
                  ↓
       Validation & Robustness Report
        ```
        """)

# ==========================================
# PAGE 2: VIRTUAL PATIENT SETUP
# ==========================================
elif selected_page == pages[1]:
    st.title("👤 Virtual Patient Setup")
    st.markdown("Configure physiological and environmental simulation parameters.")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider("Blood Glucose Concentration (mg/dL)", 50.0, 300.0, float(st.session_state['patient_config']['glucose']), step=5.0)
            temperature = st.slider("Skin Temperature (°C)", 20.0, 40.0, float(st.session_state['patient_config']['temperature']), step=0.5)
            humidity = st.slider("Ambient Relative Humidity (%)", 20.0, 90.0, float(st.session_state['patient_config']['humidity']), step=5.0)
        with col2:
            thickness = st.slider("Epidermal/Dermal Thickness (mm)", 1.0, 3.0, float(st.session_state['patient_config']['skin_thickness']), step=0.1)
            distance = st.slider("Sensor-to-Skin Distance (cm)", 1.0, 20.0, float(st.session_state['patient_config']['sensor_distance']), step=0.5)
            noise = st.slider("Measurement Receiver Noise (%)", 0.0, 5.0, float(st.session_state['patient_config']['noise_level']), step=0.1)
            
        seed = st.number_input("Random Seed (Reproducibility)", value=int(st.session_state['patient_config']['seed']))
        
        submitted = st.form_submit_button("Save Patient Profile")
        if submitted:
            st.session_state['patient_config'] = {
                'glucose': glucose,
                'temperature': temperature,
                'humidity': humidity,
                'skin_thickness': thickness,
                'sensor_distance': distance,
                'noise_level': noise,
                'seed': seed
            }
            st.success("✓ Virtual Patient profile updated!")

    st.subheader("Active Configuration Matrix")
    st.json(st.session_state['patient_config'])

# ==========================================
# PAGE 3: THz SOURCE & TISSUE SIMULATION
# ==========================================
elif selected_page == pages[2]:
    st.title("🌊 THz Source & Multilayer Tissue Physics Engine")
    
    cfg = st.session_state['patient_config']
    
    if st.button("Run Forward Physics Simulation"):
        sim = SpectrumGenerator(
            glucose_mg_dl=cfg['glucose'],
            temperature_c=cfg['temperature'],
            humidity_pct=cfg['humidity'],
            skin_thickness_mm=cfg['skin_thickness'],
            sensor_distance_cm=cfg['sensor_distance'],
            noise_level_pct=cfg['noise_level'],
            seed=cfg['seed']
        )
        st.session_state['sim_results'] = sim.generate_full_simulation()
        st.success("✓ Physics forward simulation generated successfully!")
        
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(res['frequencies_thz'], res['incident_amplitude'], color='#0d6efd', linewidth=2)
        ax1.set_title("Incident Spectral Envelope E_inc(f)")
        ax1.set_xlabel("Frequency (THz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(res['frequencies_thz'], res['ideal_reflection_amplitude'], color='#198754', linewidth=2, label="Ideal TMM")
        ax2.plot(res['frequencies_thz'], res['env_attenuated_amplitude'], color='#ffc107', linewidth=2, label="Atmospheric Loss")
        ax2.set_title("Reflection Spectrum R(f)")
        ax2.set_xlabel("Frequency (THz)")
        ax2.set_ylabel("Reflectance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        st.warning("Click 'Run Forward Physics Simulation' to generate THz response.")

# ==========================================
# PAGE 4: ELECTROMAGNETIC RESPONSE (TMM)
# ==========================================
elif selected_page == pages[3]:
    st.title("📡 Electromagnetic Multilayer Reflection (TMM)")
    
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(res['frequencies_thz'], res['ideal_reflection_amplitude'], color='blue', label='Clean TMM')
        ax1.plot(res['frequencies_thz'], res['measured_noisy_amplitude'], color='red', alpha=0.6, label='Noisy Measured')
        ax1.set_title("Reflection Amplitude |R(f)|")
        ax1.set_xlabel("Frequency (THz)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(res['frequencies_thz'], res['ideal_reflection_phase'], color='green', label='Clean Phase')
        ax2.plot(res['frequencies_thz'], res['measured_noisy_phase'], color='purple', alpha=0.6, label='Noisy Phase')
        ax2.set_title("Phase Spectrum arg(R(f))")
        ax2.set_xlabel("Frequency (THz)")
        ax2.set_ylabel("Phase (radians)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        st.info("Run physics simulation on Page 3 first.")

# ==========================================
# PAGE 5: SIGNAL PROCESSING PIPELINE
# ==========================================
elif selected_page == pages[4]:
    st.title("🎛️ Signal Processing & Calibration Pipeline")
    
    if st.session_state['sim_results'] is not None:
        sim_res = st.session_state['sim_results']
        
        if st.button("Execute Signal Processing Pipeline"):
            sp = SignalProcessor()
            st.session_state['proc_results'] = sp.process_spectrum(
                sim_res['frequencies_thz'],
                sim_res['measured_noisy_amplitude'],
                sim_res['measured_noisy_phase']
            )
            st.success("✓ Signal processing pipeline executed!")
            
        if st.session_state['proc_results'] is not None:
            proc = st.session_state['proc_results']
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(proc['frequencies_thz'], proc['raw_noisy_amplitude'], color='gray', alpha=0.5, label='1. Raw Noisy')
            ax.plot(proc['frequencies_thz'], proc['denoised_amplitude'], color='blue', label='2. Denoised (Savitzky-Golay)')
            ax.plot(proc['frequencies_thz'], proc['corrected_amplitude'], color='green', label='3. Baseline Corrected')
            ax.plot(proc['frequencies_thz'], proc['normalized_amplitude'], color='red', linewidth=2, label='4. Normalized [0, 1]')
            ax.set_title("Sequential Signal Processing Pipeline Transformation")
            ax.set_xlabel("Frequency (THz)")
            ax.set_ylabel("Amplitude Level")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    else:
        st.info("Run simulation on Page 3 first.")

# ==========================================
# PAGE 6: FEATURE EXTRACTION TABLE
# ==========================================
elif selected_page == pages[5]:
    st.title("📊 Extracted Feature Table")
    
    if st.session_state['proc_results'] is not None:
        feats = st.session_state['proc_results']['features']
        df_feats = pd.DataFrame(list(feats.items()), columns=['Feature Parameter', 'Calculated Value'])
        st.dataframe(df_feats, width="stretch")
    else:
        st.info("Execute Signal Processing on Page 5 first.")

# ==========================================
# PAGE 7: 3D HOLOGRAPHIC RECONSTRUCTION
# ==========================================
elif selected_page == pages[6]:
    st.title("🧊 3D Computational Holographic Reconstruction")
    st.markdown("Angular Spectrum Method (ASM) diffraction back-propagation volume generation.")
    
    cfg = st.session_state['patient_config']
    
    if st.button("Run 3D Holographic Reconstruction"):
        scanner = SpatialScanner(grid_size_x=16, grid_size_y=16)
        grid_data = scanner.acquire_spatial_grid(glucose_mg_dl=cfg['glucose'], base_seed=cfg['seed'])
        reconstructor = AngularSpectrumReconstructor(depth_max_mm=2.0, depth_steps=16)
        st.session_state['hologram_results'] = reconstructor.reconstruct_volume(grid_data)
        st.success("✓ 3D Hologram reconstructed!")
        
    if st.session_state['hologram_results'] is not None:
        holo = st.session_state['hologram_results']
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            im1 = ax1.imshow(holo['slice_xy'], cmap='hot')
            ax1.set_title("XY Surface Slice")
            fig1.colorbar(im1, ax=ax1)
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            im2 = ax2.imshow(holo['slice_xz'].T, cmap='magma', aspect='auto')
            ax2.set_title("XZ Depth Slice (Z = Depth)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Depth Step")
            fig2.colorbar(im2, ax=ax2)
            st.pyplot(fig2)
    else:
        st.info("Click 'Run 3D Holographic Reconstruction' to compute volume.")

# ==========================================
# PAGE 8: DATASET GENERATOR
# ==========================================
elif selected_page == pages[7]:
    st.title("💾 Synthetic Dataset Management")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'thz_glucose_dataset.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success(f"✓ Dataset present: {len(df)} samples, {len(df.columns)} columns.")
        st.dataframe(df.head(10), width="stretch")
    else:
        st.warning("Dataset not found. Run generate_dataset.py CLI command.")

# ==========================================
# PAGE 9: AI PREDICTION
# ==========================================
elif selected_page == pages[8]:
    st.title("🧠 AI Machine Learning Glucose Prediction")
    
    if st.session_state['proc_results'] is not None:
        feats = st.session_state['proc_results']['features']
        feat_names = list(feats.keys())
        feat_vals = np.array([[feats[k] for k in feat_names]])
        
        svr_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'svr_model.pkl')
        mlp_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlp_model.pkl')
        
        if os.path.exists(svr_path) and os.path.exists(mlp_path):
            svr = SVRModel()
            svr.load(svr_path)
            
            mlp = MLPModel()
            mlp.load(mlp_path)
            
            pred_svr = svr.predict(feat_vals)[0]
            pred_mlp = mlp.predict(feat_vals)[0]
            actual = st.session_state['patient_config']['glucose']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reference Glucose", f"{actual:.1f} mg/dL")
            with col2:
                st.metric("SVR Prediction", f"{pred_svr:.1f} mg/dL", delta=f"{pred_svr - actual:.1f}")
            with col3:
                st.metric("MLP Prediction", f"{pred_mlp:.1f} mg/dL", delta=f"{pred_mlp - actual:.1f}")
        else:
            st.error("Trained models not found. Run train_models.py script.")
    else:
        st.info("Execute Signal Processing on Page 5 first.")

# ==========================================
# PAGE 10: VALIDATION RESULTS
# ==========================================
elif selected_page == pages[9]:
    st.title("📈 Model Validation & Metrics")
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'metrics.json')
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m = json.load(f)
            
        st.subheader("SVR Test Set Performance")
        st.json(m['svr']['test'])
        
        st.subheader("MLP Test Set Performance")
        st.json(m['mlp']['test'])
    else:
        st.warning("Metrics file not found. Run train_models.py.")

# ==========================================
# PAGE 11: ROBUSTNESS ANALYSIS
# ==========================================
elif selected_page == pages[10]:
    st.title("🛡️ Robustness & Sensitivity Analysis")
    noise_csv = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'noise_robustness.csv')
    
    if os.path.exists(noise_csv):
        df_n = pd.read_csv(noise_csv)
        st.subheader("MAE vs Receiver Noise Level (%)")
        st.line_chart(df_n.set_index('noise_level_pct')['mean_mae'])
    else:
        st.warning("Run run_experiments.py CLI script.")

# ==========================================
# PAGE 12: EXPERIMENT REPORT & ASSUMPTIONS
# ==========================================
else:
    st.title("📄 Experiment Report & Physical Assumptions")
    st.markdown("""
    ### Physical Assumptions
    1. **Debye Model:** Interstitial fluid permittivity is governed by single/double Debye relaxation.
    2. **Transfer Matrix Method:** Multilayer skin assumed optically flat at THz scale.
    3. **Atmospheric Attenuation:** Arden Buck water vapor absorption equation.
    
    ### Scientific Limitations
    - This software simulation prototype does NOT replace clinical human trial validation.
    """)
