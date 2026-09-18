"""
THz Non-Invasive Blood Glucose Estimation Prototype
Integrated 15-Page Research Workbench Streamlit Application
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

from src.core.engine import SystemEngine
from src.core.power_manager import PowerManager
from src.physics.spectrum_generator import SpectrumGenerator
from src.signal_processing.processor import SignalProcessor
from src.holography.spatial_scan import SpatialScanner
from src.holography.reconstruction import AngularSpectrumReconstructor
from src.holography.inverse_scattering import LinearizedInverseScattering
from src.models.svr_model import SVRModel
from src.models.mlp_model import MLPModel
from src.evaluation.metrics import MetricsCalculator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="THz Glucose Estimation Research Workbench",
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

if 'inverse_results' not in st.session_state:
    st.session_state['inverse_results'] = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔬 THz Research Workbench")
st.sidebar.caption("Patent Application 202541105574 Simulation")

pages = [
    "1. Patent System Overview",
    "2. Virtual Patient Model",
    "3. THz Source Engine (Unit 104)",
    "4. Multilayer Tissue Permittivity",
    "5. Electromagnetic TMM Response",
    "6. Virtual Receiver (Unit 106)",
    "7. Signal Processing & Calibration (Unit 112)",
    "8. Feature Extraction Table",
    "9. 3D Holography Reconstruction (Unit 108)",
    "10. Inverse THz Scattering Contrast",
    "11. Dataset Generator & Management",
    "12. AI Deep Learning Prediction (Unit 110)",
    "13. Untouched Test Set Validation",
    "14. Robustness & Ablation Analysis",
    "15. Automated Research Report"
]

selected_page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Scientific Notice:** Software-only computational simulation based on Transfer Matrix Method, Debye relaxation & Rytov inverse scattering. No hardware required.")

# ==========================================
# PAGE 1: SYSTEM OVERVIEW
# ==========================================
if selected_page == pages[0]:
    st.title("🔬 THz Non-Invasive Blood Glucose Estimation Workbench")
    st.subheader("Patent Application 202541105574: Software Simulation Prototype")
    
    col1, col2 = st.columns(2)
    with col1:
        st.container(border=True).markdown("""
        ### Patent Claimed Core 5-Stage Method
        1. **STEP 302:** Acquire spectral data from THz wave interaction with skin.
        2. **STEP 304:** Analyze received spectral data (wavelet denoising, baseline correction).
        3. **STEP 306:** Extract features (amplitude, phase delay, dielectric, absorption, PCA).
        4. **STEP 308:** Construct 3D holographic representation & inverse scattering contrast.
        5. **STEP 310:** Estimate blood glucose via calibrated Deep Learning model.
        """)
        
    with col2:
        st.container(border=True).markdown("""
        ### Patent Block Diagram Units
        - **Processing Engine 102:** Core orchestrator & memory allocator.
        - **THz Transmitter 104:** 0.1–10 THz wave source simulator.
        - **THz Receiver 106:** Coherent detection & noise model.
        - **Holographic Unit 108:** Phase-resolved 3D ASM back-propagation.
        - **Deep Learning Unit 110:** Trained SVR & MLP regressors.
        - **Signal Processing Unit 112:** Wavelet denoising & calibration.
        - **Environmental Sensors 114:** Temp, humidity & distance loss.
        - **Power Unit 118:** Computational energy allocator.
        """)

# ==========================================
# PAGE 2: VIRTUAL PATIENT MODEL
# ==========================================
elif selected_page == pages[1]:
    st.title("👤 Virtual Patient Setup")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider("Blood Glucose Concentration (mg/dL)", 50.0, 500.0, float(st.session_state['patient_config']['glucose']), step=5.0)
            temperature = st.slider("Skin Temperature (°C)", 10.0, 40.0, float(st.session_state['patient_config']['temperature']), step=0.5)
            humidity = st.slider("Ambient Relative Humidity (%)", 20.0, 90.0, float(st.session_state['patient_config']['humidity']), step=5.0)
        with col2:
            thickness = st.slider("Epidermal/Dermal Thickness (mm)", 1.0, 3.0, float(st.session_state['patient_config']['skin_thickness']), step=0.1)
            distance = st.slider("Sensor Distance (cm)", 0.5, 50.0, float(st.session_state['patient_config']['sensor_distance']), step=0.5)
            noise = st.slider("Measurement Receiver Noise (%)", 0.0, 5.0, float(st.session_state['patient_config']['noise_level']), step=0.1)
            
        seed = st.number_input("Random Seed", value=int(st.session_state['patient_config']['seed']))
        
        if st.form_submit_button("Save Patient Profile"):
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

    st.json(st.session_state['patient_config'])

# ==========================================
# PAGE 3: THz SOURCE ENGINE
# ==========================================
elif selected_page == pages[2]:
    st.title("🌊 THz Source Engine (Patent Unit 104)")
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
        st.success("✓ Forward physics simulation generated!")
        
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res['frequencies_thz'], res['incident_amplitude'], color='#0d6efd', linewidth=2)
        ax.set_title("Generated Incident THz Field E_inc(f)")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Click 'Run Forward Physics Simulation' to generate source field.")

# ==========================================
# PAGE 4: MULTILAYER TISSUE PERMITTIVITY
# ==========================================
elif selected_page == pages[3]:
    st.title("🧬 Multilayer Skin Permittivity (Debye Model)")
    st.markdown("Calculates dielectric relaxation perturbation $\Delta\varepsilon^*(\omega)$ as a function of glucose concentration.")
    
    if st.session_state['sim_results'] is not None:
        st.success(f"Dermis layer permittivity calculated at glucose = {st.session_state['patient_config']['glucose']} mg/dL.")
    else:
        st.info("Run physics simulation on Page 3 first.")

# ==========================================
# PAGE 5: ELECTROMAGNETIC TMM RESPONSE
# ==========================================
elif selected_page == pages[4]:
    st.title("📡 Electromagnetic Response (TMM)")
    
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(res['frequencies_thz'], res['ideal_reflection_amplitude'], color='blue', label='Ideal TMM')
        ax1.set_title("Reflectance Magnitude |R(f)|")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(res['frequencies_thz'], res['ideal_reflection_phase'], color='green', label='Phase')
        ax2.set_title("Phase Delay arg(R(f))")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Run simulation on Page 3 first.")

# ==========================================
# PAGE 6: VIRTUAL RECEIVER
# ==========================================
elif selected_page == pages[5]:
    st.title("📟 Virtual Coherent Receiver (Patent Unit 106)")
    
    if st.session_state['sim_results'] is not None:
        res = st.session_state['sim_results']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res['frequencies_thz'], res['ideal_reflection_amplitude'], color='blue', label='Ideal Reflection')
        ax.plot(res['frequencies_thz'], res['measured_noisy_amplitude'], color='red', alpha=0.6, label='Noisy Measured')
        ax.set_title("Receiver Signal Acquisition with Thermal/Shot Noise")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Run simulation on Page 3 first.")

# ==========================================
# PAGE 7: SIGNAL PROCESSING PIPELINE
# ==========================================
elif selected_page == pages[6]:
    st.title("🎛️ Signal Processing & Calibration (Patent Unit 112)")
    
    if st.session_state['sim_results'] is not None:
        sim_res = st.session_state['sim_results']
        if st.button("Execute Signal Processing"):
            sp = SignalProcessor()
            st.session_state['proc_results'] = sp.process_spectrum(
                sim_res['frequencies_thz'],
                sim_res['measured_noisy_amplitude'],
                sim_res['measured_noisy_phase']
            )
            st.success("✓ Signal processing executed!")
            
        if st.session_state['proc_results'] is not None:
            proc = st.session_state['proc_results']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(proc['frequencies_thz'], proc['raw_noisy_amplitude'], color='gray', alpha=0.5, label='Raw Noisy')
            ax.plot(proc['frequencies_thz'], proc['denoised_amplitude'], color='blue', label='Denoised (Savitzky-Golay)')
            ax.plot(proc['frequencies_thz'], proc['normalized_amplitude'], color='red', linewidth=2, label='Normalized [0, 1]')
            ax.set_title("Signal Processing Transformation Stages")
            ax.set_xlabel("Frequency (THz)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    else:
        st.info("Run simulation on Page 3 first.")

# ==========================================
# PAGE 8: FEATURE EXTRACTION TABLE
# ==========================================
elif selected_page == pages[7]:
    st.title("📊 Extracted Features (STEP 306)")
    if st.session_state['proc_results'] is not None:
        feats = st.session_state['proc_results']['features']
        df_f = pd.DataFrame(list(feats.items()), columns=['Feature Name', 'Calculated Value'])
        st.dataframe(df_f, width="stretch")
    else:
        st.info("Execute Signal Processing on Page 7 first.")

# ==========================================
# PAGE 9: 3D HOLOGRAPHY RECONSTRUCTION
# ==========================================
elif selected_page == pages[8]:
    st.title("🧊 3D Holographic Reconstruction (Patent Unit 108)")
    cfg = st.session_state['patient_config']
    
    if st.button("Reconstruct 3D Hologram (ASM)"):
        scanner = SpatialScanner(grid_size_x=16, grid_size_y=16)
        grid_data = scanner.acquire_spatial_grid(glucose_mg_dl=cfg['glucose'], base_seed=cfg['seed'])
        reconstructor = AngularSpectrumReconstructor(depth_max_mm=2.0, depth_steps=16)
        st.session_state['hologram_results'] = reconstructor.reconstruct_volume(grid_data)
        st.success("✓ 3D Holographic volume reconstructed!")
        
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
            ax2.set_title("XZ Depth Slice")
            fig2.colorbar(im2, ax=ax2)
            st.pyplot(fig2)
    else:
        st.info("Click 'Reconstruct 3D Hologram' to compute volume.")

# ==========================================
# PAGE 10: INVERSE SCATTERING
# ==========================================
elif selected_page == pages[9]:
    st.title("🔬 Inverse THz Scattering Dielectric Contrast")
    cfg = st.session_state['patient_config']
    
    if st.button("Estimate Dielectric Contrast"):
        scanner = SpatialScanner(grid_size_x=12, grid_size_y=12)
        grid_data = scanner.acquire_spatial_grid(glucose_mg_dl=cfg['glucose'], base_seed=cfg['seed'])
        inv = LinearizedInverseScattering(depth_max_mm=2.0, depth_steps=12)
        st.session_state['inverse_results'] = inv.estimate_dielectric_contrast(grid_data)
        st.success("✓ Inverse scattering contrast estimated!")
        
    if st.session_state['inverse_results'] is not None:
        inv_res = st.session_state['inverse_results']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(inv_res['z_depths_mm'], inv_res['mean_contrast_by_depth'], color='darkred', marker='o')
        ax.set_title("Reconstructed Mean Dielectric Contrast Delta_eps vs Subcutaneous Depth (mm)")
        ax.set_xlabel("Depth z (mm)")
        ax.set_ylabel("Dielectric Contrast Delta_eps")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Click 'Estimate Dielectric Contrast' to compute.")

# ==========================================
# PAGE 11: DATASET GENERATOR
# ==========================================
elif selected_page == pages[10]:
    st.title("💾 Synthetic Dataset Management")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'thz_glucose_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success(f"✓ Dataset present: {len(df)} samples, {len(df.columns)} features.")
        st.dataframe(df.head(10), width="stretch")
    else:
        st.warning("Dataset not found. Run generate_dataset.py CLI script.")

# ==========================================
# PAGE 12: AI DEEP LEARNING PREDICTION
# ==========================================
elif selected_page == pages[11]:
    st.title("🧠 AI Deep Learning Prediction (Patent Unit 110)")
    if st.session_state['proc_results'] is not None:
        feats = st.session_state['proc_results']['features']
        feat_vals = np.array([[feats[k] for k in feats.keys()]])
        
        svr_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'svr_model.pkl')
        mlp_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlp_model.pkl')
        
        if os.path.exists(svr_path) and os.path.exists(mlp_path):
            svr = SVRModel()
            svr.load(svr_path)
            mlp = MLPModel()
            mlp.load(mlp_path)
            
            p_svr = svr.predict(feat_vals)[0]
            p_mlp = mlp.predict(feat_vals)[0]
            actual = st.session_state['patient_config']['glucose']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reference Glucose", f"{actual:.1f} mg/dL")
            with col2:
                st.metric("SVR Prediction", f"{p_svr:.1f} mg/dL", delta=f"{p_svr - actual:.1f}")
            with col3:
                st.metric("MLP Prediction", f"{p_mlp:.1f} mg/dL", delta=f"{p_mlp - actual:.1f}")
        else:
            st.error("Trained models not found. Run train_models.py.")
    else:
        st.info("Execute Signal Processing on Page 7 first.")

# ==========================================
# PAGE 13: UNTOUCHED TEST SET VALIDATION
# ==========================================
elif selected_page == pages[12]:
    st.title("📈 Untouched Test Set Metrics (STEP 310)")
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        st.json(m)
    else:
        st.warning("Metrics file not found. Run train_models.py.")

# ==========================================
# PAGE 14: ROBUSTNESS & ABLATION
# ==========================================
elif selected_page == pages[13]:
    st.title("🛡️ Robustness & Pipeline Ablation")
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments')
    ablation_csv = os.path.join(exp_dir, 'ablation_study.csv')
    if os.path.exists(ablation_csv):
        st.dataframe(pd.read_csv(ablation_csv), width="stretch")
    else:
        st.warning("Run run_experiments.py CLI script.")

# ==========================================
# PAGE 15: AUTOMATED REPORT
# ==========================================
else:
    st.title("📄 Automated Research Report & Assumptions")
    report_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'reports', 'experiment_report.html')
    if os.path.exists(report_path):
        st.success(f"✓ Experiment report available at `{report_path}`")
    else:
        st.info("Run run_experiments.py CLI script to compile HTML report.")
