# RESEARCH & SCIENTIFIC LIMITATIONS

This document explicitly outlines the technical, physical, and clinical limitations of the THz Non-Invasive Blood Glucose Estimation Software Prototype.

---

### 1. Simulation-to-Reality Gap
- **Synthetic Forward Physics:** Measurements are generated computationally via single/double Debye relaxation and planar Transfer Matrix Method (TMM) equations. Physical skin tissue contains non-uniform micro-structures (sweat ducts, hair follicles, surface roughness) that induce scattering not captured by 1D planar TMM.
- **Detector Dynamics:** The virtual receiver simulates additive Gaussian noise and low-pass bandwidth limits, but does not capture non-linear photoconductive antenna (PCA) saturation or laser pulse jitter.

---

### 2. Clinical Scope & Non-Medical Disclaimer
- **Not Clinically Validated:** This prototype is a **computational simulation benchmark** for patent claim functional correspondence. It has NOT undergone clinical human trial validation.
- **Diagnostic Boundary:** The system outputs simulated predictions ($70 - 200 \text{ mg/dL}$) for research demonstration purposes. It must NOT be used for direct clinical diagnosis or therapeutic insulin dosing.

---

### 3. Environmental Dependency
- **Water Vapor Sensitivity:** Terahertz waves ($0.1 - 2.0 \text{ THz}$) are highly absorbed by atmospheric moisture. Extreme ambient humidity ($>80\%$) or sensor distances ($>20 \text{ cm}$) severely degrade signal-to-noise ratio (SNR), requiring strict environmental baseline calibration.
