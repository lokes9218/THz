# PHYSICAL & SIMULATION ASSUMPTIONS

This document provides complete scientific documentation of all mathematical equations, physical relaxation models, atmospheric attenuation parameters, and simulation assumptions used throughout the THz Non-Invasive Blood Glucose Prototype.

---

### 1. Multilayer Tissue Permittivity (Debye Relaxation Model)

Complex dielectric permittivity $\varepsilon^*(f) = \varepsilon'(f) - j\varepsilon''(f)$ for human skin tissue layers (Stratum Corneum, Epidermis, Dermis, Subcutaneous) is modeled via the single/double Debye relaxation equation:

$$\varepsilon^*(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{1 + j\omega\tau}$$

Where:
- $\varepsilon_\infty$: High-frequency optical permittivity limit.
- $\Delta\varepsilon = \varepsilon_s - \varepsilon_\infty$: Static dielectric relaxation strength.
- $\tau$: Dielectric relaxation time (picoseconds).
- $\omega = 2\pi f$: Angular frequency ($\text{rad/s}$).

#### Glucose Sensitivity Perturbation (Dermis Layer)
Blood glucose concentration ($G$, in $\text{mg/dL}$) alters free/bound water ratios in the interstitial fluid layer, perturbing static relaxation strength:

$$\Delta\varepsilon_{\text{dermis}}(G) = \Delta\varepsilon_{\text{nominal}} + (G - 100) \times 0.0015$$
$$\varepsilon_{\infty, \text{dermis}}(G) = \varepsilon_{\infty, \text{nominal}} + (G - 100) \times 0.0005$$

---

### 2. Electromagnetic Propagation (Transfer Matrix Method - TMM)

Electromagnetic wave reflection $R(f)$ across multi-layer planar tissue interfaces is calculated via 2x2 layer characteristic matrix multiplication:

$$M_{\text{total}} = M_1 \cdot M_2 \cdot M_3 \cdot M_4$$

For each layer $i$ with thickness $d_i$ and complex refractive index $n_i^*(f) = \sqrt{\varepsilon_i^*(f)}$:

$$M_i = \begin{pmatrix} \cos(\delta_i) & -j \frac{\sin(\delta_i)}{n_i^*} \\ -j n_i^* \sin(\delta_i) & \cos(\delta_i) \end{pmatrix}$$

Where phase thickness $\delta_i = k_0 n_i^* d_i = \frac{2\pi f}{c} n_i^* d_i$.

Equivalent input admittance $Y_{\text{in}} = \frac{M_{21} + M_{22} n_{\text{sub}}}{M_{11} + M_{12} n_{\text{sub}}}$.  
Complex reflection coefficient $r(f) = \frac{n_0 - Y_{\text{in}}}{n_0 + Y_{\text{in}}}$.

---

### 3. Atmospheric Water Vapor Attenuation Model

Atmospheric THz signal attenuation $\alpha_{\text{total}}(f)$ in $\text{dB/m}$ is calculated as a function of temperature ($T$, $^\circ\text{C}$), relative humidity ($H$, $\%$), and distance ($d$, $\text{m}$):

$$\rho_v = \frac{H}{100} \cdot \frac{6.1121 \cdot e^{\frac{18.678 T}{257.14 + T}} \cdot 216.7}{T + 273.15} \quad (\text{g/m}^3)$$

Continuum absorption plus resonant water lines at $0.557, 0.752, 1.097, 1.163, 1.689 \text{ THz}$:

$$\alpha_{\text{total}}(f) = 0.05 f^2 \left(\frac{\rho_v}{10}\right) + \sum_{k} S_k \left(\frac{\rho_v}{10}\right) \frac{\gamma_k^2}{(f - f_k)^2 + \gamma_k^2} \quad (\text{dB/m})$$

Round-trip atmospheric transmission factor:

$$T_{\text{env}}(f) = \exp\left(-\frac{\alpha_{\text{total}}(f)}{8.686} \cdot 2 d\right)$$

---

### 4. 3D Computational Holography (Angular Spectrum Method - ASM)

Volumetric 3D complex field propagation $U(x,y,z)$ from 2D spatial surface measurement $U(x,y,0)$ is computed using the angular spectrum diffraction equation:

$$U(x,y,z) = \mathcal{F}^{-1}_{x,y} \left\{ \mathcal{F}_{x,y}\{U(x,y,0)\} \cdot \exp\left(j z \sqrt{k_0^2 - k_x^2 - k_y^2}\right) \right\}$$

Voxel intensity volume $I(x,y,z) = |U(x,y,z)|^2$.
