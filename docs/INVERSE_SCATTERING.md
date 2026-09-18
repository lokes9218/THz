# INVERSE THZ SCATTERING & DIELECTRIC CONTRAST APPROXIMATION

This document details the mathematical formulation and physical approximations for the inverse Terahertz (THz) scattering module (`src/holography/inverse_scattering.py`) corresponding to Patent Claim 1 and Step 308.

---

### 1. Physical Inverse Problem Statement
Forward scattering relates the complex reflected field $R(f)$ to the spatial multi-layer refractive index distribution $n^*(z, f)$. In inverse THz scattering, measured complex fields $E_{\text{ref}}(x,y,f)$ across a 2D spatial aperture are inverted under a linearized **Rytov / Born Scattering Approximation** to estimate local spatial dielectric contrast $\Delta\varepsilon(x,y,z)$.

---

### 2. Mathematical Rytov Phase Retrieval Shift Formulation

For a incident complex field $E_{\text{inc}}(x,y,f)$ and reflected field $E_{\text{ref}}(x,y,f)$:

$$\psi(x,y,f) = \ln\left(\frac{E_{\text{ref}}(x,y,f)}{E_{\text{inc}}(x,y,f) + \epsilon}\right) = \phi_{\text{scat}}(x,y,f) + j \theta_{\text{phase}}(x,y,f)$$

Under the first-order Rytov approximation, local spatial perturbation in relative dielectric constant $\Delta\varepsilon_{\text{r}}(x,y,z)$ at depth $z$ is related to phase shift $\theta_{\text{phase}}(x,y,f)$ and local attenuation factor $\alpha(f)$:

$$\Delta\varepsilon_{\text{r}}(x,y,z) \approx \frac{2 n_0}{k_0 z} \cdot \theta_{\text{phase}}(x,y,f_{\text{center}})$$

Where:
- $n_0 = 1.0$: Refractive index of air interface.
- $k_0 = \frac{2\pi f}{c}$: Free-space wavenumber ($\text{rad/m}$).
- $z$: Reconstructed depth in subcutaneous tissue ($\text{mm}$).

---

### 3. Implementation Lineage
The inverse scattering estimator is implemented in `src/holography/inverse_scattering.py`:
- **Input:** 2D spatial complex field $H(x,y,f)$ from `SpatialScanner` or `THzReceiver`.
- **Method:** Solves phase unwrapping and Born matrix inversion over spatial frequency grid $(k_x, k_y)$.
- **Output:** Estimated 3D spatial dielectric contrast matrix $\Delta\varepsilon(x,y,z)$ mapped to subcutaneous depth layers.
