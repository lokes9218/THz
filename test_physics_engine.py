"""Test the THz physics engine"""

from src.physics import Spectrum_Generator
import numpy as np

print("=" * 60)
print("THz GLUCOSE ESTIMATION - PHYSICS ENGINE TEST")
print("=" * 60)

# Test 1: Single spectrum at 100 mg/dL
print("\n[TEST 1] Generating spectrum at 100 mg/dL...")
gen_100 = Spectrum_Generator(glucose_level=100)
gen_100.generate()
print(f"✓ Spectrum generated")
print(f"  Amplitude: {np.min(gen_100._amplitude):.4f} - {np.max(gen_100._amplitude):.4f}")

# Test 2: Multiple glucose levels
print("\n[TEST 2] Generating spectra at multiple glucose levels...")
glucose_levels = [70, 100, 150, 200]
for glucose in glucose_levels:
    gen = Spectrum_Generator(glucose_level=glucose)
    gen.generate()
    amp_mean = np.mean(gen._amplitude)
    print(f"  Glucose {glucose:3d} mg/dL → Mean amplitude: {amp_mean:.4f}")

# Test 3: Check glucose sensitivity
print("\n[TEST 3] Checking glucose sensitivity...")
gen_low = Spectrum_Generator(glucose_level=70)
gen_high = Spectrum_Generator(glucose_level=180)
gen_low.generate()
gen_high.generate()

diff = np.max(np.abs(gen_high._amplitude - gen_low._amplitude))
print(f"  Amplitude difference (70→180 mg/dL): {diff:.4f}")
if diff > 0.01:
    print(f"  ✓ Glucose sensitivity detected")
else:
    print(f"  ⚠ Glucose sensitivity low - may need adjustment")

# Test 4: Visualizations
print("\n[TEST 4] Generating visualizations...")
print("  Creating spectrum plot...")
gen_100.plot_amplitude_spectrum(save_path="results/spectra/test_spectrum.png")
print("  ✓ Saved to results/spectra/test_spectrum.png")

print("\n[TEST 5] Comparing glucose levels...")
gen_100.compare_glucose_levels([70, 100, 150, 200])
print("  ✓ Comparison plot generated")

print("\n" + "=" * 60)
print("CHECKPOINT 2 TEST COMPLETE")
print("=" * 60)