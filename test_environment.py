import numpy
import scipy
import pandas
import matplotlib
import sklearn
import pywt
import skimage
import plotly
import streamlit
import torch

print("✓ Environment working!")
print()
print("Package Versions:")
print("=" * 50)
print(f"NumPy:        {numpy.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"Pandas:       {pandas.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"PyWavelets:   {pywt.__version__}")
print(f"scikit-image: {skimage.__version__}")
print(f"Plotly:       {plotly.__version__}")
print(f"Streamlit:    {streamlit.__version__}")
print(f"PyTorch:      {torch.__version__}")
print("=" * 50)
print()
print("PyTorch GPU Available:", torch.cuda.is_available())
print("(False is expected - we're using CPU)")
print()
print("✓ All imports successful!")
print("✓ Ready for THz Glucose Prototype development")