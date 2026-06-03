
# Test all installations
print("Testing installations...")

# Test Python version
import sys
print(f"✅ Python version: {sys.version}")

# Test TensorFlow
import tensorflow  
# import tensorflow as tf
print(f"✅ TensorFlow version: {tensorflow.__version__}")

# Test PyTorch
import torch 
print(f"✅ PyTorch version: {torch.__version__}")

# Test Transformers
import transformers
print(f"✅ Transformers version: {transformers.__version__}")

# Test Pandas
import pandas as pd
print(f"✅ Pandas version: {pd.__version__}")

# Test NumPy
import numpy as np
print(f"✅ NumPy version: {np.__version__}")

# Test Scikit-learn
import sklearn
print(f"✅ Scikit-learn version: {sklearn.__version__}")

# Test SHAP
import shap
print(f"✅ SHAP version: {shap.__version__}")

# Test LIME
# import lime
# print(f"✅ LIME version: {lime.__version__}")
try:
    import lime
    print(f"✅ LIME installed successfully")
except:
    print(f"❌ LIME not found")

# Test Streamlit
import streamlit
print(f"✅ Streamlit version: {streamlit.__version__}")

print("\n🎉 All installations successful!")