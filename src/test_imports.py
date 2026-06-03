import sys
print("Python version:", sys.version)

packages = [
    "fastf1",
    "requests",
    "pandas",
    "numpy",
    "matplotlib",
    "xgboost",
    "sklearn",
    "streamlit",
    "shap"
]

print("Verifying imports:")
for package in packages:
    try:
        __import__(package)
        print(f"  [OK] {package}")
    except ImportError as e:
        print(f"  [FAIL] {package}: {e}")
