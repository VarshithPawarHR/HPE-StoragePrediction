from pathlib import Path
import joblib
from tensorflow import keras

# Go up three levels from loader.py to reach project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

MODEL_DIRS = [
    BASE_DIR / "models",
    NOTEBOOKS_DIR / "models"
]

SCALER_DIRS = [
    BASE_DIR / "scalers",
    NOTEBOOKS_DIR / "scalers"
]

print("=== MODEL DIRECTORIES ===")
for d in MODEL_DIRS:
    print(f"{d.exists()} - {d}")

print("=== SCALER DIRECTORIES ===")
for d in SCALER_DIRS:
    print(f"{d.exists()} - {d}")

def load_keras_models():
    models = {}
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            continue
        for file in model_dir.glob("*.keras"):
            key = file.name.replace("_forecast_model.keras", "").lower()
            print(f" Found model file: {file.name} in {model_dir}")
            try:
                models[key] = keras.models.load_model(file)
                print(f"✅ Loaded model: {file.name}")
            except Exception as e:
                print(f"❌ Error loading model {file.name}: {e}")
    return models

def load_scalers():
    scalers = {}
    for scaler_dir in SCALER_DIRS:
        if not scaler_dir.exists():
            continue
        for file in scaler_dir.glob("*.pkl"):
            key = file.name.replace("_scaler.pkl", "").lower()
            print(f"Found scaler file: {file.name} in {scaler_dir}")
            try:
                scalers[key] = joblib.load(file)
                print(f"✅ Loaded scaler: {file.name}")
            except Exception as e:
                print(f"❌ Error loading scaler {file.name}: {e}")
    return scalers
