from pathlib import Path
import joblib
from tensorflow import keras

# Go up THREE levels from loader.py to reach project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scalers"

print(f" BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR.exists()} - {MODEL_DIR}")
print(f"SCALER_DIR: {SCALER_DIR.exists()} - {SCALER_DIR}")

def load_keras_models():
    models = {}
    for file in MODEL_DIR.glob("*.keras"):
        key = file.name.replace("_forecast_model.keras", "").lower()
        print(f" Found model file: {file.name}")
        try:
            models[key] = keras.models.load_model(file)
            print(f"Loaded model: {file.name}")
        except Exception as e:
            print(f"Error loading model {file.name}: {e}")
    return models

def load_scalers():
    scalers = {}
    for file in SCALER_DIR.glob("*.pkl"):
        key = file.name.replace("_scaler.pkl", "").lower()
        print(f"Found scaler file: {file.name}")
        try:
            scalers[key] = joblib.load(file)
            print(f"Loaded scaler: {file.name}")
        except Exception as e:
            print(f"Error loading scaler {file.name}: {e}")
    return scalers
