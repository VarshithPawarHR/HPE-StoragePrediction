import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


SCALER_DIRS = [
    BASE_DIR / "scalers",
    NOTEBOOKS_DIR / "scalers"
]

SEQUENCE_LENGTH = 96

def create_sequences_singlestep(data):
    X, y = [], []
    for i in range(len(data) - SEQUENCE_LENGTH):
        X.append(data[i:i + SEQUENCE_LENGTH])
        y.append(data[i + SEQUENCE_LENGTH])
    return np.array(X), np.array(y)


def update_model_for_directory(dir_name, df, model, scaler, evaluate_only=False):
    print(f"🔁 Updating model for: {dir_name}")
    
    df.set_index('timestamp', inplace=True)
    df = df[['storage_gb']].resample('D').mean()

    scaled_data = scaler.transform(df.values).flatten()

    X, y = create_sequences_singlestep(scaled_data)
    if len(X) == 0:
        return f"⚠️ Not enough data to update model for {dir_name}"

    X = X.reshape((X.shape[0], SEQUENCE_LENGTH, 1))
    y = y.reshape(-1, 1)

    update_size = min(100, len(X))
    old_size = min(50, len(X) - update_size)

    X_new, y_new = X[-update_size:], y[-update_size:]
    X_old, y_old = X[:old_size], y[:old_size] if old_size > 0 else ([], [])

    X_train = np.concatenate([X_old, X_new]) if old_size > 0 else X_new
    y_train = np.concatenate([y_old, y_new]) if old_size > 0 else y_new

    # Evaluate RMSE before retraining
    y_pred_before = model.predict(X_train, verbose=0)
    rmse_before = sqrt(mean_squared_error(
        scaler.inverse_transform(y_train),
        scaler.inverse_transform(y_pred_before)
    ))

    if evaluate_only:
        return f"📉 Pre-Retrain RMSE for {dir_name}: {rmse_before:.4f}"

    # Retrain model
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    # Evaluate RMSE after retraining
    y_pred_after = model.predict(X_train, verbose=0)
    rmse_after = sqrt(mean_squared_error(
        scaler.inverse_transform(y_train),
        scaler.inverse_transform(y_pred_after)
    ))

    # Save only if RMSE improves
    model_path = MODEL_DIR / f"{dir_name}_daily_forecast_model.keras"
    model.save(model_path)

    return (f"✅ Model retrained and saved for {dir_name} | "
            f"RMSE: {rmse_before:.4f} → {rmse_after:.4f}")
