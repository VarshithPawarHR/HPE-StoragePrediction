import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
from typing import Dict, Tuple


load_dotenv()
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise ValueError("MONGO_URL not set in .env")

client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
db = client["storage_simulation"]
collection = db["usage_logs"]

tf.keras.mixed_precision.set_global_policy('mixed_float16')
HORIZONS = {'1_month': 180}
SEQ_LENGTH = 42
BATCH_SIZE = 256
EPOCHS = 50

def load_and_preprocess_data() -> Dict[str, dict]:
    raw_data = pd.DataFrame(list(collection.find())).drop(columns=['_id'])
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

    processed = {}
    for directory in raw_data['directory'].unique():
        df = raw_data[raw_data['directory'] == directory].copy()
        df = df.sort_values('timestamp').set_index('timestamp')

        df = df[['storage_gb']].resample('4h').mean().ffill()
        df['hour'] = df.index.hour
        df['time_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
        df['time_cos'] = np.cos(2 * np.pi * df.index.hour / 23)

        scaler = MinMaxScaler()
        df['scaled_gb'] = scaler.fit_transform(df[['storage_gb']])

        processed[directory] = {
            'data': df[['scaled_gb', 'time_sin', 'time_cos']],
            'original': df['storage_gb'],
            'scaler': scaler
        }
    return processed

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    max_start = len(features) - seq_length - horizon
    if max_start < 0:
        return np.array([]), np.array([])

    for i in range(max_start + 1):
        X.append(features[i:i+seq_length])
        y.append(targets[i+seq_length:i+seq_length+horizon])
    return np.array(X), np.array(y)

def build_model(input_shape: Tuple[int, int], output_steps: int) :
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(output_steps)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model

def save_model_and_scaler(model, scaler, name):
    base_dir = os.getcwd()
    safe_name = os.path.basename(name).replace('/', '_').replace('\\', '_')

    model_path = os.path.join(base_dir, 'models', f"{safe_name}_monthly_forecast_model.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    scaler_path = os.path.join(base_dir, 'scalers', f"{safe_name}_monthly_scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

def train_and_evaluate(data_dict: Dict) -> None:
    for name, data in data_dict.items():
        print(f"\nTraining model for: {name}")
        df = data['data']
        scaler = data['scaler']

        total_points = len(df)
        test_size = HORIZONS['1_month'] + SEQ_LENGTH
        split_idx = total_points - test_size

        if split_idx < SEQ_LENGTH:
            print(f"Not enough data for {name}, skipping.")
            continue

        X_train, y_train = create_sequences(
            df.values[:split_idx],
            df['scaled_gb'].values[:split_idx],
            SEQ_LENGTH, HORIZONS['1_month']
        )
        X_test, y_test = create_sequences(
            df.values[split_idx:],
            df['scaled_gb'].values[split_idx:],
            SEQ_LENGTH, HORIZONS['1_month']
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Sequence creation failed for {name}")
            continue

        model = build_model((SEQ_LENGTH, 3), HORIZONS['1_month'])
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
                tf.keras.callbacks. ModelCheckpoint(f'best_{name}.keras', save_best_only=True)
            ],
            verbose=0
        )

        test_pred = model.predict(X_test)
        preds = test_pred[:, :HORIZONS['1_month']].reshape(-1, 1)
        true = y_test[:, :HORIZONS['1_month']].reshape(-1, 1)

        preds_gb = scaler.inverse_transform(preds).reshape(-1, HORIZONS['1_month'])
        true_gb = scaler.inverse_transform(true).reshape(-1, HORIZONS['1_month'])

        rmse = np.sqrt(mean_squared_error(true_gb, preds_gb))
        print(f"RMSE for {name}: {rmse:.2f} GB")

        save_model_and_scaler(model, scaler, name)

def run_training_pipeline():
    print("Loading data and starting training...")
    data_dict = load_and_preprocess_data()
    train_and_evaluate(data_dict)
    print("Training complete.")

if __name__ == "__main__":
    run_training_pipeline()
