import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib
import os
from typing import Dict, Tuple
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

# Load environment and MongoDB
load_dotenv()
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise ValueError("MONGO_URL not set in .env")

client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
db = client["storage_simulation"]
collection = db["usage_logs"]

# Configurations
tf.keras.mixed_precision.set_global_policy('mixed_float16')
HORIZONS = {'1_week': 42}
SEQ_LENGTH = 42
BATCH_SIZE = 256
EPOCHS = 50

def load_and_preprocess_data() -> Dict[str, dict]:
    raw_data = pd.DataFrame(list(collection.find()))
    raw_data = raw_data.drop(columns=['_id'])
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

def create_sequences(features: np.ndarray, targets: np.ndarray,
                    seq_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
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
    outputs =tf.keras.layers. Dense(output_steps)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model

def save_model_and_scaler(model, scaler, name):
    base_dir = os.getcwd()
    safe_name = os.path.basename(name).replace('/', '_').replace('\\', '_')

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, f"{safe_name}_weekly_forecast_model.keras"))

    scalers_dir = os.path.join(base_dir, 'scalers')
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scalers_dir, f"{safe_name}_weekly_scaler.pkl"))

def train_and_evaluate(data_dict: Dict) -> Dict:
    metrics = {}

    for name, data in data_dict.items():
        df = data['data']
        scaler = data['scaler']

        total_points = len(df)
        test_size = HORIZONS['1_week'] + SEQ_LENGTH
        split_idx = total_points - test_size

        if split_idx < SEQ_LENGTH:
            print(f"Insufficient data for {name}")
            continue

        X_train, y_train = create_sequences(
            df.values[:split_idx],
            df['scaled_gb'].values[:split_idx],
            SEQ_LENGTH, HORIZONS['1_week']
        )
        X_test, y_test = create_sequences(
            df.values[split_idx:],
            df['scaled_gb'].values[split_idx:],
            SEQ_LENGTH, HORIZONS['1_week']
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Sequence creation failed for {name}")
            continue

        model = build_model((SEQ_LENGTH, 3), HORIZONS['1_week'])
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)],
            verbose=1
        )

        test_pred = model.predict(X_test)
        metrics[name] = {}

        for horizon_name, steps in HORIZONS.items():
            preds = test_pred[:, :steps].reshape(-1, 1)
            true = y_test[:, :steps].reshape(-1, 1)

            preds_gb = scaler.inverse_transform(preds).reshape(-1, steps)
            true_gb = scaler.inverse_transform(true).reshape(-1, steps)

            rmse = np.sqrt(mean_squared_error(true_gb, preds_gb))
            metrics[name][horizon_name] = {'rmse': rmse}

        save_model_and_scaler(model, scaler, name)

    return metrics

def main():
    data_dict = load_and_preprocess_data()
    metrics = train_and_evaluate(data_dict)

    for directory in data_dict:
        if directory in metrics:
            print(f"\n{directory.upper()} PERFORMANCE")
            for horizon in HORIZONS:
                if horizon in metrics[directory]:
                    rmse = metrics[directory][horizon]['rmse']
                    print(f"{horizon.replace('_', ' ').title():<12} RMSE: {rmse:.2f} GB")

if __name__ == "__main__":
    main()
