import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
from dotenv import load_dotenv
import tensorflow as tf
from typing import Dict, Tuple
import pymongo
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise ValueError("MONGO_URL not set in .env")

client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
db = client["storage_simulation"]
collection = db["usage_logs"]

# Test MongoDB connection
try:
    client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection failed: {e}")

# Configuration
tf.keras.mixed_precision.set_global_policy('mixed_float16')
HORIZONS = {
    '3_month': 540,  # 30 days (6*30=180)
}
SEQ_LENGTH = 42  # 7 days of historical data
BATCH_SIZE = 256
EPOCHS = 50


def load_and_preprocess_data() -> Dict[str, dict]:
    """Load and preprocess data with proper feature engineering"""
    raw_data = pd.DataFrame(list(collection.find()))
    raw_data = raw_data.drop(columns=['_id'])
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

    print("\n🔍 Data Diagnostics:")
    print(f"Total records: {len(raw_data)}")
    print("Unique directories:", raw_data['directory'].unique())

    processed = {}
    for directory in raw_data['directory'].unique():
        df = raw_data[raw_data['directory'] == directory].copy()
        df = df.sort_values('timestamp').set_index('timestamp')

        # Resample to 4-hour intervals
        df = df[['storage_gb']].resample('4h').mean().ffill()

        # Feature engineering
        df['hour'] = df.index.hour
        df['time_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
        df['time_cos'] = np.cos(2 * np.pi * df.index.hour / 23)

        # Scale storage_gb
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
    """Create sequences with validation"""
    X, y = [], []
    max_start = len(features) - seq_length - horizon
    if max_start < 0:
        return np.array([]), np.array([])

    for i in range(max_start + 1):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length:i + seq_length + horizon])
    return np.array(X), np.array(y)


def build_model(input_shape: Tuple[int, int], output_steps: int):
    """Optimized forecasting model architecture"""
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Temporal pattern extraction
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    x = tf.keras.layers.GRU(64)(x)

    # Prediction head
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(output_steps)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model


def save_model_and_scaler(model, scaler, name):
    """Save trained model and scaler to disk"""
    current_dir = os.getcwd()

    # Ensure name is safe for filesystem
    safe_name = os.path.basename(name)
    safe_name = safe_name.replace('/', '_').replace('\\', '_')

    # Create models directory
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f"{safe_name}_3_monthly_forecast_model.keras")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    # Create scalers directory
    scalers_dir = os.path.join(current_dir, 'scalers')
    os.makedirs(scalers_dir, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(scalers_dir, f"{safe_name}_3_monthly_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")


def train_and_evaluate(data_dict: Dict) -> Tuple[Dict, Dict]:
    """Enhanced training with proper validation"""
    models = {}
    metrics = {}

    for name, data in data_dict.items():
        print(f"\n⚡ Processing {name}")
        df = data['data']
        scaler = data['scaler']

        # Prepare data
        total_points = len(df)
        test_size = HORIZONS['3_month'] + SEQ_LENGTH
        split_idx = total_points - test_size

        if split_idx < SEQ_LENGTH:
            print(f"⚠️ Insufficient data for {name}")
            continue

        # Create sequences
        X_train, y_train = create_sequences(
            df.values[:split_idx],
            df['scaled_gb'].values[:split_idx],
            SEQ_LENGTH, HORIZONS['3_month']
        )
        X_test, y_test = create_sequences(
            df.values[split_idx:],
            df['scaled_gb'].values[split_idx:],
            SEQ_LENGTH, HORIZONS['3_month']
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"🚫 Sequence creation failed for {name}")
            continue

        # Model setup and training
        model = build_model((SEQ_LENGTH, 3), HORIZONS['3_month'])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(f'best_{name}.keras', save_best_only=True)
            ],
            verbose=1
        )

        # Generate predictions and calculate metrics
        test_pred = model.predict(X_test)
        metrics[name] = {}

        # Calculate metrics for each horizon
        for horizon_name, steps in HORIZONS.items():
            preds = test_pred[:, :steps].reshape(-1, 1)
            true = y_test[:, :steps].reshape(-1, 1)

            # Inverse transform predictions
            preds_gb = scaler.inverse_transform(preds).reshape(-1, steps)
            true_gb = scaler.inverse_transform(true).reshape(-1, steps)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(true_gb, preds_gb))

            metrics[name][horizon_name] = {
                'rmse': rmse,
                'predictions': preds_gb[0],
                'true': true_gb[0]
            }

        models[name] = model
        save_model_and_scaler(model, scaler, name)

    return models, metrics


def main():
    """Main execution function"""
    print("🚀 Starting storage forecasting model training...")

    # Load and preprocess data
    data_dict = load_and_preprocess_data()

    # Train models
    models, metrics = train_and_evaluate(data_dict)

    # Print performance metrics
    print("\n📊 TRAINING RESULTS")
    print("=" * 50)

    for directory in data_dict:
        if directory in metrics:
            print(f"\n{directory.upper()} PERFORMANCE:")
            for horizon in HORIZONS:
                if horizon in metrics[directory]:
                    rmse = metrics[directory][horizon]['rmse']
                    print(f"  {horizon.replace('_', ' ').title():<12} RMSE: {rmse:.2f} GB")

    print(f"\n✅ Training completed! Models and scalers saved successfully.")
    print(f"📁 Models saved in: ./models/")
    print(f"📁 Scalers saved in: ./scalers/")


if __name__ == "__main__":
    main()
