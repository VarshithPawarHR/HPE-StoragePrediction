import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Dict

load_dotenv()
client = os.getenv("MONGO_URL")
client = MongoClient(client)
db = client["storage_simulation"]
collection = db["usage_logs"]

def load_and_preprocess_data() -> Dict[str, pd.DataFrame]:
    raw_data = pd.DataFrame(list(collection.find()))
    raw_data = raw_data.drop(columns=['_id'])
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

    processed = {}
    for directory in raw_data['directory'].unique():
        df = raw_data[raw_data['directory'] == directory].copy()
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df[['storage_gb']]
        processed[directory] = df
    return processed

def create_sequences_singlestep(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model_singlestep(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.25),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_single_step_forecast_model(df, dir_name, sequence_length=96):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values).flatten()

    X, y = create_sequences_singlestep(scaled_data, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1, 1)

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model_singlestep(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = model.predict(X_test)
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    data_range = y_test_original.max() - y_test_original.min()
    nmae = mae / data_range
    nrmse = rmse / data_range

    print(f"{dir_name.upper()} - MAE: {mae:.2f} GB")
    print(f"{dir_name.upper()} - RMSE: {rmse:.2f} GB")
    print(f"{dir_name.upper()} - Normalized MAE: {nmae:.4f}")
    print(f"{dir_name.upper()} - Normalized RMSE: {nrmse:.4f}")

    safe_name = dir_name.strip("/")

    models = os.path.abspath(os.path.join(os.getcwd(), '..', 'models'))
    os.makedirs(models, exist_ok=True)
    model.save(os.path.join(models, f"{safe_name}_daily_forecast_model.keras"))

    scalers = os.path.abspath(os.path.join(os.getcwd(), '..', 'scalers'))
    os.makedirs(scalers, exist_ok=True)
    scaler_file_path = os.path.join(scalers, f"{safe_name}_daily_scaler.pkl")
    joblib.dump(scaler, scaler_file_path)

data_dict = load_and_preprocess_data()
for name, data in data_dict.items():
    train_single_step_forecast_model(data, name)
