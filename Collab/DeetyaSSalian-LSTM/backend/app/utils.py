import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.dropna(subset=['timestamp'], inplace=True)
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], errors='coerce')
    raw_data.dropna(subset=['timestamp'], inplace=True)
    raw_data.set_index('timestamp', inplace=True)
    raw_data.dropna(inplace=True)
    return raw_data

def naive_predict(df: pd.DataFrame, shift: int) -> float:
    df = df.copy()
    df['target'] = df['storage_gb'].shift(-shift)
    df.dropna(inplace=True)
    if len(df) < 200:
        return None
    test = df.iloc[int(len(df)*0.8):]
    y_true = test['target']
    y_pred = test['storage_gb']
    mae = mean_absolute_error(y_true, y_pred)
    predicted_value = y_true.mean()  # use mean true value
    return predicted_value
