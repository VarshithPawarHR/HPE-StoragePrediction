# Utility functions for preprocessing input data for the model

from fastapi import HTTPException
import numpy as np
from db import collection
import pandas as pd


async def preprocess_input_daily(directory: str, scaler):
    """
    preprocess_input_daily:
    - Fetches the latest 96 15-minute interval storage_gb values (~1 day) for the given directory.
    - Ensures there are enough data points; raises HTTPException if not.
    - Converts the records to ascending order (oldest to newest).
    - Scales the storage_gb values using the provided scaler.
    - Reshapes the scaled values to (1, 96, 1) for model input.
    """
    cursor = collection.find(
        {"directory": f"/{directory}"},
        {"_id": 0, "storage_gb": 1}
    ).sort("timestamp", -1).limit(96)
    
    records = await cursor.to_list(length=96)

    if len(records) < 96:
        raise HTTPException(status_code=400, detail=f"Not enough data for {directory} (found: {len(records)}, required: 96)")

    values = [r["storage_gb"] for r in reversed(records)]
    values_array = np.array(values).reshape(-1, 1)
    scaled_values = scaler.transform(values_array)
    X_input = scaled_values.reshape(1, 96, 1)

    return X_input


async def preprocess_input(directory: str, scaler):
    """
    Preprocess input data for the model:
    - Fetch 672 latest 15-minute data points (~7 days) for the given directory.
    - Aggregate to 4-hour intervals using mean, resulting in 42 points.
    - Add cyclical time features (sin/cos of hour).
    - Scale storage_gb values.
    - Return input reshaped to (1, 42, 3) for the model.
    """
    cursor = collection.find(
        {"directory": f"/{directory}"},
        {"_id": 0, "timestamp": 1, "storage_gb": 1}
    ).sort("timestamp", -1).limit(672)

    records = await cursor.to_list(length=672)

    if len(records) < 672:
        raise HTTPException(status_code=400, detail=f"Not enough data for {directory} (found: {len(records)}, required: 672)")

    records = list(reversed(records))

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df_agg = df.resample("4H").mean()

    if len(df_agg) < 42:
        raise HTTPException(status_code=400, detail=f"Not enough aggregated data for {directory} (found: {len(df_agg)}, required: 42)")

    df_agg = df_agg.tail(42)
    df_agg['hour'] = df_agg.index.hour
    df_agg['time_sin'] = np.sin(2 * np.pi * df_agg['hour'] / 23)
    df_agg['time_cos'] = np.cos(2 * np.pi * df_agg['hour'] / 23)
    df_agg['scaled_gb'] = scaler.transform(df_agg[['storage_gb']])

    features = df_agg[['scaled_gb', 'time_sin', 'time_cos']].values
    X_input = features.reshape(1, 42, 3)

    return X_input


def reshape_input(input_data):
    """
    Reshape the input to the required shape (1, 42, 3).
    Assuming input_data is a 1D array (e.g., shape (1, 9)).
    :param input_data: The raw input data.
    :return: The reshaped input data.
    """
    input_data = np.array(input_data)
    
    if input_data.size > 126:
        input_data = input_data.flatten()[:126]
    
    elif input_data.size < 126:
        padding_length = 126 - input_data.size
        input_data = np.pad(input_data.flatten(), (0, padding_length), mode='constant')
   
    reshaped_input = input_data.reshape(1, 42, 3)
    return reshaped_input


