from fastapi import HTTPException
import numpy as np
from db import collection
import pandas as pd
from datetime import datetime, timedelta


async def preprocess_input_daily(directory: str,  scaler):
    
    cursor = collection.find(
        {"directory": f"/{directory}"},
        {"_id": 0, "storage_gb": 1}
    ).sort("timestamp", -1).limit(96)
    
    records = await cursor.to_list(length=96)

    if len(records) < 96:
        raise HTTPException(status_code=400, detail=f"Not enough data for {directory} (found: {len(records)}, required: 96)")

    # Convert to ascending order and extract values
    values = [r["storage_gb"] for r in reversed(records)]
    values_array = np.array(values).reshape(-1, 1)

    # Scale and reshape to (1, 96, 1)
    scaled_values = scaler.transform(values_array)
    X_input = scaled_values.reshape(1, 96, 1)

    return X_input




async def preprocess_input(directory: str, scaler):
    # Fetch 672 latest 15-minute data points (~7 days)
    cursor = collection.find(
        {"directory": f"/{directory}"},
        {"_id": 0, "timestamp": 1, "storage_gb": 1}
    ).sort("timestamp", -1).limit(672)

    records = await cursor.to_list(length=672)

    if len(records) < 672:
        raise HTTPException(status_code=400, detail=f"Not enough data for {directory} (found: {len(records)}, required: 672)")

    # Reverse to ascending time order
    records = list(reversed(records))

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Step 1: Resample to 4-hour intervals using mean
    df_agg = df.resample("4H").mean()

    # Ensure we have exactly 42 aggregated points
    if len(df_agg) < 42:
        raise HTTPException(status_code=400, detail=f"Not enough aggregated data for {directory} (found: {len(df_agg)}, required: 42)")

    df_agg = df_agg.tail(42)

    # Step 2: Time features
    df_agg['hour'] = df_agg.index.hour
    df_agg['time_sin'] = np.sin(2 * np.pi * df_agg['hour'] / 23)
    df_agg['time_cos'] = np.cos(2 * np.pi * df_agg['hour'] / 23)

    # Step 3: Scale storage_gb
    df_agg['scaled_gb'] = scaler.transform(df_agg[['storage_gb']])

    # Step 4: Create final input
    features = df_agg[['scaled_gb', 'time_sin', 'time_cos']].values
    X_input = features.reshape(1, 42, 3)

    return X_input

