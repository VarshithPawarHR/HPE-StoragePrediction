from fastapi import HTTPException
import numpy as np
from db import collection



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
