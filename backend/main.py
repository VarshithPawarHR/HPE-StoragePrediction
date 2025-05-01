from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from Utils.loader import load_keras_models, load_scalers
from typing import Dict, Any
import asyncio
from db import collection
from pymongo import DESCENDING
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
from pymongo import ASCENDING
from typing import List
from Utils.loader import load_keras_models, load_scalers
from Utils.preprocess import preprocess_input
from fastapi import HTTPException
import numpy as np

models = load_keras_models()
scalers = load_scalers()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load models and scalers
        models: Dict[str, Any] = await asyncio.to_thread(load_keras_models)
        scalers: Dict[str, Any] = await asyncio.to_thread(load_scalers)

        app.state.models = models
        app.state.scalers = scalers

        print("Models loaded:", list(models.keys()))
        print("Scalers loaded:", list(scalers.keys()))

    except Exception as e:
        print(f"Error during startup: {e}")
        raise

    yield

    print("Application is shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}

#APIs for real time dashboard

@app.get("/summary")
async def get_summary():
    cursor = collection.find().sort("timestamp", DESCENDING).limit(4)
    latest_entries = await cursor.to_list(length=4)

    #_id beda if any use there please don't change here
    for entry in latest_entries:
        entry.pop("_id", None)

    return {
        "summary": latest_entries
    }

# APIs for historical data and please directories are named as /info,/customer do not pass info,customer like this

@app.get("/directory-usage")
async def get_directory_usage(
    directory: str = Query(..., description="Directory name, e.g. /scratch"),
    start: datetime = Query(..., description="Start time (ISO format)"),
    end: datetime = Query(..., description="End time (ISO format)")
):
    cursor = collection.find({
        "directory": directory,
        "timestamp": {
            "$gte": start,
            "$lte": end
        }
    }).sort("timestamp", ASCENDING)

    results = await cursor.to_list(length=1000)  # adjust if needed

    formatted = [
        {
            "timestamp": doc["timestamp"],
            "directory": doc["directory"],
            "storage_gb": doc["storage_gb"]
        }
        for doc in results
    ]

    return JSONResponse({
        "directory": directory,
        "start": start,
        "end": end,
        "data": formatted
    })

@app.get("/predictions/{horizon}")
async def get_predictions(horizon: str):

    directories = ["info", "scratch", "customer", "projects"]
    results = {}

    for directory in directories:
        model_name = f"{directory}_{horizon}"
        model = models.get(model_name)
        scaler = scalers.get(model_name)

        if model is None or scaler is None:
            results[directory] = None
            continue

        
        input = await preprocess_input(directory, scaler, horizon)

        if input is None:
            results[directory] = None
            continue

        
        pred_scaled = model.predict(input)  # shape: (1, 1)

        # Step 3: Inverse transform to get the value in GB
        pred_original = scaler.inverse_transform(pred_scaled)
        results[directory] = round(float(pred_original), 2)

    return results





