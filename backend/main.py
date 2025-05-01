import pandas as pd
from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from Utils.loader import load_keras_models, load_scalers
from typing import Dict, Any
import asyncio
from db import collection
from pymongo import DESCENDING
from fastapi.responses import JSONResponse
from pymongo import ASCENDING
from typing import List
from Utils.loader import load_keras_models, load_scalers
from Utils.preprocess import preprocess_input_daily
from fastapi import HTTPException
import numpy as np
from zoneinfo import ZoneInfo
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from Utils.model_ops import update_model_for_directory


IST = ZoneInfo("Asia/Kolkata")
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

#APIs for real time dashboard

@app.get("/summary")
async def get_directory_summary(
    directory: str = Query(..., description="Directory name, e.g. /scratch")
):
    cursor = collection.find(
        {"directory": directory}
    ).sort("timestamp", DESCENDING).limit(1)

    latest_entries = await cursor.to_list(length=1)

    for entry in latest_entries:
        entry.pop("_id", None)

    return {
        "directory": directory,
        "summary": latest_entries
     }

# APIs for historical data and please directories are named as /info,/customer do not pass info,customer like this

@app.get("/directory-usage")
async def get_directory_usage(
    directory: str = Query(..., description="Directory name, e.g. /scratch")
):
    cursor = collection.find({
        "directory": directory
    }).sort("timestamp", DESCENDING).limit(96)  # Fetch last 96 entries

    results = await cursor.to_list(length=96)  # Fetch 96 entries from the database

    # Format the results
    formatted = [
        {
            "timestamp": doc["timestamp"].strftime('%Y-%m-%dT%H:%M:%S%z'),  # Format the timestamp to show in required format
            "directory": doc["directory"],
            "storage_gb": doc["storage_gb"]
        }
        for doc in results
    ]

    return JSONResponse({
        "directory": directory,
        "data": formatted
    })

@app.get("/predictions/daily")
async def get_predictions():

    directories = ["info", "scratch", "customer", "projects"]
    results = {}

    for directory in directories:
        model_name = f"{directory}_daily"
        # Fetch the model and scaler from the app state
        model = app.state.models.get(model_name)
        scaler = app.state.scalers.get(model_name)

        if model is None or scaler is None:
            results[directory] = None
            continue

        
        input = await preprocess_input_daily(directory, scaler)

        if input is None:
            results[directory] = None
            continue

        
        pred_scaled = model.predict(input)  # shape: (1, 1)

        # Step 3: Inverse transform to get the value in GB
        pred_original = scaler.inverse_transform(pred_scaled)
        results[directory] = round(float(pred_original), 2)

    return results




#growth rate endpoint logic is get the last 96th data entry then firstentry - last entry / lastentry No need for timestamps retreval from DB

@app.get("/growth-rate")
async def get_growth_rate(
    directory: str = Query(..., description="Directory name, e.g. /scratch")
):
    cursor = collection.find({
        "directory": directory
    }).sort("timestamp", DESCENDING).limit(96)

    results = await cursor.to_list(length=96)

    if len(results) < 2:
        return JSONResponse(
            {"error": "Not enough data to calculate growth rate."}, status_code=400
        )

    first_entry = results[0]["storage_gb"]
    last_entry = results[-1]["storage_gb"]

    if last_entry == 0:
        return JSONResponse(
            {"error": "Last entry storage is 0, cannot divide by zero."}, status_code=400
        )

    growth_rate = (first_entry - last_entry)  / last_entry
    growth_rate = growth_rate*100  # Convert to percentage

    return JSONResponse({
        "directory": directory,
        "first_entry": first_entry,
        "last_entry": last_entry,
        "growth_rate_percent": round(growth_rate, 2)
    })

#consumpton endpoint get the 96th data entry from start then latest- 96th entry

@app.get("/total-consumption")
async def get_total_storage_consumption(
    directory: str = Query(..., description="Directory name, e.g. /scratch")
):
    cursor = collection.find(
        {"directory": directory}
    ).sort("timestamp", DESCENDING).limit(96)

    results = await cursor.to_list(length=96)

    if len(results) < 2:
        return JSONResponse(
            {"error": "Not enough data to calculate total storage consumption."}, status_code=400
        )

    current_storage = results[0]["storage_gb"]   # Newest entry
    oldest_storage = results[-1]["storage_gb"]   # Oldest entry

    total_consumed = current_storage - oldest_storage

    return JSONResponse({
        "directory": directory,
        "initial_storage": oldest_storage,
        "current_storage": current_storage,
        "total_storage_consumed_gb": round(total_consumed, 2)
    })



#this is for pie chart

@app.get("/predictions/current")
async def get_current_storage():
    # Get distinct directory names
    directories = await collection.distinct("directory")

    result = {}

    for directory in directories:
        # Fetch latest entry for each directory
        cursor = collection.find({"directory": directory}).sort("timestamp", DESCENDING).limit(1)
        latest = await cursor.to_list(length=1)

        if latest:
            result[directory.strip("/")] = latest[0]["storage_gb"]  # remove leading slash if you want clean keys

    return JSONResponse(result)


@app.post("/retrain/daily")
async def retrain_all_daily_models():
    DIRECTORIES = ["info", "customer", "scratch", "projects"]
    models = load_keras_models()
    scalers = load_scalers()

    cursor = collection.find({}, {"_id": 0, "timestamp": 1, "directory": 1, "storage_gb": 1})
    df = pd.DataFrame(await cursor.to_list(None))

    if df.empty:
        return {"status": "error", "message": "No data found in MongoDB collection"}

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    results = {}
    for dir_name in DIRECTORIES:
        key = f"{dir_name}_daily"  # for accessing models and scalers
        dir_df = df[df["directory"] == f"/{dir_name}"].copy()

        try:
            result = update_model_for_directory(dir_name, dir_df, models[key], scalers[key])
            results[dir_name] = result
        except Exception as e:
                results[dir_name] = f"❌ Error: {str(e)}"

    return {
        "status": "completed",
        "retrain_results": results
    }
