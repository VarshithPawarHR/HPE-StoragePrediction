import uvicorn
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any
import asyncio
from db import collection
from pymongo import DESCENDING
from Utils.loader import load_keras_models, load_scalers
from Utils.preprocess import preprocess_input_daily, preprocess_input, reshape_input
from zoneinfo import ZoneInfo
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import JSONResponse
import pandas as pd
from collections import defaultdict
import subprocess

load_dotenv()
# Read from .env
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# Global variable to store directories
directories = []
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_PATH = os.path.join(BASE_DIR, "notebooks")


IST = ZoneInfo("Asia/Kolkata")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global directories
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGO_URL)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        directories_raw = await collection.distinct("directory")
        directories = [d.lstrip("/") for d in directories_raw]
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
    allow_origins=["*"],
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


@app.get("/directory-name")
async def get_recent_directories():
    # Fetch latest 4 documents sorted by timestamp
    cursor = collection.find().sort("timestamp", DESCENDING).limit(4)
    latest_entries = await cursor.to_list(length=4)
    # Extract only the directory names
    directories = [entry.get("directory") for entry in latest_entries]
    return {"recent_directories": directories}


# APIs for historical data and please directories are named as /info,/customer do not pass info,customer like this
@app.get("/directory-usage")
async def get_directory_usage(
    directory: str = Query(..., description="Directory name, e.g. /scratch")
):
    cursor = collection.find({
        "directory": directory
    }).sort("timestamp", DESCENDING).limit(96)
    results = await cursor.to_list(length=96)
    # Format the results
    formatted = [
        {
            "timestamp": doc["timestamp"].strftime('%Y-%m-%dT%H:%M:%S%z'),
            "directory": doc["directory"],
            "storage_gb": doc["storage_gb"]
        }
        for doc in results
    ]
    return JSONResponse({
        "directory": directory,
        "data": formatted
    })


#api endpoint for daily prediction for all directories predicts only the final value at next day
@app.get("/predictions/daily")
async def get_predictions():
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


#api endpoint for weekly prediction for all directories predicts the next 42 steps (7 days),
#and returns only the final predicted value (42nd step)
@app.get("/predictions/weekly")
async def get_weekly_predictions():
    results = {}
    for directory in directories:
        model_name = f"{directory}_weekly"
        model = app.state.models.get(model_name)
        scaler = app.state.scalers.get(model_name)
        if model is None or scaler is None:
            results[directory] = None
            continue
        X_input = await preprocess_input(directory, scaler)
        if X_input is None:
            results[directory] = None
            continue
        # Predict 7-day (42 steps) sequence
        pred_scaled = model.predict(X_input)  # shape: (1, 42)
        # Inverse transform the last predicted value (7th day = 42nd step)
        last_pred_scaled = pred_scaled[0, -1].reshape(-1, 1)
        last_pred_gb = scaler.inverse_transform(last_pred_scaled)
        results[directory] = round(float(last_pred_gb[0][0]), 2)
    return results


# api endpoint for monthly predictions for all directories
# for each directory (info, scratch, customer, projects) using their respective 1-month models.
# It prepares the latest input sequence, predicts the next 180 steps (30 days with 4-hour intervals),
# and returns only the final predicted value (180th step) 
@app.get("/predictions/monthly")
async def get_monthly_predictions():
    results = {}
    for directory in directories:
        model_name = f"{directory}_monthly"
        model = app.state.models.get(model_name)
        scaler = app.state.scalers.get(model_name)
        if model is None or scaler is None:
            results[directory] = None
            continue
        try:
            X_input = await preprocess_input(directory, scaler)
        except HTTPException:
            results[directory] = None
            continue
        pred_scaled = model.predict(X_input)  # (1, 180)
        pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1))  # (180, 1)
        # Return only the last (180th) value = prediction at end of 30th day
        last_value = round(float(pred_original[-1]), 2)
        results[directory] = last_value
    return results


# api endpoint for 3 months prediction for all directories
# It prepares the latest input sequence, predicts the next 540 time steps (3 months with 4-hour intervals),
# and returns only the final predicted value (540th step) in GB after inverse scaling.
@app.get("/predictions/3_months")
async def get_3_month_predictions():
    results = {}
    for directory in directories:
        model_name = f"{directory}_3_monthly"
        model = app.state.models.get(model_name)
        scaler = app.state.scalers.get(model_name)
        if model is None or scaler is None:
            results[directory] = None
            continue
        try:
            X_input = await preprocess_input(directory, scaler)
        except HTTPException:
            results[directory] = None
            continue
        # Predict
        pred_scaled = model.predict(X_input)  # Shape: (1, 540)
        pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1))  # (540, 1)
        # Get the 540th step (last value = end of 3rd month)
        last_value = round(float(pred_original[-1]), 2)
        results[directory] = last_value
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
    growth_rate = growth_rate*100
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
    current_storage = results[0]["storage_gb"]
    oldest_storage = results[-1]["storage_gb"]
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
    directories = await collection.distinct("directory")
    result = {}
    for directory in directories:
        cursor = collection.find({"directory": directory}).sort("timestamp", DESCENDING).limit(1)
        latest = await cursor.to_list(length=1)
        if latest:
            result[directory.strip("/")] = latest[0]["storage_gb"]
    return JSONResponse(result)


# API endpoints for retraining models
# These endpoints will run the respective Python scripts to retrain the models.
def run_python_script(script_name):
    script_path = f"{NOTEBOOKS_PATH}\\{script_name}"
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, check=True
        )
        return {"status": "success", "details": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "details": e.stderr}

@app.post("/retrain/daily")
def retrain_daily():
    return run_python_script("train_daily_model.py")

@app.post("/retrain/weekly")
def retrain_weekly():
    return run_python_script("train_weekly_model.py")

@app.post("/retrain/monthly")
def retrain_monthly():
    return run_python_script("train_monthly_model.py")

@app.post("/retrain/quarterly")
def retrain_quarterly():
    return run_python_script("train_3_months_model.py")


#line graph for predictions
@app.get("/predictions/weekly-line/{directory}")
async def get_weekly_line_predictions(directory: str):   
    if directory not in directories:
        return JSONResponse({"error": "Invalid directory"}, status_code=400)
    model_name = f"{directory}_weekly"
    model = app.state.models.get(model_name)
    scaler = app.state.scalers.get(model_name)
    if model is None or scaler is None:
        return JSONResponse({"error": f"Model or scaler for {directory} not found"}, status_code=404)
    input_data = await preprocess_input(directory, scaler)
    if input_data is None:
        return JSONResponse({"error": f"Failed to preprocess input for {directory}"}, status_code=400)
    input_data = reshape_input(input_data)
    pred_scaled = model.predict(input_data)
    pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    results = [{"predicted_value": round(float(val), 2)} for i, val in
               enumerate(pred_original.flatten())]
    return JSONResponse({directory: results})


# API endpoint to get overall prediction category based on quantiles
# This endpoint takes a dictionary of predictions and returns an overall category (low, moderate, high
@app.post("/predictions/category")
async def get_overall_prediction_category(predictions: dict):
    predictions = {f"/{k.lstrip('/')}" : v for k, v in predictions.items()}
    cursor = collection.find().sort("timestamp", -1).limit(2880)
    data = defaultdict(list)
    async for doc in cursor:
        data['timestamp'].append(pd.to_datetime(doc['timestamp']))
        data['directory'].append(doc['directory'])
        data['storage_gb'].append(doc['storage_gb'])
    df = pd.DataFrame(data)
    if df.empty:
        return JSONResponse({"error": "No data found in DB."}, status_code=404)
    labels = {}
    for dir_name in df['directory'].unique():
        subset = df[df['directory'] == dir_name]
        q1 = subset['storage_gb'].quantile(0.33)
        q2 = subset['storage_gb'].quantile(0.66)
        labels[dir_name] = {'low': q1, 'moderate': q2}

    label_scores = {'low': 1, 'moderate': 2, 'high': 3}
    scores = []
    for directory, predicted_value in predictions.items():
        if directory not in labels:
            return JSONResponse({"error": f"Invalid directory: {directory}"}, status_code=400)
        thresholds = labels[directory]
        if predicted_value <= thresholds['low']:
            category = 'low'
        elif predicted_value <= thresholds['moderate']:
            category = 'moderate'
        else:
            category = 'high'
        scores.append(label_scores[category])
    avg_score = sum(scores) / len(scores)
    if avg_score <= 1.5:
        overall_category = 'low'
    elif avg_score <= 2.3:
        overall_category = 'moderate'
    else:
        overall_category = 'high'
    return JSONResponse({
        "overall_category": overall_category
    })

# API endpoints for line graph predictions
@app.get("/predictions/monthly-line/{directory}")
async def get_monthly_line_predictions(directory: str):
    if directory not in directories:
        return JSONResponse({"error": "Invalid directory"}, status_code=400)
    model_name = f"{directory}_monthly"
    model = app.state.models.get(model_name)
    scaler = app.state.scalers.get(model_name)
    if model is None or scaler is None:
        return JSONResponse({"error": f"Model or scaler for {directory} not found"}, status_code=404)
    input_data = await preprocess_input(directory, scaler)
    if input_data is None:
        return JSONResponse({"error": f"Failed to preprocess input for {directory}"}, status_code=400)
    input_data = reshape_input(input_data)
    pred_scaled = model.predict(input_data)
    pred_original = scaler.inverse_transform(pred_scaled)
    results = [{"predicted_value": round(float(val), 3)} for val in pred_original.flatten()[:180]]
    return JSONResponse({directory: results})


# API endpoint for 3-monthly line predictions
@app.get("/predictions/three-monthly-line/{directory}")
async def get_monthly_line_predictions(directory: str):
    if directory not in directories:
        return JSONResponse({"error": "Invalid directory"}, status_code=400)
    model_name = f"{directory}_3_monthly"
    model = app.state.models.get(model_name)
    scaler = app.state.scalers.get(model_name)
    if model is None or scaler is None:
        return JSONResponse({"error": f"Model or scaler for {directory} not found"}, status_code=404)
    input_data = await preprocess_input(directory, scaler)
    if input_data is None:
        return JSONResponse({"error": f"Failed to preprocess input for {directory}"}, status_code=400)
    input_data = reshape_input(input_data)
    pred_scaled = model.predict(input_data)
    pred_original = scaler.inverse_transform(pred_scaled)
    results = [{"predicted_value": round(float(val), 3)} for val in pred_original.flatten()[:540]]
    return JSONResponse({directory: results})


# This endpoint is used to keep the server alive and can be used for health checks.
@app.get("/keep-alive")
@app.head("/keep-alive")
async def keep_alive():
    """Endpoint to keep the server alive."""
    return {"status": "alive"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)