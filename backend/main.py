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
from Utils.preprocess import preprocess_input_daily, preprocess_input
from fastapi import HTTPException
import numpy as np
from zoneinfo import ZoneInfo
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import subprocess
from dotenv import load_dotenv
import numpy as np



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUPYTER_PATH = os.path.join(BASE_DIR, os.getenv("JUPYTER_PATH"))

def reshape_input(input_data):
    """
    Reshape the input to the required shape (1, 42, 3).
    Assuming input_data is a 1D array (e.g., shape (1, 9)).

    :param input_data: The raw input data.
    :return: The reshaped input data.
    """
    input_data = np.array(input_data)  # Convert input to a numpy array
    input_length = input_data.shape[1]  # This should be 9 if your input shape is (1, 9)

    # If input has more than 126 features, we need to trim it
    if input_data.size > 126:
        input_data = input_data.flatten()[:126]
    # If input has less than 126 features, we need to pad it
    elif input_data.size < 126:
        padding_length = 126 - input_data.size
        input_data = np.pad(input_data.flatten(), (0, padding_length), mode='constant')

    # Reshape to (1, 42, 3), assuming we need to split it into 3 features per time step
    reshaped_input = input_data.reshape(1, 42, 3)  # Reshape to (1, 42, 3)

    return reshaped_input

def run_notebook(notebook_name):
    notebook_path = os.path.join(JUPYTER_PATH, notebook_name)
    try:
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", notebook_path, "--inplace"],
            capture_output=True, text=True, check=True
        )
        return {"status": "success", "details": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "details": e.stderr}
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

#api endpoint for daily prediction for all directories predicts only the final value at next day
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

#api endpoint for weekly prediction for all directories predicts the next 42 steps (7 days),
#and returns only the final predicted value (42nd step)
@app.get("/predictions/weekly")
async def get_weekly_predictions():
    directories = ["info", "scratch", "customer", "projects"]
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
    directories = ["info", "scratch", "customer", "projects"]
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
    directories = ["info", "scratch", "customer", "projects"]
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


@app.post("/retrain/weekly")
def retrain_weekly():
    return run_notebook("train_weekly_model.ipynb")

@app.post("/retrain/monthly")
def retrain_monthly():
    return run_notebook("train_monthly_model.ipynb")

@app.post("/retrain/quarterly")
def retrain_quarterly():
    return run_notebook("train_3_months_model.ipynb")




#line graph for predictions
@app.get("/predictions/weekly-line/{directory}")
async def get_weekly_line_predictions(directory: str):
    # Check if the directory is valid
    valid_directories = ["info", "scratch", "customer", "projects"]
    if directory not in valid_directories:
        return JSONResponse({"error": "Invalid directory"}, status_code=400)

    model_name = f"{directory}_weekly"
    model = app.state.models.get(model_name)
    scaler = app.state.scalers.get(model_name)

    if model is None or scaler is None:
        return JSONResponse({"error": f"Model or scaler for {directory} not found"}, status_code=404)

    # Preprocess input for predictions
    input_data = await preprocess_input_daily(directory, scaler)

    if input_data is None:
        return JSONResponse({"error": f"Failed to preprocess input for {directory}"}, status_code=400)

    # Reshape the input to match the model's expected shape (1, 42, 3)
    input_data = reshape_input(input_data)

    # Predict values and inverse transform
    pred_scaled = model.predict(input_data)
    pred_original = scaler.inverse_transform(pred_scaled)

    # For line graph: Return predicted values along with timestamps
    results = [{"predicted_value": round(float(val), 2)} for i, val in
               enumerate(pred_original.flatten())]

    return JSONResponse({directory: results})

from fastapi.responses import JSONResponse

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from datetime import datetime

@app.post("/predictions/category")
async def get_overall_prediction_category(predictions: dict):
    import pandas as pd
    from collections import defaultdict
    from fastapi.responses import JSONResponse

    # Normalize input keys to match DB (add leading slash)
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

    # Compute quantile thresholds per directory
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
