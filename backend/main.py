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



