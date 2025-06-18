import uvicorn
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any
import asyncio
from Utils.loader import load_keras_models, load_scalers
from Utils.preprocess import preprocess_input_daily, preprocess_input, reshape_input
from zoneinfo import ZoneInfo
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import JSONResponse


# Import routers
from api.directory_name import router as directory_name_router
from api.summary import router as summary_router
from api.directory_usage import router as directory_usage_router
from api.growth_rate import router as growth_rate_router
from api.total_consumption import router as total_consumption_router
from api.pie import router as pie_router
from api.keep_alive import router as keep_alive_router
from api.predictions_category import router as predictions_category_router
from api.retrain_model import router as retrain_model_router

load_dotenv()
# Read from .env
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# Global variable to store directories
directories = []
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')


IST = ZoneInfo("Asia/Kolkata")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global directories
    
    try:
        client = AsyncIOMotorClient(MONGO_URL)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        directories_raw = await collection.distinct("directory")
        directories = [d.lstrip("/") for d in directories_raw]
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


# API endpoints
app.include_router(summary_router)
app.include_router(directory_name_router)
app.include_router(directory_usage_router)
app.include_router(growth_rate_router)
app.include_router(total_consumption_router)
app.include_router(pie_router)
app.include_router(keep_alive_router)
app.include_router(predictions_category_router)
app.include_router(retrain_model_router)


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
        
        pred_scaled = model.predict(X_input)  # Shape: (1, 540)
        pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        # Get the 540th step (last value = end of 3rd month)
        last_value = round(float(pred_original[-1]), 2)
        results[directory] = last_value
    
    return results



# API endpoints for line graph predictions
# It includes endpoints for daily, weekly, monthly, and 3-monthly predictions
# which return the predicted values in a format suitable for line graphs.
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


# API endpoint for monthly line predictions
# It prepares the latest input sequence, predicts the next 180 steps (30 days with 4-hour intervals),
# and returns the predicted values in a format suitable for line graphs.
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
# It prepares the latest input sequence, predicts the next 540 time steps (3 months with 4-hour intervals),
# and returns the predicted values in a format suitable for line graphs.
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


# Run the application
# This will start the FastAPI server on the specified port
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)