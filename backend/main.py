from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from Utils.loader import load_keras_models, load_scalers
from typing import Dict, Any
import asyncio
from db import collection
from pymongo import DESCENDING
from fastapi.responses import JSONResponse
from zoneinfo import ZoneInfo

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