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

