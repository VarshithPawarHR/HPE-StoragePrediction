#consumpton endpoint get the 96th data entry from start then latest- 96th entry
from fastapi import APIRouter, Query
from pymongo import DESCENDING
from db import collection
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/total-consumption")
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