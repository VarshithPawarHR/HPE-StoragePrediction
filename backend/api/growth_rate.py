#growth rate endpoint logic is get the last 96th data entry then firstentry - last entry / lastentry No need for timestamps retreval from DB
from fastapi import APIRouter, Query
from pymongo import DESCENDING
from db import collection
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/growth-rate")
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