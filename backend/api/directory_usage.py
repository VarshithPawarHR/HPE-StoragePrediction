from fastapi import APIRouter, Query
from pymongo import DESCENDING
from fastapi.responses import JSONResponse
from db import collection

router = APIRouter()

@router.get("/directory-usage")
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