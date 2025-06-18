
#this is for pie chart
from fastapi import APIRouter, Query
from pymongo import DESCENDING
from db import collection
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/predictions/current")
async def get_current_storage():
    directories = await collection.distinct("directory")
    result = {}
    for directory in directories:
        cursor = collection.find({"directory": directory}).sort("timestamp", DESCENDING).limit(1)
        latest = await cursor.to_list(length=1)
        if latest:
            result[directory.strip("/")] = latest[0]["storage_gb"]
    return JSONResponse(result)
