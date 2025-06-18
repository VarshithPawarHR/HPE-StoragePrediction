from fastapi import APIRouter, Query
from pymongo import DESCENDING
from db import collection

router = APIRouter()

@router.get("/summary")
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