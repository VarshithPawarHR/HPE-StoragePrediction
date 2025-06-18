from fastapi import APIRouter
from pymongo import DESCENDING
from db import collection

router = APIRouter()

@router.get("/directory-name")
async def get_recent_directories():
    # Fetch latest 4 documents sorted by timestamp
    cursor = collection.find().sort("timestamp", DESCENDING).limit(4)
    latest_entries = await cursor.to_list(length=4)
    # Extract only the directory names
    directories = [entry.get("directory") for entry in latest_entries]
    return {"recent_directories": directories}