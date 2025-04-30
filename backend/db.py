from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB connection details
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# Setup MongoDB client
client = AsyncIOMotorClient(MONGO_URL)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]
