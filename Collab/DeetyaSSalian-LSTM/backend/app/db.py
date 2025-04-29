from pymongo import MongoClient
import pandas as pd

MONGO_URL = "mongodb+srv://bhavyanayak830:hpecppguys@cluster0.k0b3rqz.mongodb.net/"
DB_NAME = "storage_simulation"
COLLECTION_NAME = "usage_logs"

def get_data_from_mongo():
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = pd.DataFrame(list(collection.find()))
    return data
