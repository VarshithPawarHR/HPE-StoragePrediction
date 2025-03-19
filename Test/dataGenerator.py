import time
import pandas as pd
import random
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient

# MongoDB Atlas Connection String (Replace with your actual URI)
MONGO_URI = "mongodb+srv://VarshithPawarHR:Aw8VVQ0Aa80pGE2X@hpecluster.w0shn.mongodb.net/?retryWrites=true&w=majority&appName=HPEcluster"
client = MongoClient(MONGO_URI)
db = client["StorageMonitoring"]

# Directories and their dynamic storage limits
storage_limits = {
    "customers": (1000, 2500),  # 1TB - 2.5TB
    "info": (500, 1250),  # 500GB - 1.25TB
    "projects": (500, 1000),  # 500GB - 1TB
    "scratch": (500, 2000)  # 500GB - 2TB
}

# Logging setup
logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


def get_last_timestamp(collection):
    """Fetch the latest timestamp from MongoDB."""
    last_entry = collection.find_one(sort=[("Timestamp", -1)])
    return datetime.fromisoformat(last_entry["Timestamp"]) if last_entry else None


def generate_synthetic_data(directory, last_space, timestamp):
    """Generate smooth synthetic data for each directory with dynamic fluctuations."""
    min_storage, max_storage = storage_limits[directory]

    # Introduce a gradual fluctuation factor based on time
    fluctuation_factor = random.uniform(-50, 50)
    new_space = max(min_storage, min(max_storage, last_space + fluctuation_factor))

    data = {
        "Timestamp": timestamp.isoformat(),
        "Directory": f"/{directory}",
        "Files Added (GB)": random.uniform(0, 5),
        "Files Deleted (GB)": random.uniform(0, 5),
        "Files Modified (GB)": random.uniform(0, 2),
        "Current Space (GB)": new_space
    }
    return data, new_space


def update_database():
    """Generate and insert new data into MongoDB and the respective CSV file, including backfilling missing records."""
    for directory in storage_limits.keys():
        collection = db[directory]
        last_timestamp = get_last_timestamp(collection)
        if last_timestamp is None:
            last_timestamp = datetime.utcnow() - timedelta(minutes=15)

        last_space = collection.find_one(sort=[("Timestamp", -1)])
        last_space = last_space["Current Space (GB)"] if last_space else storage_limits[directory][0]

        current_time = datetime.utcnow()
        while last_timestamp < current_time:
            new_data, last_space = generate_synthetic_data(directory, last_space, last_timestamp)
            collection.insert_one(new_data)
            logging.info(f"Inserted new record for {directory}: {new_data}")

            # Remove '_id' field before saving to CSV
            new_data.pop("_id", None)

            # Append to the correct CSV file for each directory
            csv_filename = f"{directory}.csv"
            df = pd.DataFrame([new_data])
            df.to_csv(csv_filename, mode="a", header=False, index=False)

            # Increment timestamp by 15 minutes
            last_timestamp += timedelta(minutes=15)


while True:
    update_database()
    time.sleep(900)  # Run every 15 minutes
