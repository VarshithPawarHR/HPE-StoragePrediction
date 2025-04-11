from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# ------------------ Load Secrets from .env ------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ------------------ MongoDB Setup ------------------
client = MongoClient(MONGO_URI)
db = client["storage_simulation"]
collection = db["usage_logs"]

# ------------------ Storage Profiles ------------------
profiles = {
    "/scratch":   {"base": 1500, "volatility": 6.5, "drift": 0.008, "spike": 0.004, "drop": 0.003},
    "/projects":  {"base": 900,  "volatility": 4.0, "drift": 0.0055, "spike": 0.0025, "drop": 0.002},
    "/customer":  {"base": 600,  "volatility": 2.0, "drift": 0.003, "spike": 0.0015, "drop": 0.001},
    "/info":      {"base": 400,  "volatility": 1.2, "drift": 0.0018, "spike": 0.001, "drop": 0.0008}
}

# ------------------ Utility Functions ------------------
def get_last_timestamp(directory):
    latest_doc = collection.find({"directory": directory}).sort("timestamp", -1).limit(1)
    latest_ts = next(latest_doc, None)
    return latest_ts["timestamp"] if latest_ts else datetime(2025, 4, 10)

def generate_value(prev_val, cfg):
    drift = np.random.normal(loc=cfg["drift"], scale=cfg["drift"] * 0.25)
    change = np.random.normal(0, cfg["volatility"])
    if np.random.rand() < cfg["spike"]:
        change += np.random.uniform(10, 60)
    if np.random.rand() < cfg["drop"]:
        change -= np.random.uniform(5, 80)
    new_val = round(max(prev_val + drift + change, 0), 2)
    delta = new_val - prev_val
    return new_val, round(max(delta, 0), 2), round(max(-delta, 0), 2), round(abs(delta), 2)

def generate_and_insert(directory, cfg, start_ts, end_ts, prev_val):
    timestamps = pd.date_range(start=start_ts, end=end_ts, freq="15min")
    for ts in timestamps:
        current, added, deleted, updated = generate_value(prev_val, cfg)
        doc = {
            "timestamp": ts,
            "directory": directory,
            "storage_gb": current,
            "added_gb": added,
            "deleted_gb": deleted,
            "updated_gb": updated
        }
        collection.insert_one(doc)
        prev_val = current
    return prev_val

# ------------------ Main Loop ------------------
def live_data_insertion_loop():
    last_vals = {}

    print("🔁 Backfilling missing data from last known timestamps...")
    now = datetime.now().replace(second=0, microsecond=0)

    for directory, cfg in profiles.items():
        last_ts = get_last_timestamp(directory)
        prev_val_doc = collection.find({"directory": directory, "timestamp": last_ts}).limit(1)
        prev_val = next(prev_val_doc, {"storage_gb": cfg["base"]})["storage_gb"]
        last_vals[directory] = generate_and_insert(directory, cfg, last_ts + timedelta(minutes=15), now, prev_val)

    print("✅ Backfill complete. Entering live insertion mode (every 15 minutes)...")

    while True:
        now = datetime.now().replace(second=0, microsecond=0)
        for directory, cfg in profiles.items():
            prev_val = last_vals[directory]
            current, added, deleted, updated = generate_value(prev_val, cfg)
            doc = {
                "timestamp": now,
                "directory": directory,
                "storage_gb": current,
                "added_gb": added,
                "deleted_gb": deleted,
                "updated_gb": updated
            }
            collection.insert_one(doc)
            last_vals[directory] = current
        print(f"[{now}] ✅ Inserted live records for all directories.")
        time.sleep(900)  # Wait for 15 minutes

# ------------------ Script Entry Point ------------------
if __name__ == "__main__":
    live_data_insertion_loop()
