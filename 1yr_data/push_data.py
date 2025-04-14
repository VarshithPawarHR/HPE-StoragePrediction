from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# ------------------ MongoDB Setup ------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

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

def generate_and_bulk_insert(directory, cfg, start_ts, end_ts, prev_val):
    timestamps = pd.date_range(start=start_ts, end=end_ts, freq="15min")
    docs = []
    for ts in timestamps:
        current, added, deleted, updated = generate_value(prev_val, cfg)
        docs.append({
            "timestamp": ts,
            "directory": directory,
            "storage_gb": current,
            "added_gb": added,
            "deleted_gb": deleted,
            "updated_gb": updated
        })
        prev_val = current
    if docs:
        collection.insert_many(docs)
        return prev_val, timestamps[-1]
    return prev_val, start_ts - timedelta(minutes=15)

# ------------------ Main Loop ------------------
def live_data_insertion_loop():
    last_vals = {}
    latest_timestamps = {}

    print("🔁 Backfilling missing data from last known timestamps...")
    now = datetime.now().replace(second=0, microsecond=0)

    for directory, cfg in profiles.items():
        last_ts = get_last_timestamp(directory)
        prev_val_doc = collection.find({"directory": directory, "timestamp": last_ts}).limit(1)
        prev_val = next(prev_val_doc, {"storage_gb": cfg["base"]})["storage_gb"]

        start_ts = last_ts + timedelta(minutes=15)
        if start_ts <= now:
            new_prev_val, final_backfill_ts = generate_and_bulk_insert(
                directory, cfg, start_ts, now, prev_val
            )
            last_vals[directory] = new_prev_val
            latest_timestamps[directory] = final_backfill_ts
        else:
            print(f"🟡 No backfill needed for {directory}. Already up to date.")
            last_vals[directory] = prev_val
            latest_timestamps[directory] = last_ts

    print("✅ Backfill complete.")

    # Determine next 15-minute slot for live mode
    now = datetime.now().replace(second=0, microsecond=0)
    minutes = (now.minute // 15 + 1) * 15
    if minutes == 60:
        next_live_ts = now.replace(minute=0) + timedelta(hours=1)
    else:
        next_live_ts = now.replace(minute=minutes)

    print(f"🕒 Waiting until {next_live_ts} to begin live mode....")
    while datetime.now() < next_live_ts:
        time.sleep(5)

    print("🚀 Entering live insertion mode (every 15 minutes)...")

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
        time.sleep(900)

# ------------------ Script Entry Point ------------------
if __name__ == "__main__":
    live_data_insertion_loop()
