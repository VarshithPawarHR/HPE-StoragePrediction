from pymongo import MongoClient
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# ------------------ Load Secrets from .env ------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["storage_simulation"]
collection = db["usage_logs"]

# Seed for reproducibility
np.random.seed(42)

# Time setup
start = datetime(2023, 4, 1)
end = datetime(2025, 4, 1)
timestamps = pd.date_range(start=start, end=end, freq='15min')

# Profile configs with subtle growth baked in
profiles = {
    "/scratch":   {"base": 1500, "volatility": 6.5, "drift": 0.008, "spike": 0.004, "drop": 0.003},
    "/projects":  {"base": 900,  "volatility": 4.0, "drift": 0.0055, "spike": 0.0025, "drop": 0.002},
    "/customer":  {"base": 600,  "volatility": 2.0, "drift": 0.003, "spike": 0.0015, "drop": 0.001},
    "/info":      {"base": 400,  "volatility": 1.2, "drift": 0.0018, "spike": 0.001, "drop": 0.0008}
}

def generate_data(name, config, batch_size=1000):
    current = config["base"]
    batch = []

    for i, ts in enumerate(timestamps):
        previous = current

        # Drift and randomness
        drift = np.random.normal(loc=config["drift"], scale=config["drift"] * 0.25)
        change = np.random.normal(0, config["volatility"])

        if np.random.rand() < config["spike"]:
            change += np.random.uniform(10, 60)
        if np.random.rand() < config["drop"]:
            change -= np.random.uniform(5, 80)

        current = max(current + drift + change, 0)
        current = round(current, 2)

        delta = current - previous
        added_gb = round(delta, 2) if delta > 0 else 0.0
        deleted_gb = round(-delta, 2) if delta < 0 else 0.0
        updated_gb = round(abs(delta), 2)

        doc = {
            "timestamp": ts,
            "directory": name,
            "storage_gb": current,
            "added_gb": added_gb,
            "deleted_gb": deleted_gb,
            "updated_gb": updated_gb
        }

        batch.append(doc)

        # Insert in batches
        if len(batch) >= batch_size:
            collection.insert_many(batch)
            batch = []

    # Insert any remaining docs
    if batch:
        collection.insert_many(batch)

# Generate and insert data per directory
for name, cfg in profiles.items():
    print(f"Inserting data for {name}...")
    generate_data(name, cfg)

print("✅ All documents inserted efficiently with tracking fields.")