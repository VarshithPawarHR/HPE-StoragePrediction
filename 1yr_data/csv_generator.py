import numpy as np
import pandas as pd
from datetime import datetime

# Seed for reproducibility
np.random.seed(42)

# Time range setup
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

def generate_data(name, config):
    data = []
    current = config["base"]
    
    for ts in timestamps:
        drift = np.random.normal(loc=config["drift"], scale=config["drift"] * 0.25)
        change = np.random.normal(0, config["volatility"])

        if np.random.rand() < config["spike"]:
            change += np.random.uniform(10, 60)
        if np.random.rand() < config["drop"]:
            change -= np.random.uniform(5, 80)

        current = max(current + drift + change, 0)
        data.append([ts, name, round(current, 2)])

    return data

# Combine everything
final_data = []
for name, cfg in profiles.items():
    final_data.extend(generate_data(name, cfg))

# Create and save CSV
df_final = pd.DataFrame(final_data, columns=["timestamp", "directory", "storage_gb"])
df_final.to_csv("le_growth.csv", index=False)

print("✅ le_growth.csv saved successfully!")