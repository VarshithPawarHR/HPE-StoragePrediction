# Storage Forecasting HPE CPP project  – Proof of Concept (POC)

## Read This First – Limitations & Scope

This project is a **Proof of Concept** aimed at exploring **machine learning-based file storage prediction** for a specific internal storage system.

**The ML models built here are NOT direct to use or sell for systems.** You cannot expect them to work on just any storage setup. Machine Learning models are **context-specific** — they learn patterns from the system they're trained on.

**If you want accurate predictions for your environment, you need to refine this  model on your own data and then train** This project demonstrates what's possible, not what's universally applicable.

---

## 🧠 What This Project Is

This is a full-stack storage monitoring and forecasting dashboard that provides:

- Real-time insights into current storage usage  
- Forecasts for:
  - 📅 Next Day  
  - 📈 Next Week  
  - 📆 Next Month  
  - 📊 Next 3 Months
- A sleek dashboard built with **Next.js** and **TypeScript**  
- A backend powered by **FastAPI**, **MongoDB**, and **Hybrid Deep learning models** time series forecasting models

This is a **research-grade POC**, not a production-ready tool or software .

---

## 🌐 Frontend Overview

The frontend is built using **Next.js** and **TypeScript**, offering:

- Real-time visualizations of storage usage  
- Prediction graphs for multiple time horizons  
- Directory-level trend insights  
- Auto-refresh every 15 minutes (if live data ingestion is running)

To set up the frontend, see the `https://github.com/VarshithPawarHR/HPE-Dashboard`.

---

## ⚙️ Backend Setup – FastAPI

The backend handles:

- Live data ingestion and synthetic data simulation  
- Storage forecasting using **LSTM** and **ARIMA**  
- API services consumed by the frontend  
- MongoDB database integration

### 🔧 Requirements

- Python: **3.10.x**  
- MongoDB: running locally or via cloud

Install dependencies:

```bash
pip install -r requirements.txt

```
### Clone

```bash
git clone https://github.com/VarshithPawarHR/HPE-StoragePrediction
cd backend

```

### Install dependencies

```bash
pip install -r backend/requirenments.txt

```
### Setup environmental Variables

```bash
TF_ENABLE_ONEDNN_OPTS=0
MONGO_URL =
MONGO_DB = 
MONGO_COLLECTION = 
JUPYTER_PATH=notebooks
```

### Run the backend
```bash
cd backend
fastapi dev main.py
```

## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](./LICENSE) file for full details.
