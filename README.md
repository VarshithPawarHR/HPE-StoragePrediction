# Storage Forecasting Dashboard – Proof of Concept (POC)

## Read This First – Limitations & Scope

This project is a **Proof of Concept** aimed at exploring **machine learning-based file storage prediction** for a specific internal storage system.

**The ML models built here are NOT general-purpose**. You cannot expect them to work on just any storage setup. Machine Learning models are **context-specific** — they learn patterns from the system they're trained on.

**If you want accurate predictions for your environment, you need to train your own model on your own data.** This project demonstrates what's possible, not what's universally applicable.

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
- A backend powered by **FastAPI**, **MongoDB** and **LSTM** + **GRU** time series forecasting models

This is a **research-grade POC**, not a production-ready tool.

---

## 🌐 Frontend Overview

The frontend is built using **Next.js** and **TypeScript**, offering:

- Real-time visualizations of storage usage
- Prediction graphs for multiple time horizons
- Directory-level trend insights
- Auto-refresh every 15 minutes (if live data ingestion is running)

To set up the frontend, see the `frontend/README.md`.

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
- Install dependencies:


pip install -r requirements.txt


### Key Features:

- Daily, weekly, monthly, and 3-month forecasts.
- Model training with LSTM neural networks.
- MongoDB integration for fetching data.
- Real-time predictions via a backend using FastAPI.


## Setup

### 1. Clone the repository

git clone <repository-url>
cd forecasting_project

### 2. Install dependencies

Use the following command to install required libraries:

pip install -r backend/requirements.txt

This will install all the necessary libraries for training models, running the backend, and other dependencies.

### 3. Set up environment variables

Create a `.env` file in the root directory and add your MongoDB URI:

MONGO_URI=mongodb+srv://<your_mongo_db_uri>

### 4. Run the backend

To run the backend and make predictions, navigate to the `backend` directory and run the FastAPI server:

cd backend
uvicorn app:app

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for training models:

These notebooks allow you to train the models for daily, weekly, monthly, and 3-month predictions.

## Model Usage

Once trained, the models are saved in the `models/` directory. You can load these models into your backend for real-time forecasting.

from tensorflow.keras.models import load_model

# Load the trained model

model = load_model('models/daily_model.keras')

## Backend API

The backend provides real-time predictions through a FastAPI service.

- TO DO (Remaining to do)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
