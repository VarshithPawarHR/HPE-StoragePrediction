# HPE Stroage Prediction backend

## Project Overview

This project focuses on creating a forecasting system for storage data. It utilizes machine learning models, specifically LSTM-based neural networks, to predict storage usage for different time periods: daily, weekly, monthly, and 3 months ahead. The project is structured to work with data stored in MongoDB and involves training models, saving them, and deploying them in a backend for real-time forecasting.

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
