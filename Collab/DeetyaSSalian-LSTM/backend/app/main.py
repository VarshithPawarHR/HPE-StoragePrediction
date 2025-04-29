from fastapi import FastAPI
from .model import predict_storage
from .schemas import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Frontend (Next.js) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict", response_model=PredictionResponse)
def get_predictions():
    predictions = predict_storage()
    return predictions
