from fastapi import FastAPI
from contextlib import asynccontextmanager
from Utils.loader import load_keras_models, load_scalers

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and scalers
    models = load_keras_models()
    scalers = load_scalers()

    # Save them in app state
    app.state.models = models
    app.state.scalers = scalers

    # Print loaded assets
    print(" Models loaded:", list(models.keys()))
    print(" Scalers loaded:", list(scalers.keys()))

    yield

    print(" Application is shutting down.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}
