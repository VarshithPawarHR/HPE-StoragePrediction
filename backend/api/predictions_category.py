# API endpoint to get overall prediction category based on quantiles
# This endpoint takes a dictionary of predictions and returns an overall category (low, moderate, high
from fastapi import APIRouter
from db import collection
from fastapi.responses import JSONResponse
from collections import defaultdict
import pandas as pd

router = APIRouter()

@router.post("/predictions/category")
async def get_overall_prediction_category(predictions: dict):
    predictions = {f"/{k.lstrip('/')}" : v for k, v in predictions.items()}
    cursor = collection.find().sort("timestamp", -1).limit(2880)
    data = defaultdict(list)
    async for doc in cursor:
        data['timestamp'].append(pd.to_datetime(doc['timestamp']))
        data['directory'].append(doc['directory'])
        data['storage_gb'].append(doc['storage_gb'])
    df = pd.DataFrame(data)
    if df.empty:
        return JSONResponse({"error": "No data found in DB."}, status_code=404)
    labels = {}
    for dir_name in df['directory'].unique():
        subset = df[df['directory'] == dir_name]
        q1 = subset['storage_gb'].quantile(0.33)
        q2 = subset['storage_gb'].quantile(0.66)
        labels[dir_name] = {'low': q1, 'moderate': q2}

    label_scores = {'low': 1, 'moderate': 2, 'high': 3}
    scores = []
    for directory, predicted_value in predictions.items():
        if directory not in labels:
            return JSONResponse({"error": f"Invalid directory: {directory}"}, status_code=400)
        thresholds = labels[directory]
        if predicted_value <= thresholds['low']:
            category = 'low'
        elif predicted_value <= thresholds['moderate']:
            category = 'moderate'
        else:
            category = 'high'
        scores.append(label_scores[category])
    avg_score = sum(scores) / len(scores)
    if avg_score <= 1.5:
        overall_category = 'low'
    elif avg_score <= 2.3:
        overall_category = 'moderate'
    else:
        overall_category = 'high'
    return JSONResponse({
        "overall_category": overall_category
    })

