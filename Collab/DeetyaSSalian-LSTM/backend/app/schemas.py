from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    one_day: Dict[str, float]
    one_week: Dict[str, float]
    one_month: Dict[str, float]
    three_months: Dict[str, float]
