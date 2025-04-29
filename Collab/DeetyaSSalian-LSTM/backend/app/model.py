from .db import get_data_from_mongo
from .utils import preprocess_data, naive_predict

def predict_storage():
    raw_data = get_data_from_mongo()
    data = preprocess_data(raw_data)

    directories = data['directory'].unique()

    result_1day = {}
    result_1week = {}
    result_1month = {}
    result_3months = {}

    for directory in directories:
        dir_data = data[data['directory'] == directory]

        # 1-Day Forecast (15 min data)
        pred_1day = naive_predict(dir_data.drop(columns=['directory']), shift=96)

        # Weekly/Monthly Forecasts (daily data)
        dir_daily = dir_data.resample('1D').mean(numeric_only=True)
        pred_1week = naive_predict(dir_daily, shift=7)
        pred_1month = naive_predict(dir_daily, shift=30)
        pred_3months = naive_predict(dir_daily, shift=90)

        if pred_1day:
            result_1day[directory] = round(pred_1day, 2)
        if pred_1week:
            result_1week[directory] = round(pred_1week, 2)
        if pred_1month:
            result_1month[directory] = round(pred_1month, 2)
        if pred_3months:
            result_3months[directory] = round(pred_3months, 2)

    return {
        "one_day": result_1day,
        "one_week": result_1week,
        "one_month": result_1month,
        "three_months": result_3months
    }
