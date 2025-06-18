from fastapi import APIRouter
import subprocess
import os

NOTEBOOKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "notebooks"
)
print(f"Notebooks path: {NOTEBOOKS_PATH}")
router = APIRouter()

def run_python_script(script_name):
    script_path = os.path.join(NOTEBOOKS_PATH, script_name)
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, check=True
        )
        return {"status": "success", "details": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "details": e.stderr}

@router.post("/retrain/daily")
def retrain_daily():
    return run_python_script("train_daily_model.py")

@router.post("/retrain/weekly")
def retrain_weekly():
    return run_python_script("train_weekly_model.py")

@router.post("/retrain/monthly")
def retrain_monthly():
    return run_python_script("train_monthly_model.py")

@router.post("/retrain/quarterly")
def retrain_quarterly():
    return run_python_script("train_3_months_model.py")