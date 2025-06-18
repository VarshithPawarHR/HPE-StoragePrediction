# This endpoint is used to keep the server alive and can be used for health checks.
from fastapi import APIRouter

router = APIRouter()

@router.get("/keep-alive")
@router.head("/keep-alive")
async def keep_alive():
    """Endpoint to keep the server alive."""
    return {"status": "alive"}