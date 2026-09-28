"""
Health check endpoint for hamdGAI
=================================
Provides a simple JSON response with status, uptime, and version.
"""

import time
import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = structlog.get_logger()
app = FastAPI(title="hamdGAI Health API")

start_time = time.time()


@app.get("/health")
def health_check():
    """Return service health information."""
    uptime = time.time() - start_time
    response = {
        "status": "ok",
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0",
    }
    logger.info("health_check", **response)
    return JSONResponse(content=response)
