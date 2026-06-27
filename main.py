from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from scheduler import start_mlops_scheduler, execute_automated_retraining_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import contextlib


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(execute_automated_retraining_pipeline, 'interval', hours=24)
    scheduler.start()    
    yield  # The FastAPI server runs here...
    scheduler.shutdown()
    print("Automated MLOps background cron engine stopped.")

app = FastAPI(lifespan=lifespan, title="MLOps Engine", version="1.0.0")

# @app.on_event("startup")
# def on_startup():
#     start_mlops_scheduler()

class InferenceLogPayload(BaseModel):
    device_id: str = Field(..., example="DEVICE_V1_0842")
    predicted_gesture_id: int = Field(..., example=2)
    confidence_score: float = Field(..., example=0.965)
    inference_latency_ms: float = Field(..., example=14.2)

@app.post("/api/v1/telemetry/log", status_code=status.HTTP_201_CREATED)
def ingest_inference_log(payload: InferenceLogPayload, db: Session = Depends(get_db)):
    try:
        query = text("""
            INSERT INTO gesture_inference_logs (device_id, predicted_gesture_id, confidence_score, inference_latency_ms, logged_at)
            VALUES (:device_id, :predicted_gesture_id, :confidence_score, :inference_latency_ms, :logged_at)
        """)
        db.execute(query, {
            "device_id": payload.device_id,
            "predicted_gesture_id": payload.predicted_gesture_id,
            "confidence_score": payload.confidence_score,
            "inference_latency_ms": payload.inference_latency_ms,
            "logged_at": datetime.now()
        })
        db.commit()
        return {"status": "success", "message": "Telemetry metrics successfully processed and routed."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write crash: {str(e)}")

@app.post("/api/v1/analytics/refresh", status_code=status.HTTP_200_OK)
def refresh_analytics_cache(db: Session = Depends(get_db)):
    try:
        db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY daily_hardware_latencies;"))
        db.commit()
        return {"status": "success", "message": "Materialized caching indexes rebuilt successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))