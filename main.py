import os
import uuid

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime, time
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from model_hot_swap import LiveModelContainer
from scheduler import start_mlops_scheduler, execute_automated_retraining_pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import contextlib
import numpy as np


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(execute_automated_retraining_pipeline, 'interval', hours=24)
    scheduler.start()    
    yield  # The FastAPI server runs here...
    scheduler.shutdown()
    print("stopped.")

app = FastAPI(lifespan=lifespan, title="MLOps Engine", version="1.0.0")

ACTIVE_MODEL = LiveModelContainer("./models/Gesture_Classifier_model2.tflite")

# @app.on_event("startup")
# def on_startup():
#     start_mlops_scheduler()

class InferenceLogPayload(BaseModel):
    device_id: str = Field(..., example="DEVICE_V1_0842")
    predicted_gesture_id: int = Field(..., example=2)
    confidence_score: float = Field(..., example=0.965)
    inference_latency_ms: float = Field(..., example=14.2)
    accelerometer_frame: list[list[float]] = Field(..., example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...])

class DataPayload(BaseModel):
    device_id: str = Field(..., example="DEVICE_V1_0842")
    accelerometer_frame: list[list[float]] = Field(..., example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...])

@app.post("/api/v1/telemetry/log", status_code=status.HTTP_201_CREATED)
def ingest_inference_log(payload: DataPayload, db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        raw_matrix = np.array(payload.accelerometer_frame, dtype=np.float32) # Shape: (25, 3)
        
        if raw_matrix.shape != (25, 3):
            raise HTTPException(status_code=400, detail="Invalid matrix frame dimensions. Must be 25x3.")

        pred_id, score = ACTIVE_MODEL.predict(raw_matrix)

        os.makedirs("data_lake_storage", exist_ok=True)
        unique_file_id = f"log_{uuid.uuid4().hex}_{int(time.time())}.npy"
        file_storage_path = os.path.join("data_lake_storage", unique_file_id)
        
        np.save(file_storage_path, raw_matrix)

        latency_ms = (time.time() - start_time) * 1000

        query = text("""
            INSERT INTO gesture_inference_logs (device_id, predicted_gesture_id, confidence_score, inference_latency_ms, logged_at)
            VALUES (:device_id, :predicted_gesture_id, :confidence_score, :inference_latency_ms, :logged_at)
        """)
        db.execute(query, {
            "device_id": payload.device_id,
            "predicted_gesture_id": pred_id,
            "confidence_score": score,
            "inference_latency_ms": latency_ms,
            "raw_data_key": file_storage_path,
            "logged_at": datetime.now()
        })
        db.commit()
        return {"status": "success", "predicted_gesture_id": pred_id, "confidence": score, "message": "Telemetry metrics successfully processed and routed."}
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
    
@app.post("/api/v1/analytics/refresh")
def refresh_active_model_weights():
    try:
        ACTIVE_MODEL.load_model()
        return {"status": "success", "message": "Inference weights hot-swapped successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))