"""
🌐 Digital Twin T1D REST API
===========================

Cloud-ready REST API for the Digital Twin SDK.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime
import asyncio
import redis
import json

from .core import DigitalTwinSDK
from .datasets import load_diabetes_data


# Initialize FastAPI app
app = FastAPI(
    title="Digital Twin T1D API",
    description="Help 1 billion people with diabetes live without limits",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global SDK instance
sdk = None
redis_client = None


# Pydantic models for request/response
class DeviceConnection(BaseModel):
    device_type: str
    patient_id: Optional[str] = None
    
class GlucosePredictionRequest(BaseModel):
    horizon_minutes: int = 30
    patient_id: Optional[str] = None
    
class GlucosePredictionResponse(BaseModel):
    timestamp: datetime
    current_glucose: float
    predicted_glucose: float
    trend: str
    risk_level: str
    confidence: float
    recommendations: List[str]
    
class RecommendationRequest(BaseModel):
    patient_id: Optional[str] = None
    context: Optional[Dict] = None
    
class ClinicalReportRequest(BaseModel):
    patient_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    report_type: str = "comprehensive"
    
class VirtualTrialRequest(BaseModel):
    population_size: int = 100
    duration_days: int = 30
    interventions: List[str] = []
    
class DataUpload(BaseModel):
    patient_id: str
    data_type: str  # cgm, insulin, meals, etc.
    values: List[Dict]
    

@app.on_event("startup")
async def startup_event():
    """Initialize SDK and connections on startup."""
    global sdk, redis_client
    
    sdk = DigitalTwinSDK(mode='production')
    
    # Initialize Redis for caching (optional)
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected")
    except:
        redis_client = None
        print("⚠️ Redis not available, continuing without caching")


@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to Digital Twin T1D API",
        "mission": "Help 1 billion people with diabetes live without limits",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "sdk_status": "active" if sdk else "not initialized"
    }


@app.post("/device/connect")
async def connect_device(connection: DeviceConnection):
    """Connect a diabetes device."""
    try:
        sdk.connect_device(connection.device_type)
        return {
            "status": "connected",
            "device": connection.device_type,
            "patient_id": connection.patient_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/glucose")
async def predict_glucose(request: GlucosePredictionRequest):
    """Get glucose prediction."""
    try:
        # Check cache first
        cache_key = f"prediction:{request.patient_id}:{request.horizon_minutes}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Make prediction
        prediction = sdk.predict_glucose(horizon_minutes=request.horizon_minutes)
        recommendations = sdk.get_recommendations()
        
        response = GlucosePredictionResponse(
            timestamp=datetime.now(),
            current_glucose=prediction.current_glucose,
            predicted_glucose=prediction.value,
            trend=prediction.trend,
            risk_level=prediction.risk_level,
            confidence=prediction.confidence,
            recommendations=[r.action for r in recommendations[:3]]
        )
        
        # Cache result
        if redis_client:
            redis_client.setex(cache_key, 300, response.json())  # 5 min cache
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations."""
    try:
        recs = sdk.get_recommendations()
        return {
            "patient_id": request.patient_id,
            "timestamp": datetime.now(),
            "recommendations": [
                {
                    "action": r.action,
                    "reason": r.reason,
                    "priority": r.priority,
                    "category": r.category
                }
                for r in recs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clinical/report")
async def generate_clinical_report(
    request: ClinicalReportRequest,
    background_tasks: BackgroundTasks
):
    """Generate clinical report."""
    try:
        # Generate report in background
        background_tasks.add_task(
            _generate_report_async,
            request.patient_id,
            request.report_type
        )
        
        return {
            "status": "processing",
            "patient_id": request.patient_id,
            "report_type": request.report_type,
            "message": "Report generation started. Check /clinical/report/{patient_id} for status."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/virtual-trial")
async def run_virtual_trial(request: VirtualTrialRequest):
    """Run a virtual clinical trial."""
    try:
        results = sdk.run_virtual_trial(
            population_size=request.population_size,
            duration_days=request.duration_days,
            interventions=request.interventions
        )
        
        return {
            "status": "completed",
            "population_size": request.population_size,
            "duration_days": request.duration_days,
            "results": {
                "time_in_range_improvement": f"{results.tir_improvement:.1f}%",
                "hypoglycemia_reduction": f"{results.hypo_reduction:.1f}%",
                "hba1c_reduction": f"{results.hba1c_reduction:.2f}%",
                "quality_of_life_score": results.qol_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/upload")
async def upload_data(upload: DataUpload):
    """Upload patient data."""
    try:
        # Store data (in production, this would go to a database)
        if redis_client:
            key = f"data:{upload.patient_id}:{upload.data_type}"
            redis_client.lpush(key, json.dumps(upload.values))
            redis_client.expire(key, 3600)  # 1 hour expiry
        
        return {
            "status": "success",
            "patient_id": upload.patient_id,
            "data_type": upload.data_type,
            "records": len(upload.values)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from .datasets import list_available_datasets
    
    datasets = list_available_datasets()
    return {
        "count": len(datasets),
        "datasets": datasets.to_dict(orient='records')
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get information about a specific dataset."""
    from .datasets import DiabetesDatasets
    
    manager = DiabetesDatasets()
    if dataset_id not in manager.DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return manager.DATASETS[dataset_id]


@app.get("/supported-devices")
async def get_supported_devices():
    """Get list of supported devices."""
    from .integrations import DeviceFactory
    
    return {
        "cgm_devices": [
            "dexcom_g6", "dexcom_g7", "freestyle_libre", "freestyle_libre_2",
            "freestyle_libre_3", "guardian_3", "guardian_4", "eversense"
        ],
        "pump_devices": [
            "omnipod_dash", "omnipod_5", "tslim_x2", "medtronic_770g",
            "medtronic_780g", "ypsopump", "dana_rs"
        ],
        "wearables": [
            "apple_watch", "fitbit", "garmin", "samsung_galaxy_watch", "oura_ring"
        ]
    }


async def _generate_report_async(patient_id: str, report_type: str):
    """Background task to generate report."""
    # Simulate report generation
    await asyncio.sleep(5)
    
    # In production, this would save to database/storage
    if redis_client:
        report_data = {
            "patient_id": patient_id,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "status": "completed"
        }
        redis_client.setex(
            f"report:{patient_id}",
            3600,
            json.dumps(report_data)
        )


# Main entry point
def main():
    """Run the API server."""
    uvicorn.run(
        "sdk.api:app",
        host="0.0.0.0",  # nosec B104 - Intentionally binding to all interfaces for accessibility
        port=8080,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main() 