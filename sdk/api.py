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
    redoc_url="/redoc",
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

    sdk = DigitalTwinSDK(mode="production")

    # Initialize Redis for caching (optional)
    try:
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
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
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "sdk_status": "active" if sdk else "not initialized",
    }


@app.post("/device/connect")
async def connect_device(connection: DeviceConnection):
    """Connect a diabetes device."""
    assert sdk is not None, "SDK not initialized"
    try:
        # Assuming patient_id can be used as device_id for this context
        device_id = connection.patient_id if connection.patient_id else "default_device"
        sdk.connect_device(device_type=connection.device_type, device_id=device_id)
        return {
            "status": "connected",
            "device": connection.device_type,
            "patient_id": connection.patient_id,
            "device_id_used": device_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/glucose")
async def predict_glucose(request: GlucosePredictionRequest):
    """Get glucose prediction."""
    assert sdk is not None, "SDK not initialized"
    try:
        # Check cache first
        cache_key = f"prediction:{request.patient_id}:{request.horizon_minutes}"
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:  # Check if cached_data is not None
                assert isinstance(
                    cached_data, str
                ), "Cached data should be a string from Redis with decode_responses=True"
                return json.loads(cached_data)

        # Make prediction using SDK
        sdk_prediction_obj = sdk.predict_glucose(
            horizon_minutes=request.horizon_minutes
        )  # Renamed for clarity

        # Get current glucose
        latest_data_df = sdk._get_latest_data()
        current_glucose_val = (
            float(latest_data_df["cgm"].iloc[-1]) if not latest_data_df.empty else 120.0
        )

        # Predicted glucose is the first value in the prediction horizon
        predicted_glucose_val = (
            float(sdk_prediction_obj.values[0])
            if sdk_prediction_obj.values
            else current_glucose_val
        )

        # Determine trend
        trend_str = "stable"
        if len(sdk_prediction_obj.values) > 0:  # Check if there are any predicted values
            if sdk_prediction_obj.values[0] < current_glucose_val:
                trend_str = "falling"
            elif sdk_prediction_obj.values[0] > current_glucose_val:
                trend_str = "rising"

        risk_level_str = "Normal"
        if (
            sdk_prediction_obj.risk_alerts
            and isinstance(sdk_prediction_obj.risk_alerts, list)
            and sdk_prediction_obj.risk_alerts
        ):
            risk_level_str = str(sdk_prediction_obj.risk_alerts[0])

        confidence_val = 0.95  # Default confidence
        if (
            sdk_prediction_obj.confidence_intervals
            and isinstance(sdk_prediction_obj.confidence_intervals, tuple)
            and len(sdk_prediction_obj.confidence_intervals) == 2
        ):
            # Example: derive confidence from interval width, lower bound > 0
            lower, upper = sdk_prediction_obj.confidence_intervals
            if upper > lower and lower > 0 and predicted_glucose_val > 0:  # basic sanity check
                # This is a placeholder logic for confidence.
                # A proper calculation would depend on how confidence_intervals are defined (e.g. std dev, percentile)
                pass  # Keep default 0.95 for now

        # Get recommendations from SDK
        sdk_recs = sdk.get_recommendations()  # This returns a Dict
        api_recommendations: List[str] = []

        insulin_rec = sdk_recs.get("insulin")
        if isinstance(insulin_rec, dict) and insulin_rec.get("action"):
            api_recommendations.append(str(insulin_rec["action"]))

        meal_recs = sdk_recs.get("meals")
        if isinstance(meal_recs, list):
            for meal_rec_item in meal_recs:
                if isinstance(meal_rec_item, str):
                    api_recommendations.append(meal_rec_item)
                # Add more specific parsing if meal_recs can contain dicts

        activity_recs = sdk_recs.get("activity")
        if isinstance(activity_recs, list):
            for activity_rec_item in activity_recs:
                if isinstance(activity_rec_item, str):
                    api_recommendations.append(activity_rec_item)
                # Add more specific parsing if activity_recs can contain dicts

        response = GlucosePredictionResponse(
            timestamp=datetime.now(),
            current_glucose=current_glucose_val,
            predicted_glucose=predicted_glucose_val,
            trend=trend_str,
            risk_level=risk_level_str,
            confidence=confidence_val,
            recommendations=api_recommendations[:3],
        )

        # Cache result
        if redis_client:
            redis_client.setex(cache_key, 300, response.json())

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations."""
    assert sdk is not None, "SDK not initialized"
    try:
        sdk_recs_dict = sdk.get_recommendations(context=request.context)  # sdk_recs_dict is a Dict

        api_response_recommendations = []

        # Insulin recommendation
        insulin_info = sdk_recs_dict.get("insulin")
        if isinstance(insulin_info, dict):
            api_response_recommendations.append(
                {
                    "action": insulin_info.get("action", "N/A"),
                    "reason": insulin_info.get("reason", "N/A"),
                    "priority": insulin_info.get("priority", "medium"),
                    "category": "insulin",
                }
            )

        # Meal recommendations
        meal_info_list = sdk_recs_dict.get("meals")
        if isinstance(meal_info_list, list):
            for meal_item in meal_info_list:
                if isinstance(meal_item, str):  # If it's just a string
                    api_response_recommendations.append(
                        {
                            "action": meal_item,
                            "reason": "General meal advice",
                            "priority": "low",
                            "category": "meals",
                        }
                    )
                elif isinstance(meal_item, dict):  # If it's a dict with more details
                    api_response_recommendations.append(
                        {
                            "action": meal_item.get("action", "N/A"),
                            "reason": meal_item.get("reason", "N/A"),
                            "priority": meal_item.get("priority", "medium"),
                            "category": "meals",
                        }
                    )

        # Activity recommendations
        activity_info_list = sdk_recs_dict.get("activity")
        if isinstance(activity_info_list, list):
            for activity_item in activity_info_list:
                if isinstance(activity_item, str):
                    api_response_recommendations.append(
                        {
                            "action": activity_item,
                            "reason": "General activity advice",
                            "priority": "low",
                            "category": "activity",
                        }
                    )
                elif isinstance(activity_item, dict):
                    api_response_recommendations.append(
                        {
                            "action": activity_item.get("action", "N/A"),
                            "reason": activity_item.get("reason", "N/A"),
                            "priority": activity_item.get("priority", "medium"),
                            "category": "activity",
                        }
                    )

        # Alerts (if any)
        alert_info_list = sdk_recs_dict.get("alerts")
        if isinstance(alert_info_list, list):
            for alert_item in alert_info_list:
                if isinstance(alert_item, str):
                    api_response_recommendations.append(
                        {
                            "action": alert_item,
                            "reason": "System alert",
                            "priority": "high",  # Assuming alerts are high priority
                            "category": "alerts",
                        }
                    )
                elif isinstance(alert_item, dict):
                    api_response_recommendations.append(
                        {
                            "action": alert_item.get(
                                "message", "N/A"
                            ),  # Assuming alert dict has 'message'
                            "reason": alert_item.get("details", "System alert"),
                            "priority": alert_item.get("priority", "high"),
                            "category": "alerts",
                        }
                    )

        return {
            "patient_id": request.patient_id,
            "timestamp": datetime.now(),
            "recommendations": api_response_recommendations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clinical/report")
async def generate_clinical_report(
    request: ClinicalReportRequest, background_tasks: BackgroundTasks
):
    """Generate clinical report."""
    assert sdk is not None, "SDK not initialized"
    try:
        # Generate report in background
        background_tasks.add_task(_generate_report_async, request.patient_id, request.report_type)

        return {
            "status": "processing",
            "patient_id": request.patient_id,
            "report_type": request.report_type,
            "message": "Report generation started. Check /clinical/report/{patient_id} for status.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/virtual-trial")
async def run_virtual_trial(request: VirtualTrialRequest):
    """Run a virtual clinical trial."""
    assert sdk is not None, "SDK not initialized"
    try:
        results = sdk.run_virtual_trial(
            cohort_size=request.population_size,  # Parameter name mismatch with sdk.run_virtual_trial
            duration_days=request.duration_days,
            interventions=request.interventions,
        )

        # Handle cases where results might not contain expected keys (e.g., if simulator fails)
        tir_improvement = results.get("tir_improvement", 0.0)
        hypo_reduction = results.get("hypo_reduction", 0.0)
        hba1c_reduction = results.get("hba1c_reduction", 0.0)
        qol_score = results.get("qol_score", "N/A")

        return {
            "status": results.get("status", "completed"),
            "population_size": request.population_size,
            "duration_days": request.duration_days,
            "results": {
                "time_in_range_improvement": f"{tir_improvement:.1f}%",
                "hypoglycemia_reduction": f"{hypo_reduction:.1f}%",
                "hba1c_reduction": f"{hba1c_reduction:.2f}%",
                "quality_of_life_score": qol_score,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/upload")
async def upload_data(upload: DataUpload):
    """Upload patient data."""
    assert (
        sdk is not None
    ), "SDK not initialized"  # Though sdk is not directly used here, good practice
    try:
        # Store data (in production, this would go to a database)
        if redis_client:
            key = f"data:{upload.patient_id}:{upload.data_type}"
            # Ensure upload.values is a list of strings for lpush if not already
            values_to_push = [json.dumps(v) if not isinstance(v, str) else v for v in upload.values]
            if values_to_push:  # lpush expects at least one value
                redis_client.lpush(key, *values_to_push)
                redis_client.expire(key, 3600)  # 1 hour expiry

        return {
            "status": "success",
            "patient_id": upload.patient_id,
            "data_type": upload.data_type,
            "records": len(upload.values),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from .datasets import list_available_datasets

    datasets = list_available_datasets()
    return {"count": len(datasets), "datasets": datasets.to_dict(orient="records")}


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
            "dexcom_g6",
            "dexcom_g7",
            "freestyle_libre",
            "freestyle_libre_2",
            "freestyle_libre_3",
            "guardian_3",
            "guardian_4",
            "eversense",
        ],
        "pump_devices": [
            "omnipod_dash",
            "omnipod_5",
            "tslim_x2",
            "medtronic_770g",
            "medtronic_780g",
            "ypsopump",
            "dana_rs",
        ],
        "wearables": ["apple_watch", "fitbit", "garmin", "samsung_galaxy_watch", "oura_ring"],
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
            "status": "completed",
        }
        redis_client.setex(f"report:{patient_id}", 3600, json.dumps(report_data))


# Main entry point
def main():
    """Run the API server."""
    uvicorn.run(
        "sdk.api:app",
        host="0.0.0.0",  # nosec B104 - Intentionally binding to all interfaces for accessibility
        port=8080,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
