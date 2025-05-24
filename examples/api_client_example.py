"""
🌐 Digital Twin T1D API Client Example
=====================================

Παράδειγμα χρήσης του REST API από Python.
"""

import requests
import json
from datetime import datetime
import time


class DigitalTwinAPIClient:
    """Simple API client για το Digital Twin SDK."""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self):
        """Έλεγχος υγείας του API."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def connect_device(self, device_type, patient_id=None):
        """Σύνδεση συσκευής."""
        data = {
            "device_type": device_type,
            "patient_id": patient_id
        }
        response = self.session.post(
            f"{self.base_url}/device/connect",
            json=data
        )
        return response.json()
    
    def predict_glucose(self, horizon_minutes=30, patient_id=None):
        """Πρόβλεψη γλυκόζης."""
        data = {
            "horizon_minutes": horizon_minutes,
            "patient_id": patient_id
        }
        response = self.session.post(
            f"{self.base_url}/predict/glucose",
            json=data
        )
        return response.json()
    
    def get_recommendations(self, patient_id=None):
        """Λήψη συστάσεων."""
        data = {"patient_id": patient_id}
        response = self.session.post(
            f"{self.base_url}/recommendations",
            json=data
        )
        return response.json()
    
    def list_datasets(self):
        """Λίστα διαθέσιμων datasets."""
        response = self.session.get(f"{self.base_url}/datasets")
        return response.json()
    
    def get_supported_devices(self):
        """Λίστα υποστηριζόμενων συσκευών."""
        response = self.session.get(f"{self.base_url}/supported-devices")
        return response.json()


def main():
    """Παράδειγμα χρήσης του API."""
    
    print("🚀 Digital Twin T1D API Client Example")
    print("=" * 50)
    
    # Δημιουργία client
    client = DigitalTwinAPIClient()
    
    # 1. Health Check
    print("\n1️⃣ Checking API health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   SDK: {health['sdk_status']}")
    
    # 2. Υποστηριζόμενες συσκευές
    print("\n2️⃣ Getting supported devices...")
    devices = client.get_supported_devices()
    print(f"   CGM Devices: {len(devices['cgm_devices'])}")
    print(f"   Pump Devices: {len(devices['pump_devices'])}")
    print(f"   Wearables: {len(devices['wearables'])}")
    
    # 3. Σύνδεση συσκευής
    print("\n3️⃣ Connecting Dexcom G6...")
    connection = client.connect_device("dexcom_g6", "patient_001")
    print(f"   Status: {connection['status']}")
    print(f"   Device: {connection['device']}")
    
    # 4. Πρόβλεψη γλυκόζης
    print("\n4️⃣ Predicting glucose (30 min)...")
    prediction = client.predict_glucose(30, "patient_001")
    print(f"   Current: {prediction['current_glucose']:.0f} mg/dL")
    print(f"   Predicted: {prediction['predicted_glucose']:.0f} mg/dL")
    print(f"   Trend: {prediction['trend']}")
    print(f"   Risk: {prediction['risk_level']}")
    print(f"   Confidence: {prediction['confidence']:.0f}%")
    
    # 5. Συστάσεις
    print("\n5️⃣ Getting recommendations...")
    recs = client.get_recommendations("patient_001")
    print(f"   Found {len(recs['recommendations'])} recommendations:")
    for i, rec in enumerate(recs['recommendations'][:3], 1):
        print(f"   {i}. {rec['action']}")
        print(f"      Reason: {rec['reason']}")
        print(f"      Priority: {rec['priority']}")
    
    # 6. Datasets
    print("\n6️⃣ Available datasets...")
    datasets = client.list_datasets()
    print(f"   Total: {datasets['count']} datasets")
    for ds in datasets['datasets'][:3]:
        print(f"   - {ds['name']}: {ds['patients']} patients")
    
    print("\n✅ API test completed successfully!")
    

def demo_real_time_monitoring():
    """Demo real-time monitoring."""
    print("\n📊 Real-Time Glucose Monitoring Demo")
    print("=" * 50)
    
    client = DigitalTwinAPIClient()
    
    # Connect device
    client.connect_device("dexcom_g6", "demo_patient")
    
    # Monitor for 5 iterations
    for i in range(5):
        print(f"\n⏰ Reading {i+1}/5 at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get prediction
        pred = client.predict_glucose(15)
        
        # Display with emoji based on risk
        risk_emoji = {
            "low": "✅",
            "medium": "⚠️",
            "high": "🚨"
        }.get(pred['risk_level'], "❓")
        
        print(f"   {risk_emoji} Glucose: {pred['current_glucose']:.0f} → {pred['predicted_glucose']:.0f} mg/dL")
        print(f"   Trend: {pred['trend']}")
        
        # Show top recommendation
        recs = client.get_recommendations()
        if recs['recommendations']:
            print(f"   💡 Tip: {recs['recommendations'][0]['action']}")
        
        # Wait 5 seconds
        if i < 4:
            time.sleep(5)
    
    print("\n✅ Monitoring demo completed!")


if __name__ == "__main__":
    try:
        # Basic demo
        main()
        
        # Real-time demo
        demo_real_time_monitoring()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API is running: python -m sdk.api")
    except Exception as e:
        print(f"\n❌ Error: {e}") 