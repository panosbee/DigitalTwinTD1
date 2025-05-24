"""
Integration tests for Digital Twin T1D SDK.

Tests the complete workflow from device connection to prediction.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from sdk import DigitalTwinSDK
from sdk.integrations import DeviceFactory


class TestSDKIntegration:
    """Integration tests for the complete SDK workflow."""
    
    @pytest.fixture
    def sdk(self):
        """Initialize SDK for testing."""
        return DigitalTwinSDK(mode='demo')
    
    @pytest.fixture
    def mock_glucose_data(self) -> List[float]:
        """Generate realistic glucose data."""
        # Simulate 24 hours of data (5-min intervals = 288 points)
        hours = 24
        points = hours * 12
        time = np.arange(points)
        
        # Base glucose with circadian rhythm
        base = 120 + 15 * np.sin(2 * np.pi * time / 288)
        
        # Add meal spikes
        meals = np.zeros(points)
        for meal_time in [96, 156, 228]:  # 8am, 1pm, 7pm
            if meal_time < points:
                spike = 40 * np.exp(-((time - meal_time - 24) ** 2) / (2 * 15 ** 2))
                meals += spike
        
        # Add noise
        noise = np.random.normal(0, 5, points)
        
        # Combine
        glucose = base + meals + noise
        return np.clip(glucose, 70, 250).tolist()
    
    def test_complete_workflow(self, sdk, mock_glucose_data):
        """Test the complete workflow from connection to prediction."""
        # 1. Connect to a mock device
        assert sdk.connect_device('mock_cgm') is True
        
        # 2. Set patient profile
        profile = sdk.set_patient_profile(
            age=35,
            weight=75,
            diabetes_type='T1D',
            diagnosis_date='2015-01-01'
        )
        assert profile['status'] == 'success'
        
        # 3. Load historical data
        sdk.load_historical_data(mock_glucose_data)
        
        # 4. Make a prediction
        prediction = sdk.predict_glucose(horizon_minutes=30)
        
        # Validate prediction structure
        assert 'value' in prediction
        assert 'confidence_interval' in prediction
        assert 'risk_level' in prediction
        assert 'timestamp' in prediction
        
        # Validate prediction values
        assert 40 <= prediction['value'] <= 400  # Physiological range
        assert prediction['risk_level'] in ['low', 'medium', 'high']
        
    def test_multi_model_predictions(self, sdk, mock_glucose_data):
        """Test predictions from multiple models."""
        sdk.load_historical_data(mock_glucose_data[-48:])  # Last 4 hours
        
        models = ['lstm', 'transformer', 'mamba']
        predictions = {}
        
        for model in models:
            pred = sdk.predict_glucose(
                horizon_minutes=30,
                model_name=model
            )
            predictions[model] = pred
        
        # All models should return valid predictions
        for model, pred in predictions.items():
            assert 40 <= pred['value'] <= 400
            
        # Ensemble should be close to individual predictions
        ensemble_pred = sdk.predict_glucose(
            horizon_minutes=30,
            model_name='ensemble'
        )
        
        values = [p['value'] for p in predictions.values()]
        assert min(values) <= ensemble_pred['value'] <= max(values)
    
    def test_real_time_monitoring(self, sdk):
        """Test real-time monitoring capabilities."""
        # Start monitoring
        sdk.start_real_time_monitoring()
        
        # Simulate 5 new readings
        for i in range(5):
            glucose = 120 + np.random.normal(0, 10)
            alert = sdk.process_real_time_reading(glucose)
            
            # Check if alerts are generated correctly
            if glucose < 70:
                assert alert['type'] == 'hypoglycemia'
            elif glucose > 180:
                assert alert['type'] == 'hyperglycemia'
            else:
                assert alert is None or alert['type'] == 'in_range'
        
        # Stop monitoring
        sdk.stop_real_time_monitoring()
    
    def test_clinical_recommendations(self, sdk, mock_glucose_data):
        """Test clinical recommendation generation."""
        sdk.load_historical_data(mock_glucose_data)
        
        # Get recommendations
        recommendations = sdk.get_recommendations()
        
        # Validate structure
        assert 'insulin_adjustment' in recommendations
        assert 'lifestyle_changes' in recommendations
        assert 'next_actions' in recommendations
        assert 'risk_assessment' in recommendations
        
        # Validate risk assessment
        risk = recommendations['risk_assessment']
        assert 'hypoglycemia_risk' in risk
        assert 'hyperglycemia_risk' in risk
        assert all(0 <= r <= 1 for r in risk.values())
    
    @pytest.mark.parametrize("device_type", [
        "dexcom_g6",
        "freestyle_libre",
        "medtronic_guardian"
    ])
    def test_device_compatibility(self, sdk, device_type):
        """Test compatibility with different CGM devices."""
        # Mock device should support all types
        device = DeviceFactory.create_device(device_type)
        
        # Test basic operations
        assert device.connect() is True
        assert 40 <= device.get_current_glucose() <= 400
        
        # Test integration with SDK
        assert sdk.connect_device(device_type) is True
    
    def test_error_handling(self, sdk):
        """Test error handling in various scenarios."""
        # Test with invalid device
        with pytest.raises(ValueError):
            sdk.connect_device('invalid_device')
        
        # Test prediction without data
        with pytest.raises(ValueError):
            sdk.predict_glucose()
        
        # Test with invalid patient profile
        with pytest.raises(ValueError):
            sdk.set_patient_profile(age=-5)
    
    def test_performance_requirements(self, sdk, mock_glucose_data):
        """Test that performance meets requirements."""
        import time
        
        sdk.load_historical_data(mock_glucose_data[-48:])
        
        # Measure prediction latency
        start = time.time()
        prediction = sdk.predict_glucose(horizon_minutes=30)
        latency = (time.time() - start) * 1000  # Convert to ms
        
        # Should be under 50ms as per requirements
        assert latency < 50, f"Prediction took {latency:.1f}ms, exceeding 50ms requirement"
    
    def test_data_privacy(self, sdk, mock_glucose_data):
        """Test data privacy features."""
        # Load data with privacy mode
        sdk.enable_privacy_mode()
        sdk.load_historical_data(mock_glucose_data)
        
        # Export should be anonymized
        export = sdk.export_data()
        assert 'patient_id' not in export
        assert 'anonymized_id' in export
        
        # Telemetry should be disabled
        assert sdk.is_telemetry_enabled() is False


class TestClinicalIntegration:
    """Test clinical protocol integration."""
    
    def test_clinical_alerts(self):
        """Test generation of clinical alerts."""
        from sdk.clinical import ClinicalProtocols
        
        protocols = ClinicalProtocols()
        
        # Test hypoglycemia alert
        glucose_values = [65, 62, 58, 55, 52]  # Declining glucose
        assessment = protocols.assess_glucose_control(
            glucose_values, 
            patient_category='adult_standard'
        )
        
        alerts = protocols.generate_clinical_alerts(assessment)
        
        # Should generate severe hypoglycemia alert
        assert any(alert.severity == 'critical' for alert in alerts)
        assert any('hypoglycemia' in alert.category for alert in alerts)
    
    def test_therapy_recommendations(self):
        """Test therapy adjustment recommendations."""
        from sdk.clinical import ClinicalProtocols
        
        protocols = ClinicalProtocols()
        
        # High glucose scenario
        glucose_values = [200] * 20  # Persistent hyperglycemia
        assessment = protocols.assess_glucose_control(
            glucose_values,
            patient_category='adult_standard'
        )
        
        current_therapy = {
            'basal_rate': 1.0,
            'carb_ratio': 10,
            'sensitivity_factor': 40
        }
        
        recommendations = protocols.recommend_therapy_adjustments(
            assessment, 
            current_therapy
        )
        
        # Should recommend basal adjustment
        assert any('basal_adjustment' in rec['type'] for rec in recommendations)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 