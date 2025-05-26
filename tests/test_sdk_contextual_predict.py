# tests/test_sdk_contextual_predict.py
import pytest
import numpy as np
from datetime import datetime
from typing import List # Added import for List

try:
    from sdk.core import DigitalTwinSDK, Prediction
    from agent import CognitiveAgent, GlucoseEncoder, VectorMemoryStore
except ImportError:
    # Fallback for local testing if structure changes or not installed
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sdk.core import DigitalTwinSDK, Prediction
    from agent import CognitiveAgent, GlucoseEncoder, VectorMemoryStore

@pytest.fixture
def sdk_with_agent():
    """Provides a DigitalTwinSDK instance with the agent enabled."""
    # Configure agent with minimal params for testing
    agent_config = {
        "encoder_params": {"input_dim": 1, "embedding_dim": 16, "hidden_dim": 32},
        "memory_store_params": {"embedding_dim": 16}
    }
    sdk = DigitalTwinSDK(mode='test', use_agent=True, agent_config=agent_config)
    # Ensure a device is connected so predict_glucose doesn't fail early
    sdk.connect_device("mock_cgm", device_id="contextual_mock_cgm")
    return sdk

@pytest.fixture
def sample_glucose_window() -> List[float]:
    return [100.0, 105.0, 110.0, 112.0, 115.0, 113.0, 110.0, 108.0, 105.0, 100.0, 98.0, 95.0]

def test_contextual_predict_smoke(sdk_with_agent, sample_glucose_window):
    """
    Smoke test for the contextual_predict method.
    Ensures it runs without crashing and returns a Prediction object.
    """
    assert sdk_with_agent.use_agent is True
    assert sdk_with_agent.agent is not None
    
    prediction_obj = sdk_with_agent.contextual_predict(
        glucose_history_window=sample_glucose_window,
        horizon_minutes=30
    )
    
    assert isinstance(prediction_obj, Prediction)
    assert isinstance(prediction_obj.values, list)
    assert len(prediction_obj.values) > 0
    assert isinstance(prediction_obj.timestamp, datetime)
    assert prediction_obj.horizon_minutes == 30
    
    # Check if contextual info was added (even if it's an error message or a found pattern)
    assert isinstance(prediction_obj.risk_alerts, list)
    # A basic check, can be made more specific if we know what contextual_info to expect
    # For now, if agent ran, it should add something or at least not remove existing alerts.
    # The base predict_glucose might produce its own alerts.
    # If contextual_info was added, risk_alerts list might be longer or contain specific strings.

def test_contextual_predict_agent_finds_pattern(sdk_with_agent, sample_glucose_window):
    """
    Test that contextual_predict includes information when a similar pattern is found.
    """
    # Add the sample window to the agent's memory so it can be found
    sdk_with_agent.agent.process_glucose_window(sample_glucose_window, timestamp=datetime.now(), extra_metadata={"event_type": "test_pattern"})

    prediction_obj = sdk_with_agent.contextual_predict(
        glucose_history_window=sample_glucose_window, # Query with the same window
        horizon_minutes=30
    )
    
    assert isinstance(prediction_obj, Prediction)
    assert isinstance(prediction_obj.risk_alerts, list)
    
    found_context_message = False
    for alert in prediction_obj.risk_alerts:
        if "Context: Found similar past pattern" in alert:
            found_context_message = True
            break
    assert found_context_message, "Contextual info about found pattern missing from risk_alerts"

def test_contextual_predict_agent_disabled(sample_glucose_window):
    """
    Test contextual_predict when agent is not enabled.
    It should behave like predict_glucose.
    """
    sdk_no_agent = DigitalTwinSDK(mode='test', use_agent=False)
    sdk_no_agent.connect_device("mock_cgm", device_id="no_agent_mock_cgm")

    # Get base prediction for comparison
    base_pred = sdk_no_agent.predict_glucose(horizon_minutes=30)

    contextual_pred = sdk_no_agent.contextual_predict(
        glucose_history_window=sample_glucose_window,
        horizon_minutes=30
    )
    
    assert isinstance(contextual_pred, Prediction)
    # Ensure no agent-specific context was added
    has_agent_context_message = any("Context:" in alert for alert in (contextual_pred.risk_alerts or []))
    assert not has_agent_context_message, "Agent context message found when agent was disabled"
    
    # It should essentially return the same as predict_glucose
    assert contextual_pred.values == base_pred.values
    assert contextual_pred.timestamp is not None # Timestamps will be different, just check existence
    assert contextual_pred.horizon_minutes == base_pred.horizon_minutes