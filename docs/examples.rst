Examples
========

This section contains various examples of using the Digital Twin T1D SDK.

Glucose Prediction
------------------

Basic glucose prediction example::

    from sdk import DigitalTwinSDK
    import numpy as np
    
    sdk = DigitalTwinSDK()
    
    # Recent glucose history (last 2 hours)
    glucose_history = [120, 125, 130, 135, 140, 138, 135, 130]
    
    # Predict for next 30 minutes
    prediction = sdk.predict_glucose(
        glucose_history=glucose_history,
        horizon_minutes=30
    )

Device Integration
------------------

Example of integrating with a CGM device::

    from sdk.integrations import DeviceFactory
    
    # Create device instance
    device = DeviceFactory.create_device('dexcom_g6')
    
    # Connect to device
    if device.connect():
        # Get current reading
        current_glucose = device.get_current_glucose()
        print(f"Current glucose: {current_glucose} mg/dL")

More Examples Coming Soon
-------------------------

Additional examples will be added in future releases. 