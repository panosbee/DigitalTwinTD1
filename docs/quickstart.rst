Quick Start Guide
=================

This guide will help you get started with the Digital Twin T1D SDK.

Basic Usage
-----------

Here's a simple example to get you started::

    from sdk import DigitalTwinSDK
    
    # Initialize the SDK
    sdk = DigitalTwinSDK(mode='production')
    
    # Connect to a device
    sdk.connect_device('dexcom_g6')
    
    # Make a prediction
    prediction = sdk.predict_glucose(horizon_minutes=30)
    
    print(f"Predicted glucose: {prediction.value} mg/dL")
    print(f"Risk level: {prediction.risk_level}")

Next Steps
----------

For more detailed examples, see the Examples section. 