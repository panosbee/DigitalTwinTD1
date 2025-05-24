SDK Core API Reference
======================

Main SDK Class
--------------

.. automodule:: sdk.core
   :members:
   :undoc-members:
   :show-inheritance:

DigitalTwinSDK
~~~~~~~~~~~~~~

.. autoclass:: sdk.core.DigitalTwinSDK
   :members:
   :special-members: __init__
   :show-inheritance:
   
   .. rubric:: Initialization
   
   .. code-block:: python
   
      from sdk import DigitalTwinSDK
      
      # Production mode with real-time predictions
      sdk = DigitalTwinSDK(mode='production')
      
      # Research mode with advanced features
      sdk = DigitalTwinSDK(mode='research')
      
      # Mobile mode optimized for battery life
      sdk = DigitalTwinSDK(mode='mobile')
   
   .. rubric:: Device Connection
   
   .. automethod:: connect_device
   .. automethod:: disconnect_device
   .. automethod:: get_device_status
   
   .. rubric:: Glucose Prediction
   
   .. automethod:: predict_glucose
   .. automethod:: get_glucose_history
   .. automethod:: assess_risk
   
   .. rubric:: Recommendations
   
   .. automethod:: get_recommendations
   .. automethod:: suggest_insulin_dose
   .. automethod:: suggest_carb_intake
   
   .. rubric:: Clinical Features
   
   .. automethod:: generate_clinical_report
   .. automethod:: export_data
   .. automethod:: run_virtual_trial

Data Classes
------------

PredictionResult
~~~~~~~~~~~~~~~~

.. autoclass:: sdk.core.PredictionResult
   :members:
   :show-inheritance:

Recommendation
~~~~~~~~~~~~~~

.. autoclass:: sdk.core.Recommendation
   :members:
   :show-inheritance:

ClinicalReport
~~~~~~~~~~~~~~

.. autoclass:: sdk.core.ClinicalReport
   :members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: sdk.core.validate_glucose_data
.. autofunction:: sdk.core.calculate_time_in_range
.. autofunction:: sdk.core.estimate_hba1c

Constants
---------

.. autodata:: sdk.core.GLUCOSE_TARGET_RANGES
.. autodata:: sdk.core.RISK_LEVELS
.. autodata:: sdk.core.SUPPORTED_DEVICES

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from sdk import DigitalTwinSDK
   
   # Initialize and connect
   sdk = DigitalTwinSDK()
   sdk.connect_device('dexcom_g6')
   
   # Get prediction
   prediction = sdk.predict_glucose(horizon_minutes=30)
   print(f"Glucose in 30 min: {prediction.value} mg/dL")
   print(f"Confidence: {prediction.confidence}%")
   
   # Get recommendations
   recs = sdk.get_recommendations()
   for rec in recs:
       print(f"- {rec.action}: {rec.reason}")

Virtual Clinical Trial
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run a virtual trial
   results = sdk.run_virtual_trial(
       population_size=1000,
       duration_days=90,
       interventions=['cgm_alerts', 'ai_recommendations']
   )
   
   print(f"Time in range improved by: {results.tir_improvement}%")
   print(f"Hypoglycemia reduced by: {results.hypo_reduction}%")

See Also
--------

- :doc:`devices` - Device integration documentation
- :doc:`models` - AI model documentation
- :doc:`clinical` - Clinical protocols documentation 