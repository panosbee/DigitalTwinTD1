"""
🚀 Universal Digital Twin SDK για Διαβήτη Τύπου 1
================================================

Το SDK που θα αλλάξει τον κόσμο του διαβήτη!

Plug-and-play για:
- 🏭 Κατασκευαστές CGM/Insulin pumps
- 💻 Developers εφαρμογών υγείας  
- 🔬 Ερευνητές
- 👨‍⚕️ Γιατρούς και κλινικές
- 🏥 Νοσοκομεία
"""

from .core import DigitalTwinSDK, quick_predict, assess_glucose_risk
from .integrations import (
    DeviceIntegration,
    CGMDevice,
    InsulinPump,
    SmartWatch,
    DeviceFactory
)
from .clinical import ClinicalProtocols
from .datasets import (
    DiabetesDatasets,
    load_diabetes_data,
    list_available_datasets
)

__version__ = "1.0.0"
__all__ = [
    "DigitalTwinSDK",
    "DeviceIntegration", 
    "CGMDevice",
    "InsulinPump",
    "SmartWatch",
    "DeviceFactory",
    "ClinicalProtocols",
    "quick_predict",
    "assess_glucose_risk",
    "DiabetesDatasets",
    "load_diabetes_data",
    "list_available_datasets"
] 