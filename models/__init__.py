"""
Μοντέλα ψηφιακού διδύμου για διαβήτη τύπου 1.

Περιλαμβάνει:
- Μηχανιστικά μοντέλα (UVA/Padova, Hovorka)
- ML μοντέλα (LSTM, Transformer)
- Baseline μοντέλα (ARIMA, Prophet)
- Advanced μοντέλα (Mamba, Neural ODE, Multi-Modal)
"""

from .mechanistic import MechanisticModel
from .lstm import LSTMModel
from .transformer import TransformerModel
from .baseline import ARIMAModel, ProphetModel

# Try to import advanced models
try:
    from .advanced import (
        MambaModel, MambaGlucosePredictor,
        NeuralODEModel, NeuralODEGlucoseModel,
        MultiModalModel, MultiModalTransformer,
        EdgeOptimizedModel
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = [
    'MechanisticModel',
    'LSTMModel', 
    'TransformerModel',
    'ARIMAModel',
    'ProphetModel'
]

# Add advanced models if available
if ADVANCED_AVAILABLE:
    __all__.extend([
        'MambaModel', 'MambaGlucosePredictor',
        'NeuralODEModel', 'NeuralODEGlucoseModel', 
        'MultiModalModel', 'MultiModalTransformer',
        'EdgeOptimizedModel'
    ])

__version__ = "2.0.0"
__author__ = "Digital Twin T1D Consortium" 