"""
LSTM Model για glucose prediction - Alias για backward compatibility.
"""

# Import από το υπάρχον LSTM module
from .lstm import LSTMModel

# Export για το testing script
__all__ = ['LSTMModel'] 