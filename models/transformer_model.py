"""
Transformer Model για glucose prediction - Alias για backward compatibility.
"""

# Import από το υπάρχον Transformer module
from .transformer import TransformerModel

# Export για το testing script
__all__ = ['TransformerModel'] 