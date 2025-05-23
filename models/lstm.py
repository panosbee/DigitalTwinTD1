"""
LSTM μοντέλο για πρόβλεψη γλυκόζης.

Βασίζεται σε PyTorch και υποστηρίζει:
- Πολυπαραγοντική είσοδο (CGM, γεύματα, ινσουλίνη, άσκηση)  
- Προβλέψεις πολλαπλών ορίζοντων
- Bidirectional και stacked LSTM architectures
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import BaseModel


class LSTMNetwork(nn.Module):
    """PyTorch LSTM δίκτυο για πρόβλεψη γλυκόζης."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 output_horizon: int = 12):  # 12 σημεία (60 min / 5 min intervals)
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_horizon = output_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_horizon)
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Παίρνουμε την τελευταία έξοδο της ακολουθίας
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Πρόβλεψη για τον ορίζοντα
        predictions = self.fc(last_output)  # (batch_size, output_horizon)
        
        return predictions


class LSTMModel(BaseModel):
    """
    LSTM μοντέλο για πρόβλεψη γλυκόζης.
    
    Παράδειγμα χρήσης:
    >>> model = LSTMModel(sequence_length=12, hidden_size=64)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test, horizon=60)
    """
    
    def __init__(self,
                 sequence_length: int = 12,  # 1 ώρα σε 5-min intervals
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 device: Optional[str] = None):
        """
        Αρχικοποίηση LSTM μοντέλου.
        
        Args:
            sequence_length: Μήκος εισόδου σε time steps
            hidden_size: Μέγεθος κρυμμένων επιπέδων
            num_layers: Αριθμός LSTM επιπέδων
            dropout: Ποσοστό dropout
            bidirectional: Αν True, χρησιμοποιεί bidirectional LSTM
            learning_rate: Ρυθμός μάθησης
            batch_size: Μέγεθος batch
            epochs: Αριθμός epochs
            patience: Patience για early stopping
            device: CUDA device ('cuda' ή 'cpu')
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Ορισμός device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Αρχικοποίηση
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}
        
    def _prepare_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Δημιουργία ακολουθιών για LSTM."""
        X_sequences = []
        y_sequences = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq = X.iloc[i:i + self.sequence_length].values
            X_sequences.append(X_seq)
            
            if y is not None:
                # Για training, παίρνουμε την επόμενη τιμή
                if i + self.sequence_length < len(y):
                    y_seq = y.iloc[i + self.sequence_length]
                    y_sequences.append(y_seq)
                else:
                    break
        
        X_sequences = np.array(X_sequences)
        if y_sequences:
            y_sequences = np.array(y_sequences)
        
        return X_sequences, y_sequences
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LSTMModel':
        """
        Εκπαίδευση του LSTM μοντέλου.
        
        Args:
            X: Features DataFrame
            y: Target Series (CGM values)
            **kwargs: Επιπλέον παράμετροι
        """
        self.feature_names = X.columns.tolist()
        
        # Κανονικοποίηση δεδομένων
        X_scaled = pd.DataFrame(
            self.scaler_X.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        y_scaled = pd.Series(
            self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
            index=y.index
        )
        
        # Δημιουργία ακολουθιών
        X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_scaled)
        
        if len(X_sequences) == 0:
            raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για δημιουργία ακολουθιών")
        
        # Μετατροπή σε PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)
        
        # Split σε train/validation
        val_split = kwargs.get('validation_split', 0.2)
        val_size = int(len(X_tensor) * val_split)
        train_size = len(X_tensor) - val_size
        
        train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
        val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Δημιουργία μοντέλου
        input_size = X_sequences.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            output_horizon=1  # Για τώρα προβλέπουμε 1 βήμα
        ).to(self.device)
        
        # Loss και optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Αποθήκευση καλύτερου μοντέλου
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping στο epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self.model.train()
        
        # Φόρτωση καλύτερου μοντέλου
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        
        # Αποθήκευση ιστορικού
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, horizon: int = 30, **kwargs) -> np.ndarray:
        """
        Πρόβλεψη γλυκόζης.
        
        Args:
            X: Features DataFrame
            horizon: Ορίζοντας πρόβλεψης σε λεπτά
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Προβλέψεις γλυκόζης
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")
        
        # Κανονικοποίηση
        X_scaled = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Δημιουργία ακολουθιών
        X_sequences, _ = self._prepare_sequences(X_scaled)
        
        if len(X_sequences) == 0:
            raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για πρόβλεψη")
        
        # Πρόβλεψη
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)
            
            # Για κάθε ακολουθία, κάνουμε πρόβλεψη
            for i in range(len(X_tensor)):
                # Παίρνουμε τα τελευταία sequence_length σημεία
                current_seq = X_tensor[i:i+1]  # (1, seq_len, features)
                
                # Πρόβλεψη για έναν ορίζοντα
                pred = self.model(current_seq)
                predictions.append(pred.cpu().numpy())
        
        # Μετατροπή σε numpy array και αποκανονικοποίηση
        predictions = np.array(predictions).squeeze()
        if predictions.ndim == 0:
            predictions = predictions.reshape(1)
        
        # Αποκανονικοποίηση
        predictions_rescaled = self.scaler_y.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        return predictions_rescaled
    
    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        return {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience
        }
    
    def set_params(self, **params) -> 'LSTMModel':
        """Ορισμός παραμέτρων του μοντέλου."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Άγνωστη παράμετρος: {key}")
        
        # Επαναρχικοποίηση αν χρειάζεται
        if any(param in params for param in ['hidden_size', 'num_layers', 'dropout', 'bidirectional']):
            self.model = None
            self.is_fitted = False
        
        return self
    
    def get_training_history(self) -> Dict:
        """Επιστροφή ιστορικού εκπαίδευσης."""
        return self.training_history.copy() 