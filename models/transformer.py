"""
Transformer μοντέλο για πρόβλεψη γλυκόζης.

Βασίζεται σε self-attention mechanism και υποστηρίζει:
- Temporal Fusion Transformer (TFT) architecture
- Multi-head attention για χρονοσειρές
- Συνδυασμό με LSTM για υβριδικές αρχιτεκτονικές
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, List, Tuple, Union  # Added Union
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding για Transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerBlock(nn.Module):
    """Βασικός Transformer block με self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attention_weights


class TimeSeriesTransformer(nn.Module):
    """Transformer για χρονοσειρές πρόβλεψης γλυκόζης."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_horizon: int = 12,
        max_seq_len: int = 200,
    ):
        super().__init__()

        self.d_model = d_model
        self.output_horizon = output_horizon

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_horizon),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Input projection and positional encoding
        src = self.input_projection(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)

        # Apply transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            src, attn_weights = layer(src, src_mask)
            attention_weights.append(attn_weights)

        # Global average pooling over sequence dimension
        src = torch.mean(src, dim=1)  # (batch_size, d_model)

        # Output projection
        output = self.output_projection(src)  # (batch_size, output_horizon)

        return output, attention_weights


class HybridTransformerLSTM(nn.Module):
    """Υβριδικό μοντέλο Transformer + LSTM όπως στη βιβλιογραφία."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
        output_horizon: int = 12,
    ):
        super().__init__()

        self.d_model = d_model
        self.output_horizon = output_horizon

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer για μακροχρόνιες εξαρτήσεις
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, d_model * 2, dropout)
                for _ in range(num_transformer_layers)
            ]
        )

        # LSTM για βραχυχρόνιες εξαρτήσεις
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Fusion layer
        self.fusion = nn.Linear(d_model + lstm_hidden_size, d_model)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_horizon),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size, seq_len, _ = src.shape

        # Input projection
        src_proj = self.input_projection(src) * np.sqrt(self.d_model)

        # Transformer path για μακροχρόνιες εξαρτήσεις
        transformer_out = src_proj.transpose(0, 1)  # (seq_len, batch_size, d_model)
        transformer_out = self.pos_encoder(transformer_out)
        transformer_out = transformer_out.transpose(0, 1)  # (batch_size, seq_len, d_model)

        for layer in self.transformer_layers:
            transformer_out, _ = layer(transformer_out)

        # Global average pooling για transformer
        transformer_pooled = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)

        # LSTM path για βραχυχρόνιες εξαρτήσεις
        lstm_out, _ = self.lstm(src_proj)
        lstm_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)

        # Fusion των δύο paths
        fused = torch.cat([transformer_pooled, lstm_last], dim=1)
        fused = self.fusion(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)

        # Output projection
        output = self.output_projection(fused)

        return output


class TransformerModel(BaseModel):
    """
    Transformer μοντέλο για πρόβλεψη γλυκόζης.

    Παράδειγμα χρήσης:
    >>> model = TransformerModel(sequence_length=24, d_model=128, hybrid=True)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test, horizon=60)
    """

    def __init__(
        self,
        sequence_length: int = 24,  # 2 ώρες σε 5-min intervals
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        hybrid: bool = False,  # Αν True, χρησιμοποιεί Transformer+LSTM
        device: Optional[str] = None,
    ):
        """
        Αρχικοποίηση Transformer μοντέλου.

        Args:
            sequence_length: Μήκος εισόδου σε time steps
            d_model: Διάσταση μοντέλου
            nhead: Αριθμός attention heads
            num_layers: Αριθμός Transformer layers
            dim_feedforward: Διάσταση feed-forward δικτύου
            dropout: Ποσοστό dropout
            learning_rate: Ρυθμός μάθησης
            batch_size: Μέγεθος batch
            epochs: Αριθμός epochs
            patience: Patience για early stopping
            hybrid: Αν True, χρησιμοποιεί υβριδική αρχιτεκτονική
            device: CUDA device
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.hybrid = hybrid

        # Ορισμός device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Αρχικοποίηση
        self.model: Optional[Union[TimeSeriesTransformer, HybridTransformerLSTM]] = (
            None  # Explicitly type hint
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}

    def _prepare_sequences(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Δημιουργία ακολουθιών για Transformer."""
        X_sequences = []
        y_sequences_list: List[Any] = []  # Initialize as a list

        for i in range(len(X) - self.sequence_length + 1):
            X_seq = X.iloc[i : i + self.sequence_length].values
            X_sequences.append(X_seq)

            if y is not None:
                if i + self.sequence_length < len(y):
                    y_seq = y.iloc[i + self.sequence_length]
                    y_sequences_list.append(y_seq)
                else:
                    if len(X_sequences) > len(y_sequences_list):
                        X_sequences.pop()
                    break

        final_X_sequences = np.array(X_sequences)

        final_y_sequences: Optional[np.ndarray] = None
        if y is not None:
            if y_sequences_list:
                final_y_sequences = np.array(y_sequences_list)
            else:
                if len(final_X_sequences) > 0 and not y_sequences_list:
                    final_X_sequences = np.array([])

        return final_X_sequences, final_y_sequences

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TransformerModel":
        """
        Εκπαίδευση του Transformer μοντέλου.

        Args:
            X: Features DataFrame
            y: Target Series (CGM values)
            **kwargs: Επιπλέον παράμετροι
        """
        self.feature_names = X.columns.tolist()

        # Κανονικοποίηση δεδομένων
        X_scaled = pd.DataFrame(self.scaler_X.fit_transform(X), columns=X.columns, index=X.index)
        y_scaled = pd.Series(
            self.scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)).flatten(), index=y.index
        )

        # Δημιουργία ακολουθιών
        X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_scaled)

        if len(X_sequences) == 0:
            raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για δημιουργία ακολουθιών")

        # Μετατροπή σε PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)

        # Split σε train/validation
        val_split = kwargs.get("validation_split", 0.2)
        val_size = int(len(X_tensor) * val_split)
        train_size = len(X_tensor) - val_size

        train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
        val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Δημιουργία μοντέλου
        input_size = X_sequences.shape[2]

        if self.hybrid:
            self.model = HybridTransformerLSTM(
                input_size=input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_transformer_layers=self.num_layers,
                lstm_hidden_size=self.d_model // 2,
                dropout=self.dropout,
                output_horizon=1,
            ).to(self.device)
        else:
            self.model = TimeSeriesTransformer(
                input_size=input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                output_horizon=1,
                max_seq_len=self.sequence_length * 2,
            ).to(self.device)

        assert self.model is not None, "Model should be initialized at this point in fit method."

        # Loss και optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, verbose=False
        )

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if self.hybrid:
                    outputs = self.model(batch_X)
                else:
                    outputs, _ = self.model(batch_X)

                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Gradient clipping για σταθερότητα
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if self.hybrid:
                        outputs = self.model(batch_X)
                    else:
                        outputs, _ = self.model(batch_X)

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
                torch.save(self.model.state_dict(), "best_transformer_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping στο epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            self.model.train()

        # Φόρτωση καλύτερου μοντέλου
        self.model.load_state_dict(
            torch.load("best_transformer_model.pth")
        )  # nosec B614 - Assuming trusted model file
        self.model.eval()

        # Αποθήκευση ιστορικού
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
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
        assert self.model is not None, "Model should be initialized if is_fitted is True."

        # Κανονικοποίηση
        X_scaled = pd.DataFrame(self.scaler_X.transform(X), columns=X.columns, index=X.index)

        # Δημιουργία ακολουθιών
        X_sequences, _ = self._prepare_sequences(X_scaled)

        if len(X_sequences) == 0:
            raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για πρόβλεψη")

        # Πρόβλεψη
        self.model.eval()
        predictions = []

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)

            for i in range(len(X_tensor)):
                current_seq = X_tensor[i : i + 1]

                if self.hybrid:
                    pred = self.model(current_seq)
                else:
                    pred, _ = self.model(current_seq)

                predictions.append(pred.cpu().numpy())

        # Μετατροπή και αποκανονικοποίηση
        predictions = np.array(predictions).squeeze()
        if predictions.ndim == 0:
            predictions = predictions.reshape(1)

        predictions_rescaled = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions_rescaled

    def get_attention_weights(self, X: pd.DataFrame) -> Optional[List[torch.Tensor]]:
        """Επιστροφή attention weights για ανάλυση."""
        if not self.is_fitted or self.hybrid:
            return None

        X_scaled = pd.DataFrame(self.scaler_X.transform(X), columns=X.columns, index=X.index)

        X_sequences, _ = self._prepare_sequences(X_scaled)
        if len(X_sequences) == 0:
            return None

        # After is_fitted and not hybrid checks, self.model should be a TimeSeriesTransformer
        assert self.model is not None, "Model should be fitted and not hybrid here."
        assert isinstance(
            self.model, TimeSeriesTransformer
        ), "Model must be a TimeSeriesTransformer to get attention weights."

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequences[:1]).to(
                self.device
            )  # Using only the first sequence for example
            _, attention_weights = self.model(X_tensor)

        return attention_weights

    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        return {
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "hybrid": self.hybrid,
        }

    def set_params(self, **params) -> "TransformerModel":
        """Ορισμός παραμέτρων του μοντέλου."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Άγνωστη παράμετρος: {key}")

        # Επαναρχικοποίηση αν χρειάζεται
        architecture_params = ["d_model", "nhead", "num_layers", "dim_feedforward", "hybrid"]
        if any(param in params for param in architecture_params):
            self.model = None
            self.is_fitted = False

        return self

    def get_training_history(self) -> Dict:
        """Επιστροφή ιστορικού εκπαίδευσης."""
        return self.training_history.copy()
