"""
Προηγμένα μοντέλα ψηφιακού διδύμου με state-of-the-art αρχιτεκτονικές.

Περιλαμβάνει:
- Mamba/State Space Models για long-range dependencies
- Neural ODEs για continuous-time modeling
- Multi-modal learning (CGM + wearables + lifestyle)
- Diffusion models για uncertainty quantification
- Federated learning για privacy-preserving training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, List, Tuple, Union
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import BaseModel

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from torchdyn.core import NeuralODE

    NEURAL_ODE_AVAILABLE = True
except ImportError:
    NEURAL_ODE_AVAILABLE = False


class StateSpaceLayer(nn.Module):
    """
    Custom State Space Model για continuous glucose monitoring.
    Βασίζεται στην αρχιτεκτονική Mamba αλλά προσαρμοσμένη για διαβήτη.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        d_inner = int(self.expand * d_model)

        # Projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # State space parameters (learnable)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Glucose-specific parameters
        self.glucose_gate = nn.Linear(d_model, d_model)
        self.insulin_gate = nn.Linear(d_model, d_model)

    def forward(self, x, glucose_features=None, insulin_features=None):
        """
        Args:
            x: (B, L, D) input sequence
            glucose_features: (B, L, D) glucose-specific features
            insulin_features: (B, L, D) insulin-specific features
        """
        batch, length, dim = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :length]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # Apply SiLU activation
        x = F.silu(x)

        # State space computation
        A_b_C = self.x_proj(x)  # (B, L, 2*d_state)
        A, B_C = A_b_C.split([self.d_state, self.d_state], dim=-1)

        # Discretization
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        # Apply glucose and insulin gates if provided
        if glucose_features is not None:
            glucose_gate = torch.sigmoid(self.glucose_gate(glucose_features))
            x = x * glucose_gate

        if insulin_features is not None:
            insulin_gate = torch.sigmoid(self.insulin_gate(insulin_features))
            x = x * insulin_gate

        # Selective scan (simplified version)
        # In practice, this would use the full selective scan algorithm
        y = self._selective_scan(x, A, B_C, dt)

        # Apply gate and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def _selective_scan(self, x, A, B_C, dt):
        """Simplified selective scan operation."""
        batch, length, d_inner = x.shape

        # Initialize state
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(length):
            # Discretize
            A_i = torch.exp(A[:, i] * dt[:, i].unsqueeze(-1))  # (B, d_state)
            B_i = B_C[:, i]  # (B, d_state)

            # Update state
            h = A_i * h + B_i * x[:, i].unsqueeze(-1)  # (B, d_state)

            # Output
            y_i = (h * B_i).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class MambaModel(BaseModel):
    """
    State-of-the-art Mamba μοντέλο για glucose prediction.

    Χρησιμοποιεί selective state space models για efficient modeling
    των long-range dependencies στα CGM δεδομένα.
    """

    def __init__(
        self,
        sequence_length: int = 288,  # 24 ώρες
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: Optional[str] = None,
    ):

        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def _build_model(self, input_size: int):
        """Δημιουργία του Mamba network."""

        class MambaNetwork(nn.Module):
            def __init__(self, input_size, d_model, n_layers, d_state, dropout):
                super().__init__()

                # Input embedding
                self.input_proj = nn.Linear(input_size, d_model)

                # Feature extractors για διαφορετικά τύπου δεδομένων
                self.glucose_encoder = nn.Linear(1, d_model // 4)
                self.carb_encoder = nn.Linear(1, d_model // 4)
                self.insulin_encoder = nn.Linear(1, d_model // 4)
                self.activity_encoder = nn.Linear(1, d_model // 4)

                # Mamba layers
                self.layers = nn.ModuleList(
                    [StateSpaceLayer(d_model, d_state) for _ in range(n_layers)]
                )

                # Layer norms
                self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

                # Output layers με uncertainty estimation
                self.output_mean = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                )

                self.output_var = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                    nn.Softplus(),  # Ensure positive variance
                )

                self.dropout = nn.Dropout(dropout)

            def forward(self, x, return_uncertainty=False):
                # Input projection
                x = self.input_proj(x)
                x = self.dropout(x)

                # Extract specific features (assuming certain column order)
                if x.shape[-1] >= 4:
                    glucose_feat = self.glucose_encoder(x[..., :1])
                    carb_feat = self.carb_encoder(x[..., 1:2])
                    insulin_feat = self.insulin_encoder(x[..., 2:3])
                    activity_feat = self.activity_encoder(x[..., 3:4])
                else:
                    glucose_feat = insulin_feat = None

                # Apply Mamba layers
                for layer, norm in zip(self.layers, self.layer_norms):
                    residual = x
                    x = layer(x, glucose_feat, insulin_feat)
                    x = norm(x + residual)
                    x = self.dropout(x)

                # Global average pooling
                x = x.mean(dim=1)

                # Output with uncertainty
                mean = self.output_mean(x)

                if return_uncertainty:
                    var = self.output_var(x)
                    return mean, var

                return mean

        return MambaNetwork(input_size, self.d_model, self.n_layers, self.d_state, self.dropout)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "MambaModel":
        """Εκπαίδευση του Mamba μοντέλου."""
        self.feature_names = X.columns.tolist()

        # Normalization
        X_scaled = pd.DataFrame(self.scaler_X.fit_transform(X), columns=X.columns, index=X.index)
        y_scaled = pd.Series(
            self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index
        )

        # Create sequences
        X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_scaled)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)

        # Data loaders
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = self._build_model(X_sequences.shape[2]).to(self.device)

        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training loop με uncertainty-aware loss
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward with uncertainty
                mean, var = self.model(batch_X, return_uncertainty=True)

                # Negative log-likelihood loss (for uncertainty)
                loss = 0.5 * torch.log(var) + 0.5 * (batch_y.unsqueeze(-1) - mean) ** 2 / var
                loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self, X: pd.DataFrame, horizon: int = 30, return_uncertainty: bool = False, **kwargs
    ) -> np.ndarray:
        """Πρόβλεψη με uncertainty estimation."""
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")

        X_scaled = pd.DataFrame(self.scaler_X.transform(X), columns=X.columns, index=X.index)

        X_sequences, _ = self._prepare_sequences(X_scaled)

        self.model.eval()
        predictions = []
        uncertainties = []

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)

            for i in range(len(X_tensor)):
                current_seq = X_tensor[i : i + 1]

                if return_uncertainty:
                    mean, var = self.model(current_seq, return_uncertainty=True)
                    uncertainties.append(var.cpu().numpy())
                else:
                    mean = self.model(current_seq)

                predictions.append(mean.cpu().numpy())

        predictions = np.array(predictions).squeeze()
        predictions_rescaled = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        if return_uncertainty:
            uncertainties = np.array(uncertainties).squeeze()
            uncertainties_rescaled = uncertainties * (self.scaler_y.scale_[0] ** 2)
            return predictions_rescaled, uncertainties_rescaled

        return predictions_rescaled

    def _prepare_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Δημιουργία ακολουθιών για Mamba."""
        X_sequences = []
        y_sequences = [] if y is not None else None

        for i in range(len(X) - self.sequence_length + 1):
            X_seq = X.iloc[i : i + self.sequence_length].values
            X_sequences.append(X_seq)

            if y is not None and i + self.sequence_length < len(y):
                y_sequences.append(y.iloc[i + self.sequence_length])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences else None

        return X_sequences, y_sequences

    def get_params(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def set_params(self, **params) -> "MambaModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class NeuralODEModel(BaseModel):
    """
    Neural ODE μοντέλο για continuous-time glucose dynamics.

    Μοντελοποιεί τη γλυκόζη ως continuous differential equation
    που μαθαίνεται από τα δεδομένα.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        integration_time: float = 1.0,
        solver: str = "dopri5",
        learning_rate: float = 0.001,
        epochs: int = 100,
        device: Optional[str] = None,
    ):

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.integration_time = integration_time
        self.solver = solver
        self.learning_rate = learning_rate
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _build_ode_func(self, input_dim: int):
        """Δημιουργία της ODE function."""

        class ODEFunc(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super().__init__()

                layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
                for _ in range(num_layers - 2):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
                layers.append(nn.Linear(hidden_dim, input_dim))

                self.net = nn.Sequential(*layers)

                # Glucose-specific constraints
                self.glucose_constraint = nn.Parameter(torch.tensor(0.1))

            def forward(self, t, x):
                # x shape: (batch, features)
                dx_dt = self.net(x)

                # Apply physiological constraints
                # Glucose should not change too rapidly
                glucose_change = dx_dt[:, 0] * torch.sigmoid(self.glucose_constraint)
                dx_dt = torch.cat([glucose_change.unsqueeze(1), dx_dt[:, 1:]], dim=1)

                return dx_dt

        return ODEFunc(input_dim, self.hidden_dim, self.num_layers)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "NeuralODEModel":
        """Εκπαίδευση του Neural ODE μοντέλου."""
        if not NEURAL_ODE_AVAILABLE:
            raise ImportError("Το torchdyn είναι απαραίτητο για Neural ODE. pip install torchdyn")

        # Προετοιμασία δεδομένων
        X_scaled = self.scaler.fit_transform(X.values)

        # Δημιουργία time points
        t_span = torch.linspace(0, self.integration_time, len(X))

        # Build model
        ode_func = self._build_ode_func(X_scaled.shape[1]).to(self.device)
        self.model = NeuralODE(ode_func, solver=self.solver).to(self.device)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Initial condition
            x0 = X_tensor[0:1]  # First observation

            # Integrate ODE
            trajectory = self.model(x0, t_span)

            # Extract glucose predictions
            glucose_pred = trajectory[:, 0, 0]  # First feature (glucose)

            # Loss
            loss = F.mse_loss(glucose_pred, y_tensor)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, horizon: int = 30, **kwargs) -> np.ndarray:
        """Πρόβλεψη με Neural ODE."""
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")

        X_scaled = self.scaler.transform(X.values)

        # Time points for prediction
        t_pred = torch.linspace(0, horizon / 60, horizon // 5)  # Convert minutes to hours

        with torch.no_grad():
            x0 = torch.FloatTensor(X_scaled[-1:]).to(self.device)
            trajectory = self.model(x0, t_pred)
            predictions = trajectory[:, 0, 0].cpu().numpy()

        return predictions

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "integration_time": self.integration_time,
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def set_params(self, **params) -> "NeuralODEModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class MultiModalModel(BaseModel):
    """
    Multi-modal μοντέλο που συνδυάζει:
    - CGM δεδομένα
    - Wearable data (heart rate, sleep, activity)
    - Lifestyle factors (stress, meals, medications)
    - Environmental data (weather, location)
    """

    def __init__(
        self,
        cgm_encoder_dim: int = 128,
        wearable_encoder_dim: int = 64,
        lifestyle_encoder_dim: int = 32,
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        device: Optional[str] = None,
    ):

        self.cgm_encoder_dim = cgm_encoder_dim
        self.wearable_encoder_dim = wearable_encoder_dim
        self.lifestyle_encoder_dim = lifestyle_encoder_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scalers = {}
        self.is_fitted = False

    def _build_model(self, input_dims: Dict[str, int]):
        """Δημιουργία multi-modal network."""

        class MultiModalNetwork(nn.Module):
            def __init__(
                self,
                input_dims,
                cgm_dim,
                wearable_dim,
                lifestyle_dim,
                fusion_dim,
                num_heads,
                dropout,
            ):
                super().__init__()

                # Modality-specific encoders
                self.cgm_encoder = nn.Sequential(
                    nn.Linear(input_dims["cgm"], cgm_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(cgm_dim, cgm_dim),
                )

                self.wearable_encoder = nn.Sequential(
                    nn.Linear(input_dims["wearable"], wearable_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(wearable_dim, wearable_dim),
                )

                self.lifestyle_encoder = nn.Sequential(
                    nn.Linear(input_dims["lifestyle"], lifestyle_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(lifestyle_dim, lifestyle_dim),
                )

                # Cross-modal attention
                total_dim = cgm_dim + wearable_dim + lifestyle_dim
                self.cross_attention = nn.MultiheadAttention(
                    total_dim, num_heads, dropout=dropout, batch_first=True
                )

                # Fusion network
                self.fusion = nn.Sequential(
                    nn.Linear(total_dim, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim // 2, 1),
                )

                # Attention weights για interpretability
                self.attention_weights = None

            def forward(self, cgm_data, wearable_data, lifestyle_data):
                # Encode each modality
                cgm_encoded = self.cgm_encoder(cgm_data)
                wearable_encoded = self.wearable_encoder(wearable_data)
                lifestyle_encoded = self.lifestyle_encoder(lifestyle_data)

                # Concatenate modalities
                fused = torch.cat([cgm_encoded, wearable_encoded, lifestyle_encoded], dim=-1)

                # Self-attention για cross-modal interactions
                if fused.dim() == 2:
                    fused = fused.unsqueeze(1)  # Add sequence dimension

                attended, self.attention_weights = self.cross_attention(fused, fused, fused)

                # Final prediction
                output = self.fusion(attended.squeeze(1))

                return output

            def get_attention_weights(self):
                return self.attention_weights

        return MultiModalNetwork(
            input_dims,
            self.cgm_encoder_dim,
            self.wearable_encoder_dim,
            self.lifestyle_encoder_dim,
            self.fusion_dim,
            self.num_heads,
            self.dropout,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, modality_columns: Dict[str, List[str]], **kwargs
    ) -> "MultiModalModel":
        """
        Εκπαίδευση multi-modal μοντέλου.

        Args:
            modality_columns: Dictionary με τις στήλες για κάθε modality
                            {'cgm': [...], 'wearable': [...], 'lifestyle': [...]}
        """

        # Διαχωρισμός δεδομένων ανά modality
        modality_data = {}
        input_dims = {}

        for modality, columns in modality_columns.items():
            available_columns = [col for col in columns if col in X.columns]
            if available_columns:
                modality_data[modality] = X[available_columns]

                # Normalization
                scaler = StandardScaler()
                modality_data[modality] = pd.DataFrame(
                    scaler.fit_transform(modality_data[modality]),
                    columns=available_columns,
                    index=X.index,
                )
                self.scalers[modality] = scaler
                input_dims[modality] = len(available_columns)

        # Build model
        self.model = self._build_model(input_dims).to(self.device)

        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Convert to tensors
        tensors = {}
        for modality, data in modality_data.items():
            tensors[modality] = torch.FloatTensor(data.values).to(self.device)

        y_tensor = torch.FloatTensor(y.values).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            output = self.model(
                tensors.get("cgm", torch.zeros(len(y), 1, device=self.device)),
                tensors.get("wearable", torch.zeros(len(y), 1, device=self.device)),
                tensors.get("lifestyle", torch.zeros(len(y), 1, device=self.device)),
            )

            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        self.modality_columns = modality_columns
        return self

    def predict(self, X: pd.DataFrame, horizon: int = 30, **kwargs) -> np.ndarray:
        """Πρόβλεψη με multi-modal μοντέλο."""
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")

        # Prepare modality data
        tensors = {}
        for modality, columns in self.modality_columns.items():
            available_columns = [col for col in columns if col in X.columns]
            if available_columns and modality in self.scalers:
                data = self.scalers[modality].transform(X[available_columns])
                tensors[modality] = torch.FloatTensor(data).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                tensors.get("cgm", torch.zeros(len(X), 1, device=self.device)),
                tensors.get("wearable", torch.zeros(len(X), 1, device=self.device)),
                tensors.get("lifestyle", torch.zeros(len(X), 1, device=self.device)),
            )

        return output.cpu().numpy().flatten()

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Επιστροφή attention weights για interpretability."""
        if self.model and hasattr(self.model, "get_attention_weights"):
            return self.model.get_attention_weights()
        return None

    def get_params(self) -> Dict[str, Any]:
        return {
            "cgm_encoder_dim": self.cgm_encoder_dim,
            "wearable_encoder_dim": self.wearable_encoder_dim,
            "lifestyle_encoder_dim": self.lifestyle_encoder_dim,
            "fusion_dim": self.fusion_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def set_params(self, **params) -> "MultiModalModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
