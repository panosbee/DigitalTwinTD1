# agent/encoder.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union

class GlucoseEncoder(nn.Module):
    def __init__(self, input_dim: int = 1, embedding_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Simple LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.is_trained = False # Placeholder for training status

    def forward(self, glucose_window: torch.Tensor) -> torch.Tensor:
        # glucose_window shape: (batch_size, sequence_length, input_dim)
        # If input_dim is 1, ensure it's (batch_size, sequence_length, 1)
        if glucose_window.ndim == 2: # (batch_size, sequence_length)
            glucose_window = glucose_window.unsqueeze(-1)
        
        lstm_out, (hidden, _) = self.lstm(glucose_window)
        # Use the last hidden state as the sequence representation
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We take the hidden state of the last layer
        sequence_embedding = hidden[-1] # (batch_size, hidden_dim)
        
        embedding = self.fc(sequence_embedding) # (batch_size, embedding_dim)
        return embedding

    def encode(self, glucose_window: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Encodes a window of glucose values into a cognitive fingerprint (embedding).

        Args:
            glucose_window: A list or NumPy array of glucose values.
                            Expected to be a 1D array representing a sequence.

        Returns:
            A NumPy array representing the embedding.
        """
        if not self.is_trained:
            # In a real scenario, the model should be trained.
            # For now, we'll allow encoding with initial weights for structure.
            # print("Warning: GlucoseEncoder is not trained. Using initial weights for encoding.")
            pass

        self.eval() # Set model to evaluation mode

        if isinstance(glucose_window, list):
            glucose_window = np.array(glucose_window, dtype=np.float32)
        
        if not isinstance(glucose_window, np.ndarray):
            raise ValueError("glucose_window must be a list or NumPy array.")

        if glucose_window.ndim == 1:
            # Reshape to (1, sequence_length, input_dim) for batch processing
            glucose_tensor = torch.tensor(glucose_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        elif glucose_window.ndim == 2 and glucose_window.shape[0] == 1: # Already (1, sequence_length)
             glucose_tensor = torch.tensor(glucose_window, dtype=torch.float32).unsqueeze(-1)
        elif glucose_window.ndim == 2 and glucose_window.shape[1] == 1: # (sequence_length, 1)
             glucose_tensor = torch.tensor(glucose_window, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError("glucose_window must be a 1D array or a 2D array of shape (1, seq_len) or (seq_len, 1).")

        with torch.no_grad():
            embedding_tensor = self.forward(glucose_tensor)
        
        return embedding_tensor.squeeze(0).cpu().numpy()

    def train_encoder(self, data_loader, epochs=10, learning_rate=0.001): # pragma: no cover
        """
        Placeholder for a training method.
        In a real scenario, this would involve a proper dataset and loss function
        (e.g., contrastive loss, autoencoder loss, or supervised task).
        """
        print(f"Placeholder: Training GlucoseEncoder for {epochs} epochs...")
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        # for epoch in range(epochs):
        #     epoch_loss = 0
        #     for batch_glucose in data_loader: 
        #         optimizer.zero_grad()
        #         embeddings = self.forward(batch_glucose)
        #         # reconstruction = self.decoder(embeddings) # Decoder needed
        #         # loss = nn.MSELoss()(reconstruction, batch_glucose)
        #         # loss.backward()
        #         # optimizer.step()
        #         # epoch_loss += loss.item()
        #     # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
        #     pass 

        self.is_trained = True
        print("Placeholder: GlucoseEncoder training complete.")

if __name__ == '__main__': # pragma: no cover
    # Example Usage
    encoder = GlucoseEncoder(input_dim=1, embedding_dim=32, hidden_dim=64)
    
    window1 = np.array([100, 105, 110, 115, 120, 118, 115, 110, 105, 100, 95, 90])
    window2 = np.array([150, 160, 170, 180, 175, 170, 160, 150, 140, 130, 120, 110])
    
    embedding1 = encoder.encode(window1)
    embedding2 = encoder.encode(window2)
    
    print("Embedding for Window 1:", embedding1.shape)
    print(embedding1)
    print("Embedding for Window 2:", embedding2.shape)
    print(embedding2)