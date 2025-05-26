# tests/test_agent_encoder.py
import pytest
import numpy as np
import torch

# Ensure the agent package is discoverable.
# This might require adjusting sys.path or using `pip install -e .`
# For now, assuming the tests are run from the root of the project
# and the `agent` directory is directly under `src/digital_twin_t1d` (after src layout)
# or directly under the root if src layout is not yet applied.
# We will adjust imports if needed once the src layout is done.
try:
    from agent.encoder import GlucoseEncoder
except ImportError:
    # Fallback for current structure if src layout not yet done
    # This assumes 'agent' is a top-level directory for now.
    # This will need to be updated after src layout.
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from agent.encoder import GlucoseEncoder


@pytest.fixture
def encoder():
    """Provides a default GlucoseEncoder instance."""
    return GlucoseEncoder(input_dim=1, embedding_dim=32, hidden_dim=64, num_layers=1)


def test_encoder_initialization(encoder):
    assert encoder.input_dim == 1
    assert encoder.embedding_dim == 32
    assert encoder.hidden_dim == 64
    assert not encoder.is_trained


def test_encoder_forward_pass(encoder):
    # Batch size 2, sequence length 12, input_dim 1
    dummy_window = torch.randn(2, 12, 1)
    embedding = encoder.forward(dummy_window)
    assert embedding.shape == (2, 32), "Embedding shape mismatch"


def test_encoder_forward_pass_single_sequence_2d(encoder):
    # Batch size 1, sequence length 12 (passed as 2D)
    dummy_window = torch.randn(1, 12)
    embedding = encoder.forward(dummy_window)
    assert embedding.shape == (1, 32), "Embedding shape mismatch for 2D input"


def test_encoder_encode_list_input(encoder):
    glucose_list = [
        100.0,
        105.0,
        110.0,
        115.0,
        120.0,
        118.0,
        115.0,
        110.0,
        105.0,
        100.0,
        95.0,
        90.0,
    ]
    embedding = encoder.encode(glucose_list)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (32,), "Embedding shape mismatch for list input"


def test_encoder_encode_numpy_1d_input(encoder):
    glucose_np_1d = np.array(
        [100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    )
    embedding = encoder.encode(glucose_np_1d)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (32,), "Embedding shape mismatch for 1D numpy input"


def test_encoder_encode_numpy_2d_row_vector_input(encoder):
    glucose_np_2d_row = np.array(
        [[100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]]
    )
    embedding = encoder.encode(glucose_np_2d_row)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (32,), "Embedding shape mismatch for 2D row vector numpy input"


def test_encoder_encode_numpy_2d_col_vector_input(encoder):
    glucose_np_2d_col = np.array(
        [
            [100.0],
            [105.0],
            [110.0],
            [115.0],
            [120.0],
            [118.0],
            [115.0],
            [110.0],
            [105.0],
            [100.0],
            [95.0],
            [90.0],
        ]
    )
    embedding = encoder.encode(glucose_np_2d_col)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (32,), "Embedding shape mismatch for 2D col vector numpy input"


def test_encoder_encode_invalid_input_type(encoder):
    with pytest.raises(ValueError, match="glucose_window must be a list or NumPy array."):
        encoder.encode("not a list or array")


def test_encoder_encode_invalid_numpy_shape(encoder):
    glucose_np_invalid = np.array([[[100.0]]])  # 3D array
    with pytest.raises(ValueError, match="glucose_window must be a 1D array or a 2D array"):
        encoder.encode(glucose_np_invalid)


def test_encoder_training_placeholder(encoder, capsys):
    # This test just checks if the placeholder training runs without error
    # and sets the is_trained flag. A real test would mock the data_loader.
    encoder.train_encoder(data_loader=None, epochs=1)  # Pass None for data_loader
    captured = capsys.readouterr()
    assert "Placeholder: Training GlucoseEncoder for 1 epochs..." in captured.out
    assert "Placeholder: GlucoseEncoder training complete." in captured.out
    assert encoder.is_trained


def test_encode_consistency(encoder):
    window = [100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    embedding1 = encoder.encode(window)
    embedding2 = encoder.encode(window)
    assert np.array_equal(
        embedding1, embedding2
    ), "Embeddings for the same input are not consistent"


def test_encode_different_inputs_different_embeddings(encoder):
    window1 = [100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    window2 = [150.0, 160.0, 170.0, 180.0, 175.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0]
    embedding1 = encoder.encode(window1)
    embedding2 = encoder.encode(window2)
    assert not np.array_equal(
        embedding1, embedding2
    ), "Embeddings for different inputs are identical"
    # A more robust check would be cosine similarity not being 1, but this is fine for now.
    cosine_similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    assert cosine_similarity < 0.99999, "Embeddings for different inputs are too similar"
