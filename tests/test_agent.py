# tests/test_agent.py
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Adjust import based on project structure
try:
    from agent.agent import CognitiveAgent
    from agent.encoder import GlucoseEncoder
    from agent.memory_store import VectorMemoryStore
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from agent.agent import CognitiveAgent
    from agent.encoder import GlucoseEncoder
    from agent.memory_store import VectorMemoryStore

EMBEDDING_DIM_AGENT = 16  # Using a smaller dim for agent tests for speed


@pytest.fixture
def mock_encoder():
    encoder = MagicMock(spec=GlucoseEncoder)
    encoder.embedding_dim = EMBEDDING_DIM_AGENT
    # Configure the mock's encode method to return a consistent dummy embedding
    dummy_embedding = np.random.rand(EMBEDDING_DIM_AGENT).astype(np.float32)
    encoder.encode.return_value = dummy_embedding
    return encoder


@pytest.fixture
def mock_memory_store():
    store = MagicMock(spec=VectorMemoryStore)
    store.embedding_dim = EMBEDDING_DIM_AGENT
    store.add_embedding.return_value = None
    # Make len(store) work
    store.__len__.return_value = 0

    # Mock search_similar to return a list of tuples
    # (embedding, metadata, distance)
    dummy_search_result_embedding = np.random.rand(EMBEDDING_DIM_AGENT).astype(np.float32)
    dummy_metadata = {"id": "mock_search_result", "timestamp": datetime.now().isoformat()}
    dummy_distance = 0.1
    store.search_similar.return_value = [
        (dummy_search_result_embedding, dummy_metadata, dummy_distance)
    ]
    return store


@pytest.fixture
def cognitive_agent(mock_encoder, mock_memory_store):
    """Provides a CognitiveAgent instance with mocked dependencies."""
    return CognitiveAgent(encoder=mock_encoder, memory_store=mock_memory_store)


def test_cognitive_agent_initialization(cognitive_agent, mock_encoder, mock_memory_store):
    assert cognitive_agent.encoder == mock_encoder
    assert cognitive_agent.memory_store == mock_memory_store


def test_process_glucose_window_list_input(cognitive_agent, mock_encoder, mock_memory_store):
    glucose_list = [100.0, 105.0, 110.0]
    ts = datetime.now()
    extra_meta = {"user_id": "test_user"}

    result = cognitive_agent.process_glucose_window(
        glucose_list, timestamp=ts, extra_metadata=extra_meta
    )

    mock_encoder.encode.assert_called_once_with(glucose_list)

    expected_metadata = {
        "source": "agent_processed",
        "timestamp": ts.isoformat(),
        "user_id": "test_user",
    }
    # Check that add_embedding was called with the embedding from encoder and correct metadata
    # We can't directly check the embedding value if it's generated inside, but we know what encode returns
    mock_memory_store.add_embedding.assert_called_once_with(
        mock_encoder.encode.return_value, expected_metadata
    )

    assert np.array_equal(result["embedding"], mock_encoder.encode.return_value)
    assert result["metadata"] == expected_metadata


def test_process_glucose_window_numpy_input(cognitive_agent, mock_encoder, mock_memory_store):
    glucose_np = np.array([100.0, 105.0, 110.0])

    cognitive_agent.process_glucose_window(glucose_np)

    mock_encoder.encode.assert_called_once_with(glucose_np)
    # Ensure the first argument to add_embedding is the result of encoder.encode
    args, _ = mock_memory_store.add_embedding.call_args
    assert np.array_equal(args[0], mock_encoder.encode.return_value)


def test_process_glucose_window_invalid_input(cognitive_agent):
    with pytest.raises(ValueError, match="glucose_window must be a list or NumPy array."):
        cognitive_agent.process_glucose_window("not a list")


def test_find_similar_patterns(cognitive_agent, mock_encoder, mock_memory_store):
    query_window = [100.0, 105.0, 110.0]
    k_neighbors = 5

    results = cognitive_agent.find_similar_patterns(query_window, k=k_neighbors)

    mock_encoder.encode.assert_called_once_with(query_window)
    mock_memory_store.search_similar.assert_called_once_with(
        mock_encoder.encode.return_value, k=k_neighbors
    )

    assert len(results) == 1  # Based on mock_memory_store setup
    first_result = results[0]
    assert "pattern_embedding" in first_result
    assert "metadata" in first_result
    assert "similarity_score" in first_result
    # mock_memory_store.search_similar returns distance, agent converts to similarity
    expected_similarity = 1 - mock_memory_store.search_similar.return_value[0][2]
    assert abs(first_result["similarity_score"] - expected_similarity) < 1e-9


def test_find_similar_patterns_invalid_input(cognitive_agent):
    with pytest.raises(ValueError, match="glucose_window must be a list or NumPy array."):
        cognitive_agent.find_similar_patterns("not a list")


def test_get_memory_size(cognitive_agent, mock_memory_store):
    mock_memory_store.__len__.return_value = 5
    assert cognitive_agent.get_memory_size() == 5
    mock_memory_store.__len__.assert_called_once()


def test_train_agent_components_placeholder(cognitive_agent, mock_encoder, capsys):
    # Test that the placeholder training function can be called
    # and that it calls the encoder's training method if available.

    # Scenario 1: Encoder has train_encoder
    mock_encoder.train_encoder = MagicMock()
    mock_data_loader = MagicMock()  # A dummy data loader
    cognitive_agent.train_agent_components(training_data_loader=mock_data_loader, encoder_epochs=3)

    captured = capsys.readouterr()
    assert "Starting training for agent components..." in captured.out
    assert "Training GlucoseEncoder..." in captured.out
    mock_encoder.train_encoder.assert_called_once_with(mock_data_loader, epochs=3)
    assert "Agent components training placeholder complete." in captured.out

    # Scenario 2: Encoder does not have train_encoder (or no data_loader)
    mock_encoder.reset_mock()
    del mock_encoder.train_encoder  # Remove the method to simulate it not existing

    cognitive_agent.train_agent_components(training_data_loader=None)
    captured = capsys.readouterr()
    assert "GlucoseEncoder training skipped" in captured.out
