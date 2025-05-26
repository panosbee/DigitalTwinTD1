# tests/test_agent_memory_store.py
import pytest
import numpy as np

# Adjust import based on project structure (after src layout or current)
try:
    from agent.memory_store import VectorMemoryStore
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from agent.memory_store import VectorMemoryStore

EMBEDDING_DIM = 32


@pytest.fixture
def memory_store():
    """Provides a VectorMemoryStore instance."""
    return VectorMemoryStore(embedding_dim=EMBEDDING_DIM)


@pytest.fixture
def populated_store():
    """Provides a VectorMemoryStore instance populated with some data."""
    store = VectorMemoryStore(embedding_dim=EMBEDDING_DIM)
    for i in range(10):
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        meta = {"id": i, "timestamp": f"2024-01-01T{i:02d}:00:00Z"}
        store.add_embedding(emb, meta)
    store._build_index()  # Ensure index is built
    return store


def test_memory_store_initialization(memory_store):
    assert memory_store.embedding_dim == EMBEDDING_DIM
    assert len(memory_store) == 0
    assert memory_store.embeddings.shape == (0, EMBEDDING_DIM)
    assert memory_store.index is None


def test_add_embedding_single(memory_store):
    emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    meta = {"id": 1}
    memory_store.add_embedding(emb, meta)
    assert len(memory_store) == 1
    assert memory_store.embeddings.shape == (1, EMBEDDING_DIM)
    assert np.array_equal(memory_store.embeddings[0], emb)
    assert memory_store.metadata[0] == meta
    assert memory_store._index_needs_rebuild  # Index should need rebuild


def test_add_embedding_multiple(memory_store):
    for i in range(5):
        emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        memory_store.add_embedding(emb, {"id": i})
    assert len(memory_store) == 5
    assert memory_store.embeddings.shape == (5, EMBEDDING_DIM)


def test_add_embedding_dimension_mismatch(memory_store):
    emb_wrong_dim = np.random.rand(EMBEDDING_DIM + 1).astype(np.float32)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        memory_store.add_embedding(emb_wrong_dim)


def test_add_embedding_1d_reshaped_correctly(memory_store):
    # Test adding a (1, D) shaped embedding
    emb_1_d = np.random.rand(1, EMBEDDING_DIM).astype(np.float32)
    memory_store.add_embedding(emb_1_d, {"id": "1d_shape"})
    assert len(memory_store) == 1
    assert memory_store.embeddings.shape == (1, EMBEDDING_DIM)
    assert np.array_equal(memory_store.embeddings[0], emb_1_d.flatten())


def test_build_index(memory_store):
    emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    memory_store.add_embedding(emb)
    memory_store._build_index()
    assert memory_store.index is not None
    assert not memory_store._index_needs_rebuild


def test_build_index_empty_store(memory_store):
    memory_store._build_index()
    assert memory_store.index is None  # Index should be None for an empty store
    assert not memory_store._index_needs_rebuild  # Flag reset even if no index built


def test_search_similar_empty_store(memory_store):
    query_emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    results = memory_store.search_similar(query_emb, k=3)
    assert len(results) == 0


def test_search_similar_populated_store(populated_store):
    query_emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    results = populated_store.search_similar(query_emb, k=3)
    assert len(results) == 3
    for emb, meta, dist in results:
        assert emb.shape == (EMBEDDING_DIM,)
        assert isinstance(meta, dict)
        assert isinstance(dist, float)
        assert dist >= 0  # Cosine distance is >= 0


def test_search_similar_k_greater_than_store_size(populated_store):
    query_emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    # Store has 10 items
    results = populated_store.search_similar(query_emb, k=15)
    assert len(results) == 10  # Should return all items


def test_search_similar_exact_match(populated_store):
    # Take an existing embedding as query
    query_emb = populated_store.embeddings[0]
    results = populated_store.search_similar(query_emb, k=1)
    assert len(results) == 1
    # Cosine distance for identical vector should be close to 0
    assert results[0][2] < 1e-6
    assert results[0][1]["id"] == 0  # Assuming metadata id matches index


def test_search_similar_query_dim_mismatch(populated_store):
    query_emb_wrong_dim = np.random.rand(EMBEDDING_DIM + 1).astype(np.float32)
    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        populated_store.search_similar(query_emb_wrong_dim)


def test_get_all_embeddings(populated_store):
    embeddings, metadata_list = populated_store.get_all_embeddings()
    assert embeddings.shape == (10, EMBEDDING_DIM)
    assert len(metadata_list) == 10
    assert metadata_list[0]["id"] == 0


def test_clear_store(populated_store):
    assert len(populated_store) == 10
    populated_store.clear()
    assert len(populated_store) == 0
    assert populated_store.embeddings.shape == (0, EMBEDDING_DIM)
    assert len(populated_store.metadata) == 0
    assert populated_store.index is None


def test_len_method(memory_store):
    assert len(memory_store) == 0
    memory_store.add_embedding(np.random.rand(EMBEDDING_DIM).astype(np.float32))
    assert len(memory_store) == 1


def test_search_rebuilds_index_if_needed(memory_store):
    emb1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    memory_store.add_embedding(emb1, {"id": 1})
    assert memory_store._index_needs_rebuild

    query_emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    results = memory_store.search_similar(query_emb, k=1)  # This should trigger _build_index

    assert not memory_store._index_needs_rebuild
    assert memory_store.index is not None
    assert len(results) == 1
