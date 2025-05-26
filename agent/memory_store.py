# agent/memory_store.py
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.neighbors import NearestNeighbors


class VectorMemoryStore:
    def __init__(self, embedding_dim: int, initial_capacity: int = 1000):
        self.embedding_dim = embedding_dim
        self.embeddings = np.empty((0, embedding_dim), dtype=np.float32)
        self.metadata: List[Dict] = []  # Stores associated metadata for each embedding
        self.index: Optional[NearestNeighbors] = None
        self._index_needs_rebuild = False
        self.initial_capacity = initial_capacity

    def add_embedding(self, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Adds an embedding and its metadata to the store.

        Args:
            embedding: A NumPy array representing the embedding.
            metadata: An optional dictionary containing metadata (e.g., timestamp, pattern_type).
        """
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] == 1 and embedding.shape[1] == self.embedding_dim:  # (1, D)
                embedding = embedding.reshape(self.embedding_dim)
            else:  # (D, 1) or other incorrect shapes
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape}"
                )

        if self.embeddings.size == 0:
            self.embeddings = np.array([embedding], dtype=np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self.metadata.append(metadata or {})
        self._index_needs_rebuild = True

    def search_similar(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[np.ndarray, Dict, float]]:
        """
        Searches for the k most similar embeddings to the query embedding.

        Args:
            query_embedding: The query embedding as a NumPy array.
            k: The number of similar embeddings to return.

        Returns:
            A list of tuples, where each tuple contains (embedding, metadata, distance).
            Returns an empty list if the store is empty.
        """
        if self.embeddings.shape[0] == 0:
            return []

        if self._index_needs_rebuild or self.index is None:
            self._build_index()

        # Ensure index is built and store is not empty before proceeding
        if self.index is None:
            # This can happen if the store was empty initially or became empty.
            # _build_index() sets self.index to None for an empty store.
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Reshape to (1, D)

        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {query_embedding.shape[1]}"
            )

        distances, indices = self.index.kneighbors(
            query_embedding, n_neighbors=min(k, self.embeddings.shape[0])
        )

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            results.append((self.embeddings[idx], self.metadata[idx], float(dist)))

        return results

    def _build_index(self):
        """Builds or rebuilds the k-NN index."""
        if self.embeddings.shape[0] > 0:
            self.index = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="cosine")
            self.index.fit(self.embeddings)
            self._index_needs_rebuild = False
        else:
            self.index = None  # Cannot build index for empty store

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """Returns all stored embeddings and their metadata."""
        return self.embeddings, self.metadata

    def clear(self):
        """Clears all embeddings and metadata from the store."""
        self.embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.metadata = []
        self.index = None
        self._index_needs_rebuild = False

    def __len__(self) -> int:
        return self.embeddings.shape[0]


if __name__ == "__main__":  # pragma: no cover
    # Example Usage
    store = VectorMemoryStore(embedding_dim=32)

    # Add some embeddings
    for i in range(10):
        emb = np.random.rand(32).astype(np.float32)
        meta = {"id": i, "timestamp": f"2024-01-01T{i:02d}:00:00Z", "type": "random"}
        store.add_embedding(emb, meta)

    print(f"Store size: {len(store)}")

    # Search for similar embeddings
    query_emb = np.random.rand(32).astype(np.float32)
    similar_items = store.search_similar(query_emb, k=3)

    print("\nMost similar items to query:")
    for emb, meta, dist in similar_items:
        print(f"  ID: {meta.get('id')}, Distance: {dist:.4f}, Embedding snippet: {emb[:3]}...")

    all_embs, all_meta = store.get_all_embeddings()
    print(f"\nTotal embeddings stored: {all_embs.shape[0]}")

    store.clear()
    print(f"Store size after clear: {len(store)}")
