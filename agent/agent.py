# agent/agent.py
import numpy as np
from typing import List, Dict, Optional, Union # Removed Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass  # Added for PumpContext

from .encoder import GlucoseEncoder
from .memory_store import VectorMemoryStore


@dataclass
class PumpContext:
    """
    Represents the context of insulin pump actions and status at a given time.
    All fields are optional to allow for partial information.
    """
    timestamp: datetime
    bolus_type: Optional[str] = None            # e.g., "normal", "extended", "super-bolus"
    bolus_amount: Optional[float] = None         # Units
    programmed_bolus_amount: Optional[float] = None  # Units, if different from delivered (e.g. max bolus)
    active_basal_rate: Optional[float] = None    # Current effective basal rate (U/hr)

    # Temporary Basal Information
    temp_basal_active: Optional[bool] = False
    temp_basal_type: Optional[str] = None        # e.g., "percentage", "absolute"
    temp_basal_rate_value: Optional[float] = None  # The rate (U/hr) or percentage (e.g., 150 for 150%)
    temp_basal_duration_minutes: Optional[int] = None  # Remaining or total duration
    
    # Scheduled Basal Information (could be from a profile)
    scheduled_basal_rate: Optional[float] = None # U/hr

    # Consumables / Status
    insulin_on_board: Optional[float] = None   # Units
    carbs_on_board: Optional[float] = None     # Grams
    sensor_glucose_value: Optional[float] = None # mg/dL, if available from pump's integrated CGM
    pump_battery_level: Optional[int] = None   # Percentage
    pump_reservoir_units: Optional[float] = None # Units remaining
    
    # Pump-generated alerts or states
    pump_status: Optional[str] = None          # e.g., "normal", "suspended", "low_reservoir", "occlusion"
    control_iq_status: Optional[str] = None    # For Tandem pumps, e.g., "active", "sleep", "exercise"

    def __post_init__(self):
        # Basic validation or normalization can go here if needed
        if self.bolus_amount is not None and self.bolus_amount < 0:
            raise ValueError("Bolus amount cannot be negative.")
        if self.active_basal_rate is not None and self.active_basal_rate < 0:
            raise ValueError("Basal rate cannot be negative.")

class CognitiveAgent:
    def __init__(self, encoder: GlucoseEncoder, memory_store: VectorMemoryStore):
        """
        Initializes the Cognitive Agent.

        Args:
            encoder: An instance of GlucoseEncoder.
            memory_store: An instance of VectorMemoryStore.
        """
        self.encoder = encoder
        self.memory_store = memory_store

    def process_glucose_window(
        self, 
        glucose_window: Union[List[float], np.ndarray], 
        timestamp: Optional[datetime] = None,
        extra_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Processes a window of glucose data: encodes it and stores the embedding.

        Args:
            glucose_window: A list or NumPy array of glucose values.
            timestamp: Optional timestamp for when the window ends or is representative.
            extra_metadata: Optional dictionary with additional data to store alongside the embedding.

        Returns:
            A dictionary containing the generated embedding and any stored metadata.
        """
        if not isinstance(glucose_window, (list, np.ndarray)):
            raise ValueError("glucose_window must be a list or NumPy array.")

        embedding = self.encoder.encode(glucose_window)
        
        metadata = {"source": "agent_processed"}
        if timestamp:
            metadata["timestamp"] = timestamp.isoformat()
        if extra_metadata:
            metadata.update(extra_metadata)
            
        self.memory_store.add_embedding(embedding, metadata)
        
        return {"embedding": embedding, "metadata": metadata}

    def find_similar_patterns(
        self, 
        glucose_window: Union[List[float], np.ndarray], 
        k: int = 3
    ) -> List[Dict]:
        """
        Finds glucose patterns similar to the given window.

        Args:
            glucose_window: A list or NumPy array of glucose values for the query pattern.
            k: The number of similar patterns to retrieve.

        Returns:
            A list of dictionaries, each containing 'pattern_embedding', 
            'metadata', and 'similarity_score' (distance).
        """
        if not isinstance(glucose_window, (list, np.ndarray)):
            raise ValueError("glucose_window must be a list or NumPy array.")

        query_embedding = self.encoder.encode(glucose_window)
        similar_results = self.memory_store.search_similar(query_embedding, k=k)
        
        formatted_results = []
        for emb, meta, dist in similar_results:
            formatted_results.append({
                "pattern_embedding": emb,
                "metadata": meta,
                "similarity_score": 1 - dist # Convert distance to similarity (cosine: 1 is identical)
            })
        return formatted_results

    def train_agent_components(self, training_data_loader=None, encoder_epochs=5): # pragma: no cover
        """
        Placeholder for training agent components, primarily the encoder.
        The memory store itself is typically not "trained" in this context.
        """
        print("Starting training for agent components...")
        if hasattr(self.encoder, 'train_encoder') and training_data_loader:
            print("Training GlucoseEncoder...")
            self.encoder.train_encoder(training_data_loader, epochs=encoder_epochs)
        else:
            print("GlucoseEncoder training skipped (no 'train_encoder' method or no data_loader).")
        
        # VectorMemoryStore does not require explicit training in this setup
        print("Agent components training placeholder complete.")

    def get_memory_size(self) -> int:
        """Returns the number of items in the memory store."""
        return len(self.memory_store)

if __name__ == '__main__': # pragma: no cover
    # Example Usage
    # 1. Initialize components
    glucose_enc = GlucoseEncoder(input_dim=1, embedding_dim=32, hidden_dim=64)
    memory = VectorMemoryStore(embedding_dim=32)
    agent = CognitiveAgent(encoder=glucose_enc, memory_store=memory)

    # 2. Process some glucose windows
    window1 = [100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    window2 = [150.0, 160.0, 170.0, 180.0, 175.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0]
    window3 = [90.0, 95.0, 100.0, 105.0, 110.0, 112.0, 108.0, 105.0, 100.0, 98.0, 92.0, 88.0] # Similar to window1

    ts1 = datetime.now() - timedelta(minutes=60)
    ts2 = datetime.now() - timedelta(minutes=30)
    ts3 = datetime.now()

    result1 = agent.process_glucose_window(window1, timestamp=ts1, extra_metadata={"event": "pre_meal"})
    result2 = agent.process_glucose_window(window2, timestamp=ts2, extra_metadata={"event": "post_meal_high"})
    result3 = agent.process_glucose_window(window3, timestamp=ts3, extra_metadata={"event": "stable_recovery"})

    print(f"Memory size after processing: {agent.get_memory_size()}")

    # 3. Find patterns similar to a new window
    query_window = [95.0, 100.0, 108.0, 112.0, 115.0, 110.0, 105.0, 100.0, 96.0, 92.0, 88.0, 85.0] # New window, similar to window1/3
    print(f"\nSearching for patterns similar to: {query_window}")
    similar = agent.find_similar_patterns(query_window, k=2)

    for item in similar:
        print(f"  Found similar pattern (Timestamp: {item['metadata'].get('timestamp')}, Event: {item['metadata'].get('event')})")
        print(f"  Similarity score: {item['similarity_score']:.4f}")
        # print(f"  Embedding: {item['pattern_embedding'][:5]}...") # Show snippet

    # Placeholder for training
    # agent.train_agent_components() # Needs a data_loader