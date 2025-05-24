"""
⚡ Performance Optimization Module
=================================

Blazing fast predictions με caching, parallel processing, και optimized models.
"""

import numpy as np
from functools import lru_cache, wraps
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Any, Callable
import asyncio
import aioredis
import pickle
import hashlib
import numba
from numba import jit, prange
import torch
import torch.jit


class PerformanceOptimizer:
    """Optimize SDK performance για lightning-fast predictions."""
    
    def __init__(self, cache_size=1000, use_gpu=False):
        self.cache_size = cache_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Process pool for CPU-intensive operations
        n_cores = mp.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=n_cores)
        
        # Initialize cache
        self._init_cache()
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_prediction_time': 0,
            'total_predictions': 0
        }
        
    def _init_cache(self):
        """Initialize caching system."""
        self.memory_cache = {}
        self.redis_client = None
        
        # Try to connect to Redis for distributed caching
        try:
            import redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False
            )
            self.redis_client.ping()
            print("✅ Redis cache connected")
        except:
            print("⚠️ Redis not available, using memory cache only")
    
    def cache_key(self, *args, **kwargs):
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached_prediction(self, func: Callable) -> Callable:
        """Decorator for caching predictions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = self.cache_key(*args, **kwargs)
            
            # Check memory cache first
            if key in self.memory_cache:
                self.metrics['cache_hits'] += 1
                return self.memory_cache[key]
            
            # Check Redis cache
            if self.redis_client:
                try:
                    cached = self.redis_client.get(f"pred:{key}")
                    if cached:
                        self.metrics['cache_hits'] += 1
                        result = pickle.loads(cached)
                        self.memory_cache[key] = result
                        return result
                except:
                    pass
            
            # Cache miss - compute result
            self.metrics['cache_misses'] += 1
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics['total_predictions'] += 1
            self.metrics['avg_prediction_time'] = (
                (self.metrics['avg_prediction_time'] * (self.metrics['total_predictions'] - 1) + elapsed) /
                self.metrics['total_predictions']
            )
            
            # Store in cache
            self.memory_cache[key] = result
            if len(self.memory_cache) > self.cache_size:
                # Remove oldest entry
                oldest = next(iter(self.memory_cache))
                del self.memory_cache[oldest]
            
            # Store in Redis with 5-minute expiry
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"pred:{key}",
                        300,
                        pickle.dumps(result)
                    )
                except:
                    pass
            
            return result
        
        return wrapper
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def fast_glucose_prediction(history: np.ndarray, horizon: int) -> float:
        """
        Ultra-fast glucose prediction using Numba JIT compilation.
        
        Args:
            history: Glucose history array
            horizon: Prediction horizon in steps
            
        Returns:
            Predicted glucose value
        """
        n = len(history)
        if n < 3:
            return history[-1] if n > 0 else 120.0
        
        # Fast linear regression with Numba
        x = np.arange(n, dtype=np.float64)
        y = history.astype(np.float64)
        
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope and intercept
        numerator = 0.0
        denominator = 0.0
        
        for i in prange(n):
            numerator += (x[i] - x_mean) * (y[i] - y_mean)
            denominator += (x[i] - x_mean) ** 2
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict future value
        future_x = n + horizon - 1
        prediction = slope * future_x + intercept
        
        # Constrain to realistic range
        return max(40.0, min(400.0, prediction))
    
    async def parallel_predict_batch(self, patient_data: List[Dict]) -> List[float]:
        """
        Parallel batch predictions για multiple patients.
        
        Args:
            patient_data: List of patient data dictionaries
            
        Returns:
            List of predictions
        """
        tasks = []
        
        for data in patient_data:
            task = asyncio.create_task(
                self._async_predict_single(data)
            )
            tasks.append(task)
        
        predictions = await asyncio.gather(*tasks)
        return predictions
    
    async def _async_predict_single(self, data: Dict) -> float:
        """Async prediction for single patient."""
        # Convert to numpy array
        history = np.array(data.get('glucose_history', []))
        horizon = data.get('horizon', 6)  # 30 minutes default
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None,
            self.fast_glucose_prediction,
            history,
            horizon
        )
        
        return float(prediction)
    
    def optimize_model_inference(self, model):
        """
        Optimize PyTorch model για faster inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        if not isinstance(model, torch.nn.Module):
            return model
        
        # Put in evaluation mode
        model.eval()
        
        # Move to GPU if available
        if self.use_gpu:
            model = model.cuda()
            print("🚀 Model moved to GPU")
        
        # JIT compilation for faster inference
        try:
            example_input = torch.randn(1, 48)  # Example: 4 hours of data
            if self.use_gpu:
                example_input = example_input.cuda()
            
            model_jit = torch.jit.trace(model, example_input)
            print("⚡ Model JIT compiled")
            return model_jit
        except:
            print("⚠️ JIT compilation failed, using original model")
            return model
    
    def batch_process_cgm_data(self, cgm_data: np.ndarray, window_size: int = 48) -> np.ndarray:
        """
        Vectorized processing of CGM data για training.
        
        Args:
            cgm_data: Raw CGM data
            window_size: Size of sliding window
            
        Returns:
            Processed windows
        """
        n_samples = len(cgm_data) - window_size + 1
        
        # Create sliding windows using stride tricks
        stride = cgm_data.strides[0]
        windows = np.lib.stride_tricks.as_strided(
            cgm_data,
            shape=(n_samples, window_size),
            strides=(stride, stride)
        )
        
        return windows
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        cache_hit_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = (
                self.metrics['cache_hits'] / 
                (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100
            )
        
        return {
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_prediction_time_ms': f"{self.metrics['avg_prediction_time'] * 1000:.1f}",
            'total_predictions': self.metrics['total_predictions'],
            'gpu_available': self.use_gpu,
            'cpu_cores': mp.cpu_count(),
            'cache_size': self.cache_size
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        if self.redis_client:
            self.redis_client.close()


class FastDataLoader:
    """Optimized data loader για training."""
    
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def normalize_batch(data: np.ndarray) -> np.ndarray:
        """Fast batch normalization με Numba."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return data - mean
        
        normalized = np.empty_like(data)
        for i in prange(data.shape[0]):
            for j in prange(data.shape[1]):
                normalized[i, j] = (data[i, j] - mean) / std
        
        return normalized
    
    def load_batch_parallel(self, indices: List[int]) -> np.ndarray:
        """Load batch in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._load_single_sample, idx)
                for idx in indices
            ]
            
            batch = np.array([
                future.result() for future in futures
            ])
            
        return self.normalize_batch(batch)
    
    def _load_single_sample(self, idx: int) -> np.ndarray:
        """Load single sample - to be implemented based on data format."""
        # Placeholder - implement based on actual data format
        return np.random.randn(48)  # 4 hours of 5-minute samples


# Benchmark utilities
def benchmark_prediction(optimizer: PerformanceOptimizer, n_iterations: int = 1000):
    """Benchmark prediction performance."""
    print(f"\n🏁 Running performance benchmark ({n_iterations} iterations)...")
    
    # Generate test data
    test_history = np.random.normal(120, 20, 48)
    
    # Warm up
    for _ in range(10):
        optimizer.fast_glucose_prediction(test_history, 6)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(n_iterations):
        result = optimizer.fast_glucose_prediction(test_history, 6)
    
    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    print(f"✅ Average prediction time: {avg_time_ms:.3f} ms")
    print(f"✅ Predictions per second: {n_iterations / elapsed:.0f}")
    
    return avg_time_ms


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = PerformanceOptimizer(use_gpu=True)
    
    # Benchmark
    benchmark_prediction(optimizer)
    
    # Test batch predictions
    test_patients = [
        {'glucose_history': np.random.normal(120, 20, 48).tolist(), 'horizon': 6}
        for _ in range(100)
    ]
    
    # Run async batch predictions
    async def test_batch():
        start = time.time()
        predictions = await optimizer.parallel_predict_batch(test_patients)
        elapsed = time.time() - start
        print(f"\n✅ Batch prediction for {len(test_patients)} patients: {elapsed:.2f}s")
        print(f"✅ Average time per patient: {elapsed/len(test_patients)*1000:.1f}ms")
    
    asyncio.run(test_batch())
    
    # Performance report
    print("\n📊 Performance Report:")
    report = optimizer.get_performance_report()
    for key, value in report.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    optimizer.cleanup() 