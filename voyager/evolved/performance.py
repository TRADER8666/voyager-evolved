"""Performance Optimization Module for Voyager Evolved (Linux Optimized).

This module provides performance enhancements:
- LLM response caching
- Batch processing for observations
- Async processing utilities
- Memory management
- Linux-specific optimizations
"""

import os
import time
import json
import hashlib
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import voyager.utils as U

# Linux-specific imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class CacheEntry:
    """A cached LLM response entry."""
    key: str
    value: str
    timestamp: float
    hits: int = 0
    size_bytes: int = 0
    
    def is_expired(self, max_age: float) -> bool:
        return time.time() - self.timestamp > max_age


class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_entries: int = 1000, max_size_mb: float = 100):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, prompt: str, model: str = "", temperature: float = 0.0) -> str:
        """Create a cache key from inputs."""
        content = f"{model}:{temperature}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, prompt: str, model: str = "", temperature: float = 0.0) -> Optional[str]:
        """Get a cached response."""
        key = self._make_key(prompt, model, temperature)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.value
            self.misses += 1
            return None
    
    def put(self, prompt: str, response: str, model: str = "", temperature: float = 0.0):
        """Store a response in cache."""
        key = self._make_key(prompt, model, temperature)
        size = len(response.encode('utf-8'))
        
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_size -= old_entry.size_bytes
            
            # Evict entries if needed
            while (len(self.cache) >= self.max_entries or 
                   self.current_size + size > self.max_size_bytes) and self.cache:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.current_size -= oldest_entry.size_bytes
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=response,
                timestamp=time.time(),
                size_bytes=size
            )
            self.cache[key] = entry
            self.current_size += size
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                "entries": len(self.cache),
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }
    
    def save(self, path: str):
        """Save cache to disk."""
        with self.lock:
            data = {
                key: {
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                    "hits": entry.hits
                }
                for key, entry in self.cache.items()
            }
            U.dump_json(data, path)
    
    def load(self, path: str):
        """Load cache from disk."""
        if not U.f_exists(path):
            return
        
        try:
            data = U.load_json(path)
            with self.lock:
                for key, entry_data in data.items():
                    entry = CacheEntry(
                        key=key,
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        hits=entry_data.get("hits", 0),
                        size_bytes=len(entry_data["value"].encode('utf-8'))
                    )
                    self.cache[key] = entry
                    self.current_size += entry.size_bytes
        except Exception as e:
            print(f"\033[33mWarning: Could not load cache: {e}\033[0m")


# Global LLM cache instance
_llm_cache = LRUCache(max_entries=500, max_size_mb=50)


def cache_llm_response(func):
    """Decorator to cache LLM responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract prompt from args/kwargs
        prompt = kwargs.get('prompt') or (args[0] if args else None)
        model = kwargs.get('model_name', '')
        temperature = kwargs.get('temperature', 0.0)
        
        # Only cache if temperature is 0 (deterministic)
        if temperature == 0 and prompt:
            cached = _llm_cache.get(prompt, model, temperature)
            if cached:
                return cached
        
        result = func(*args, **kwargs)
        
        # Cache the result
        if temperature == 0 and prompt and result:
            _llm_cache.put(prompt, result, model, temperature)
        
        return result
    return wrapper


class BatchProcessor:
    """Batch processor for observations and other data."""
    
    def __init__(self, batch_size: int = 10, process_interval: float = 1.0):
        self.batch_size = batch_size
        self.process_interval = process_interval
        self.queue: List[Any] = []
        self.last_process_time = time.time()
        self.lock = threading.Lock()
        self.processor: Optional[Callable] = None
        self.results: List[Any] = []
    
    def set_processor(self, processor: Callable[[List[Any]], List[Any]]):
        """Set the batch processing function."""
        self.processor = processor
    
    def add(self, item: Any) -> bool:
        """Add an item to the batch queue. Returns True if batch was processed."""
        with self.lock:
            self.queue.append(item)
            
            current_time = time.time()
            should_process = (
                len(self.queue) >= self.batch_size or
                current_time - self.last_process_time >= self.process_interval
            )
            
            if should_process and self.processor:
                batch = self.queue.copy()
                self.queue.clear()
                self.last_process_time = current_time
                
                # Process batch
                results = self.processor(batch)
                self.results.extend(results)
                return True
            
            return False
    
    def flush(self) -> List[Any]:
        """Process remaining items and return all results."""
        with self.lock:
            if self.queue and self.processor:
                batch = self.queue.copy()
                self.queue.clear()
                results = self.processor(batch)
                self.results.extend(results)
            
            all_results = self.results.copy()
            self.results.clear()
            return all_results
    
    def get_results(self) -> List[Any]:
        """Get accumulated results without clearing."""
        with self.lock:
            return self.results.copy()


class AsyncTaskManager:
    """Manages async tasks for non-blocking operations."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit a task for async execution."""
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.pending_tasks[task_id] = future
        
        def callback(f):
            with self.lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
                try:
                    self.completed_tasks[task_id] = f.result()
                except Exception as e:
                    self.completed_tasks[task_id] = {"error": str(e)}
        
        future.add_done_callback(callback)
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of a completed task."""
        with self.lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks.pop(task_id)
            
            if task_id in self.pending_tasks:
                future = self.pending_tasks[task_id]
                try:
                    return future.result(timeout=timeout)
                except:
                    return None
        return None
    
    def is_pending(self, task_id: str) -> bool:
        """Check if a task is still pending."""
        with self.lock:
            return task_id in self.pending_tasks
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


class MemoryManager:
    """Linux-optimized memory management."""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.cleanup_callbacks: List[Callable] = []
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # Seconds
    
    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback."""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not HAS_PSUTIL:
            return {"available": True}
        
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        return {
            "system_percent": memory.percent,
            "system_available_mb": memory.available / (1024 * 1024),
            "process_rss_mb": process_memory.rss / (1024 * 1024),
            "process_vms_mb": process_memory.vms / (1024 * 1024)
        }
    
    def check_memory(self) -> bool:
        """Check if memory usage is acceptable. Returns False if cleanup needed."""
        if not HAS_PSUTIL:
            return True
        
        memory = psutil.virtual_memory()
        return memory.percent < self.max_memory_percent
    
    def trigger_cleanup(self, force: bool = False):
        """Trigger memory cleanup if needed."""
        current_time = time.time()
        
        if not force and current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        if force or not self.check_memory():
            print("\033[33mTriggering memory cleanup...\033[0m")
            self.last_cleanup = current_time
            
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"\033[31mCleanup error: {e}\033[0m")
            
            # Force garbage collection
            import gc
            gc.collect()
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for memory optimization."""
        suggestions = []
        
        usage = self.get_memory_usage()
        
        if usage.get("system_percent", 0) > 70:
            suggestions.append("Consider reducing batch sizes")
        
        if usage.get("process_rss_mb", 0) > 1000:
            suggestions.append("Process using >1GB RAM - consider clearing caches")
        
        cache_stats = _llm_cache.get_stats()
        if cache_stats["hit_rate"] < 0.3 and cache_stats["entries"] > 100:
            suggestions.append("Low cache hit rate - consider adjusting cache strategy")
        
        return suggestions


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.timers: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Start a timer."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and record the duration."""
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            
            # Keep only last 100 measurements
            if len(self.metrics[name]) > 100:
                self.metrics[name] = self.metrics[name][-100:]
        
        return duration
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
            
            if len(self.metrics[name]) > 100:
                self.metrics[name] = self.metrics[name][-100:]
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "last": values[-1]
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


# Global instances
_memory_manager = MemoryManager()
_perf_monitor = PerformanceMonitor()
_task_manager = AsyncTaskManager(max_workers=4)


def get_llm_cache() -> LRUCache:
    """Get the global LLM cache."""
    return _llm_cache


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager."""
    return _memory_manager


def get_perf_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _perf_monitor


def get_task_manager() -> AsyncTaskManager:
    """Get the global async task manager."""
    return _task_manager


def timed(metric_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _perf_monitor.start_timer(metric_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _perf_monitor.stop_timer(metric_name)
        return wrapper
    return decorator


def optimize_for_linux():
    """Apply Linux-specific optimizations."""
    import gc
    
    # Optimize garbage collection
    gc.set_threshold(700, 10, 10)
    
    # Set process nice value if possible
    if HAS_PSUTIL:
        try:
            p = psutil.Process(os.getpid())
            # Lower priority to be nicer to system
            p.nice(5)
        except:
            pass
    
    # Set memory-efficient environment variables
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')
    
    print("\033[32mLinux optimizations applied\033[0m")


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report."""
    return {
        "cache": _llm_cache.get_stats(),
        "memory": _memory_manager.get_memory_usage(),
        "metrics": _perf_monitor.get_all_stats(),
        "optimization_suggestions": _memory_manager.get_optimization_suggestions()
    }
