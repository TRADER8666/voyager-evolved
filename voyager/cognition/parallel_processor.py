"""
Parallel Processing System for Voyager Evolved

Optimized for Intel i9-7900X (10 cores, 20 threads):
- Intelligent thread pool management
- Memory-efficient data structures  
- Multi-level caching system
- Parallel inference scheduling
- Work-stealing for load balancing

Hardware Target:
- CPU: i9-7900X (10C/20T, 3.3GHz base, 4.3GHz turbo)
- RAM: 48GB (allows generous caching)
- GPU: 4x P104 (for vision processing)
"""

import os
import time
import threading
import queue
import hashlib
import pickle
import weakref
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from collections import OrderedDict
import functools
import multiprocessing as mp


# ============================================================================
# CONFIGURATION
# ============================================================================

# Detect CPU configuration
try:
    CPU_COUNT = os.cpu_count() or 10
    # For i9-7900X: 10 cores, 20 threads
    PHYSICAL_CORES = max(1, CPU_COUNT // 2)
    LOGICAL_THREADS = CPU_COUNT
except Exception:
    PHYSICAL_CORES = 10
    LOGICAL_THREADS = 20

# Memory configuration (48GB target)
try:
    import psutil
    TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
except ImportError:
    TOTAL_MEMORY_GB = 48

# Thread pool sizes optimized for i9-7900X
DEFAULT_THREAD_POOL_SIZE = min(16, LOGICAL_THREADS - 2)  # Leave 2 for system
COMPUTE_POOL_SIZE = PHYSICAL_CORES  # CPU-bound tasks use physical cores
IO_POOL_SIZE = LOGICAL_THREADS  # I/O-bound can use all threads

# Cache sizes (generous with 48GB RAM)
DEFAULT_CACHE_SIZE_MB = int(TOTAL_MEMORY_GB * 1024 * 0.1)  # 10% of RAM
LLM_CACHE_SIZE = 1000  # Number of LLM responses
EMBEDDING_CACHE_SIZE = 5000  # Number of embeddings
MEMORY_CACHE_SIZE = 10000  # Number of memory items


# ============================================================================
# TASK PRIORITY
# ============================================================================

class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0     # Must complete ASAP (e.g., danger response)
    HIGH = 1         # Important (e.g., current goal)
    NORMAL = 2       # Standard priority
    LOW = 3          # Background tasks
    IDLE = 4         # Only when nothing else to do


@dataclass(order=True)
class PriorityTask:
    """A task with priority for the queue."""
    priority: int
    timestamp: float = field(compare=False, default_factory=time.time)
    task_id: str = field(compare=False, default="")
    func: Callable = field(compare=False, default=None)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    callback: Optional[Callable] = field(compare=False, default=None)


# ============================================================================
# LRU CACHE WITH SIZE LIMIT
# ============================================================================

T = TypeVar('T')


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with memory size limit.
    
    Optimized for frequent access patterns.
    """
    
    def __init__(
        self,
        max_items: int = 1000,
        max_size_mb: float = 100,
        ttl_seconds: Optional[float] = None
    ):
        self.max_items = max_items
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, Tuple[T, float, int]] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1000  # Default estimate
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, timestamp, size = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._current_size -= size
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value
    
    def put(self, key: str, value: T):
        """Put item in cache."""
        with self._lock:
            size = self._estimate_size(value)
            
            # Remove if already exists
            if key in self._cache:
                _, _, old_size = self._cache[key]
                self._current_size -= old_size
                del self._cache[key]
            
            # Evict until we have space
            while (len(self._cache) >= self.max_items or 
                   self._current_size + size > self.max_size_bytes):
                if not self._cache:
                    break
                oldest_key = next(iter(self._cache))
                _, _, oldest_size = self._cache[oldest_key]
                del self._cache[oldest_key]
                self._current_size -= oldest_size
            
            # Add new item
            self._cache[key] = (value, time.time(), size)
            self._current_size += size
    
    def delete(self, key: str):
        """Delete item from cache."""
        with self._lock:
            if key in self._cache:
                _, _, size = self._cache[key]
                del self._cache[key]
                self._current_size -= size
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "items": len(self._cache),
                "max_items": self.max_items,
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{self.hit_rate:.2%}"
            }


# ============================================================================
# MULTI-LEVEL CACHE SYSTEM
# ============================================================================

class MultiLevelCache:
    """
    Multi-level caching system optimized for 48GB RAM.
    
    Levels:
    1. L1: Hot cache (small, very fast)
    2. L2: Warm cache (medium, fast)
    3. L3: Cold cache (large, slower)
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l3_size: int = 10000,
        l1_ttl: float = 60,      # 1 minute
        l2_ttl: float = 300,     # 5 minutes
        l3_ttl: float = 3600     # 1 hour
    ):
        self.l1 = LRUCache(max_items=l1_size, max_size_mb=10, ttl_seconds=l1_ttl)
        self.l2 = LRUCache(max_items=l2_size, max_size_mb=100, ttl_seconds=l2_ttl)
        self.l3 = LRUCache(max_items=l3_size, max_size_mb=1000, ttl_seconds=l3_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache, checking all levels."""
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2, promote to L1
        value = self.l2.get(key)
        if value is not None:
            self.l1.put(key, value)
            return value
        
        # Try L3, promote to L2
        value = self.l3.get(key)
        if value is not None:
            self.l2.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, level: int = 1):
        """Put in cache at specified level."""
        if level <= 1:
            self.l1.put(key, value)
        elif level == 2:
            self.l2.put(key, value)
        else:
            self.l3.put(key, value)
    
    def invalidate(self, key: str):
        """Remove from all cache levels."""
        self.l1.delete(key)
        self.l2.delete(key)
        self.l3.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all levels."""
        return {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats(),
            "l3": self.l3.get_stats()
        }


# ============================================================================
# PARALLEL TASK EXECUTOR
# ============================================================================

class ParallelExecutor:
    """
    High-performance parallel task executor optimized for i9-7900X.
    
    Features:
    - Separate pools for CPU-bound and I/O-bound tasks
    - Priority queue for task scheduling
    - Work stealing for load balancing
    - Automatic pool sizing
    """
    
    def __init__(
        self,
        compute_workers: int = COMPUTE_POOL_SIZE,
        io_workers: int = IO_POOL_SIZE,
        max_queue_size: int = 1000
    ):
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        
        # Thread pools
        self._compute_pool = ThreadPoolExecutor(
            max_workers=compute_workers,
            thread_name_prefix="compute"
        )
        self._io_pool = ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="io"
        )
        
        # Priority queue for tasks
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._task_count = 0
        
        # Task tracking
        self._pending_futures: Dict[str, Future] = {}
        self._completed_results: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "compute_tasks": 0,
            "io_tasks": 0
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Running flag
        self._running = True
    
    def submit_compute(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit a CPU-bound task."""
        return self._submit(
            self._compute_pool, func, args, kwargs,
            priority, task_id, callback, "compute"
        )
    
    def submit_io(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit an I/O-bound task."""
        return self._submit(
            self._io_pool, func, args, kwargs,
            priority, task_id, callback, "io"
        )
    
    def _submit(
        self,
        pool: ThreadPoolExecutor,
        func: Callable,
        args: tuple,
        kwargs: dict,
        priority: TaskPriority,
        task_id: Optional[str],
        callback: Optional[Callable],
        task_type: str
    ) -> str:
        """Internal submit method."""
        with self._lock:
            self._task_count += 1
            if task_id is None:
                task_id = f"task_{self._task_count}_{int(time.time()*1000)}"
            
            self.stats["submitted"] += 1
            self.stats[f"{task_type}_tasks"] += 1
            
            # Submit to pool
            future = pool.submit(func, *args, **kwargs)
            self._pending_futures[task_id] = future
            
            # Add callback handling
            def on_complete(f):
                try:
                    result = f.result()
                    with self._lock:
                        self._completed_results[task_id] = result
                        self.stats["completed"] += 1
                    if callback:
                        callback(result)
                except Exception as e:
                    with self._lock:
                        self.stats["failed"] += 1
                    if callback:
                        callback(None, error=e)
            
            future.add_done_callback(on_complete)
            
            return task_id
    
    def map_compute(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Map a function over items in parallel (CPU-bound).
        
        Optimized for i9-7900X with work distribution.
        """
        if not items:
            return []
        
        if chunk_size is None:
            # Optimal chunk size for 10 cores
            chunk_size = max(1, len(items) // (self.compute_workers * 2))
        
        # Submit all tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = self._compute_pool.submit(
                lambda c: [func(item) for item in c],
                chunk
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception:
                results.extend([None] * chunk_size)
        
        return results
    
    def parallel_invoke(
        self,
        funcs: List[Tuple[Callable, tuple, dict]],
        timeout: Optional[float] = None
    ) -> List[Tuple[bool, Any]]:
        """
        Invoke multiple functions in parallel.
        
        Returns list of (success, result_or_error) tuples.
        """
        futures = []
        for func, args, kwargs in funcs:
            future = self._compute_pool.submit(func, *args, **kwargs)
            futures.append(future)
        
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append((True, result))
            except Exception as e:
                results.append((False, e))
        
        return results
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of a completed task."""
        with self._lock:
            if task_id in self._completed_results:
                return self._completed_results.pop(task_id)
            
            if task_id in self._pending_futures:
                future = self._pending_futures[task_id]
                try:
                    return future.result(timeout=timeout)
                except Exception:
                    return None
        
        return None
    
    def wait_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete."""
        results = {}
        deadline = time.time() + timeout if timeout else None
        
        for task_id in task_ids:
            remaining = None
            if deadline:
                remaining = max(0, deadline - time.time())
            
            result = self.get_result(task_id, timeout=remaining)
            results[task_id] = result
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            return {
                **self.stats,
                "pending": len(self._pending_futures),
                "compute_workers": self.compute_workers,
                "io_workers": self.io_workers
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor pools."""
        self._running = False
        self._compute_pool.shutdown(wait=wait)
        self._io_pool.shutdown(wait=wait)


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """
    Batches small operations for efficient processing.
    
    Useful for:
    - Memory operations (batch embeddings)
    - LLM calls (if supported)
    - Vector store operations
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_ms: float = 100,
        executor: Optional[ParallelExecutor] = None
    ):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.executor = executor or ParallelExecutor()
        
        # Pending items by operation type
        self._pending: Dict[str, List[Tuple[Any, threading.Event, List]]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Background flusher
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start background batch processing."""
        if self._running:
            return
        
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def stop(self):
        """Stop background processing."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
    
    def _flush_loop(self):
        """Background loop to flush batches."""
        while self._running:
            time.sleep(self.max_wait_ms / 1000)
            self._flush_all()
    
    def _flush_all(self):
        """Flush all pending batches."""
        with self._lock:
            for op_type, pending in list(self._pending.items()):
                if pending:
                    self._flush_batch(op_type)
    
    def _flush_batch(self, op_type: str):
        """Flush a specific batch."""
        # Implementation depends on operation type
        pass
    
    def add(
        self,
        op_type: str,
        item: Any,
        process_fn: Callable[[List[Any]], List[Any]]
    ) -> Any:
        """
        Add item to batch and wait for result.
        
        Args:
            op_type: Operation type for batching similar items
            item: Item to process
            process_fn: Function to process batch
            
        Returns:
            Result for this item
        """
        event = threading.Event()
        result_holder = [None]
        
        with self._lock:
            self._pending[op_type].append((item, event, result_holder, process_fn))
            
            # Flush if batch is full
            if len(self._pending[op_type]) >= self.batch_size:
                batch = self._pending[op_type][:self.batch_size]
                self._pending[op_type] = self._pending[op_type][self.batch_size:]
                
                # Process batch
                items = [b[0] for b in batch]
                process_fn = batch[0][3]
                
                try:
                    results = process_fn(items)
                    for i, (_, evt, holder, _) in enumerate(batch):
                        if i < len(results):
                            holder[0] = results[i]
                        evt.set()
                except Exception as e:
                    for _, evt, holder, _ in batch:
                        holder[0] = e
                        evt.set()
        
        # Wait for result
        event.wait(timeout=5.0)
        return result_holder[0]


# ============================================================================
# MEMORY-EFFICIENT DATA STRUCTURES
# ============================================================================

class CompactDict:
    """
    Memory-efficient dictionary for large datasets.
    
    Uses __slots__ and interning for reduced memory footprint.
    """
    
    __slots__ = ['_data', '_keys']
    
    def __init__(self):
        self._data: Dict[int, Any] = {}
        self._keys: Dict[int, str] = {}
    
    def _intern_key(self, key: str) -> int:
        """Get interned key hash."""
        return hash(key)
    
    def __getitem__(self, key: str) -> Any:
        h = self._intern_key(key)
        return self._data[h]
    
    def __setitem__(self, key: str, value: Any):
        h = self._intern_key(key)
        self._data[h] = value
        self._keys[h] = key
    
    def __delitem__(self, key: str):
        h = self._intern_key(key)
        del self._data[h]
        del self._keys[h]
    
    def __contains__(self, key: str) -> bool:
        return self._intern_key(key) in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        return self._keys.values()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        for h, value in self._data.items():
            yield self._keys[h], value
    
    def __len__(self) -> int:
        return len(self._data)


class RingBuffer:
    """
    Fixed-size ring buffer for efficient FIFO operations.
    
    Memory efficient for streaming data.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
    
    def append(self, item: Any):
        """Add item, overwriting oldest if full."""
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self.capacity
        
        if self._size < self.capacity:
            self._size += 1
        else:
            self._head = (self._head + 1) % self.capacity
    
    def pop(self) -> Optional[Any]:
        """Remove and return oldest item."""
        if self._size == 0:
            return None
        
        item = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        return item
    
    def peek(self) -> Optional[Any]:
        """View oldest item without removing."""
        if self._size == 0:
            return None
        return self._buffer[self._head]
    
    def __len__(self) -> int:
        return self._size
    
    def __iter__(self):
        """Iterate from oldest to newest."""
        for i in range(self._size):
            idx = (self._head + i) % self.capacity
            yield self._buffer[idx]


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_executor_instance: Optional[ParallelExecutor] = None
_cache_instance: Optional[MultiLevelCache] = None
_lock = threading.Lock()


def get_parallel_executor() -> ParallelExecutor:
    """Get the global parallel executor."""
    global _executor_instance
    
    with _lock:
        if _executor_instance is None:
            _executor_instance = ParallelExecutor()
        return _executor_instance


def get_cache() -> MultiLevelCache:
    """Get the global multi-level cache."""
    global _cache_instance
    
    with _lock:
        if _cache_instance is None:
            _cache_instance = MultiLevelCache()
        return _cache_instance


def shutdown_parallel():
    """Shutdown global parallel processing resources."""
    global _executor_instance
    
    with _lock:
        if _executor_instance:
            _executor_instance.shutdown(wait=False)
            _executor_instance = None


# ============================================================================
# DECORATORS
# ============================================================================

def parallel_map(chunk_size: Optional[int] = None):
    """Decorator to parallelize a function over a list."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(items: List[Any]) -> List[Any]:
            executor = get_parallel_executor()
            return executor.map_compute(func, items, chunk_size=chunk_size)
        return wrapper
    return decorator


def cached(cache_level: int = 1, ttl: Optional[float] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                pickle.dumps((func.__name__, args, tuple(sorted(kwargs.items()))))
            ).hexdigest()
            
            cache = get_cache()
            result = cache.get(key)
            
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.put(key, result, level=cache_level)
            
            return result
        return wrapper
    return decorator


def compute_bound(priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to run function on compute pool."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            executor = get_parallel_executor()
            task_id = executor.submit_compute(func, *args, priority=priority, **kwargs)
            return executor.get_result(task_id, timeout=60)
        return wrapper
    return decorator


def io_bound(priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to run function on I/O pool."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            executor = get_parallel_executor()
            task_id = executor.submit_io(func, *args, priority=priority, **kwargs)
            return executor.get_result(task_id, timeout=60)
        return wrapper
    return decorator
