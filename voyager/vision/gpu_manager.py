"""GPU Manager for multi-GPU vision processing.

Optimized for 4x P104 GPU setup with 48GB total RAM.
"""

import os
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import queue

logger = logging.getLogger(__name__)

# Global GPU manager instance
_gpu_manager: Optional['GPUManager'] = None
_gpu_lock = threading.Lock()


class GPUStatus(Enum):
    """GPU device status."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    device_id: int
    name: str
    total_memory: int  # bytes
    free_memory: int
    used_memory: int
    utilization: float  # 0-1
    temperature: int  # Celsius
    status: GPUStatus


class GPUManager:
    """Manages multiple GPUs for distributed vision processing.
    
    Features:
    - Automatic GPU detection and health monitoring
    - Load balancing across GPUs
    - Memory management with automatic cleanup
    - CUDA graph optimization
    - Mixed precision (FP16) support
    """
    
    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.8,
        enable_cuda_graphs: bool = True,
        mixed_precision: bool = True,
    ):
        self.device_ids = device_ids or [0, 1, 2, 3]
        self.memory_fraction = memory_fraction
        self.enable_cuda_graphs = enable_cuda_graphs
        self.mixed_precision = mixed_precision
        
        self._torch_available = False
        self._cuda_available = False
        self._device_info: Dict[int, GPUInfo] = {}
        self._device_locks: Dict[int, threading.Lock] = {}
        self._work_queues: Dict[int, queue.Queue] = {}
        self._initialized = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize GPU manager and detect available GPUs."""
        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            
            if self._cuda_available:
                logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s) detected")
                self._setup_devices()
            else:
                logger.warning("CUDA not available, falling back to CPU")
                
        except ImportError:
            logger.warning("PyTorch not installed, GPU acceleration disabled")
            self._torch_available = False
            self._cuda_available = False
        
        self._initialized = True
    
    def _setup_devices(self) -> None:
        """Set up available GPU devices."""
        import torch
        
        available_gpus = torch.cuda.device_count()
        valid_device_ids = [d for d in self.device_ids if d < available_gpus]
        
        if not valid_device_ids:
            logger.warning(f"No valid GPU devices from {self.device_ids}, using device 0")
            valid_device_ids = [0] if available_gpus > 0 else []
        
        self.device_ids = valid_device_ids
        
        for device_id in self.device_ids:
            # Initialize device info
            self._device_info[device_id] = self._get_device_info(device_id)
            self._device_locks[device_id] = threading.Lock()
            self._work_queues[device_id] = queue.Queue()
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                try:
                    torch.cuda.set_per_process_memory_fraction(
                        self.memory_fraction, device_id
                    )
                except Exception as e:
                    logger.warning(f"Could not set memory fraction for GPU {device_id}: {e}")
        
        # Enable TF32 for tensor cores on Ampere+ GPUs
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"Initialized {len(self.device_ids)} GPU(s): {self.device_ids}")
    
    def _get_device_info(self, device_id: int) -> GPUInfo:
        """Get information about a specific GPU."""
        import torch
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory
            
            # Get memory info
            torch.cuda.set_device(device_id)
            free_memory, total = torch.cuda.mem_get_info(device_id)
            used_memory = total - free_memory
            
            # Get utilization (approximate)
            utilization = used_memory / total_memory if total_memory > 0 else 0
            
            # Temperature not directly available in PyTorch, default to 0
            temperature = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            return GPUInfo(
                device_id=device_id,
                name=props.name,
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                utilization=utilization,
                temperature=temperature,
                status=GPUStatus.AVAILABLE,
            )
        except Exception as e:
            logger.error(f"Error getting GPU {device_id} info: {e}")
            return GPUInfo(
                device_id=device_id,
                name="Unknown",
                total_memory=0,
                free_memory=0,
                used_memory=0,
                utilization=0,
                temperature=0,
                status=GPUStatus.ERROR,
            )
    
    def get_best_device(self) -> int:
        """Get the GPU with most free memory."""
        if not self._cuda_available or not self.device_ids:
            return -1  # CPU
        
        best_device = self.device_ids[0]
        max_free = 0
        
        for device_id in self.device_ids:
            info = self._get_device_info(device_id)
            if info.status == GPUStatus.AVAILABLE and info.free_memory > max_free:
                max_free = info.free_memory
                best_device = device_id
        
        return best_device
    
    def get_device(self, strategy: str = "best") -> 'torch.device':
        """Get a torch device based on strategy.
        
        Args:
            strategy: "best" (most free memory), "round_robin", or "random"
        
        Returns:
            torch.device for the selected GPU or CPU
        """
        import torch
        
        if not self._cuda_available:
            return torch.device('cpu')
        
        if strategy == "best":
            device_id = self.get_best_device()
        elif strategy == "round_robin":
            device_id = self._round_robin_device()
        else:
            import random
            device_id = random.choice(self.device_ids) if self.device_ids else -1
        
        if device_id >= 0:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cpu')
    
    _rr_counter = 0
    _rr_lock = threading.Lock()
    
    def _round_robin_device(self) -> int:
        """Get next device in round-robin fashion."""
        if not self.device_ids:
            return -1
        
        with self._rr_lock:
            device = self.device_ids[GPUManager._rr_counter % len(self.device_ids)]
            GPUManager._rr_counter += 1
        return device
    
    def distribute_batch(
        self,
        batch_size: int,
        min_per_gpu: int = 1
    ) -> Dict[int, Tuple[int, int]]:
        """Distribute a batch across available GPUs.
        
        Args:
            batch_size: Total batch size
            min_per_gpu: Minimum items per GPU
        
        Returns:
            Dict mapping device_id to (start_idx, end_idx)
        """
        if not self._cuda_available or not self.device_ids:
            return {-1: (0, batch_size)}  # All on CPU
        
        num_gpus = len(self.device_ids)
        
        # Calculate items per GPU based on free memory
        total_free = sum(
            self._get_device_info(d).free_memory
            for d in self.device_ids
        )
        
        distribution = {}
        current_idx = 0
        
        for i, device_id in enumerate(self.device_ids):
            info = self._get_device_info(device_id)
            
            if total_free > 0:
                # Proportional distribution based on free memory
                items = int(batch_size * (info.free_memory / total_free))
            else:
                # Equal distribution
                items = batch_size // num_gpus
            
            # Ensure minimum items
            items = max(items, min_per_gpu)
            
            # Handle last GPU - gets remaining items
            if i == num_gpus - 1:
                items = batch_size - current_idx
            
            if items > 0 and current_idx < batch_size:
                end_idx = min(current_idx + items, batch_size)
                distribution[device_id] = (current_idx, end_idx)
                current_idx = end_idx
        
        return distribution
    
    def synchronize_all(self) -> None:
        """Synchronize all GPU devices."""
        if not self._cuda_available:
            return
        
        import torch
        for device_id in self.device_ids:
            torch.cuda.synchronize(device_id)
    
    def clear_cache(self, device_id: Optional[int] = None) -> None:
        """Clear GPU memory cache."""
        if not self._cuda_available:
            return
        
        import torch
        
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        else:
            for d in self.device_ids:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[int, Dict[str, int]]:
        """Get memory statistics for all GPUs."""
        stats = {}
        
        if not self._cuda_available:
            return stats
        
        import torch
        
        for device_id in self.device_ids:
            info = self._get_device_info(device_id)
            stats[device_id] = {
                'total': info.total_memory,
                'free': info.free_memory,
                'used': info.used_memory,
                'utilization': info.utilization,
            }
        
        return stats
    
    def create_stream(self, device_id: int) -> Optional['torch.cuda.Stream']:
        """Create a CUDA stream for async operations."""
        if not self._cuda_available or device_id not in self.device_ids:
            return None
        
        import torch
        return torch.cuda.Stream(device_id)
    
    def get_autocast_context(self, device_id: int = 0):
        """Get autocast context for mixed precision."""
        import torch
        
        if self.mixed_precision and self._cuda_available:
            return torch.cuda.amp.autocast(enabled=True)
        else:
            # Return a no-op context manager
            import contextlib
            return contextlib.nullcontext()
    
    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._cuda_available and len(self.device_ids) > 0
    
    @property
    def device_count(self) -> int:
        """Get number of available GPUs."""
        return len(self.device_ids) if self._cuda_available else 0
    
    def __repr__(self) -> str:
        return (
            f"GPUManager(devices={self.device_ids}, "
            f"cuda_available={self._cuda_available})"
        )


def get_gpu_manager(
    device_ids: Optional[List[int]] = None,
    memory_fraction: float = 0.8,
    **kwargs
) -> GPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    
    with _gpu_lock:
        if _gpu_manager is None:
            _gpu_manager = GPUManager(
                device_ids=device_ids,
                memory_fraction=memory_fraction,
                **kwargs
            )
        return _gpu_manager


def reset_gpu_manager() -> None:
    """Reset the global GPU manager."""
    global _gpu_manager
    
    with _gpu_lock:
        if _gpu_manager is not None:
            _gpu_manager.clear_cache()
            _gpu_manager = None
