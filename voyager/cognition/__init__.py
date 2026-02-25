"""
Cognition Module for Voyager Evolved

Higher-order cognitive systems that enable human-like thinking:
- Meta-Cognition: Self-awareness and self-regulation
- Strategy Selection: Choosing appropriate problem-solving approaches
- Confidence Calibration: Accurate self-assessment
- Learning from Mistakes: Error-driven improvement
- Parallel Processing: Optimized for i9-7900X (20 threads)
"""

from .metacognition import (
    MetaCognition,
    CognitiveStrategy,
    CognitiveState,
    ConfidenceLevel,
    KnowledgeGap,
    Mistake,
    ConfidenceCalibrator,
    UncertaintyQuantifier,
    StrategySelector,
    get_metacognition,
    reset_metacognition
)

from .parallel_processor import (
    ParallelExecutor,
    MultiLevelCache,
    LRUCache,
    BatchProcessor,
    CompactDict,
    RingBuffer,
    TaskPriority,
    get_parallel_executor,
    get_cache,
    shutdown_parallel,
    parallel_map,
    cached,
    compute_bound,
    io_bound,
    PHYSICAL_CORES,
    LOGICAL_THREADS,
    TOTAL_MEMORY_GB
)

__all__ = [
    # Meta-Cognition
    "MetaCognition",
    "CognitiveStrategy",
    "CognitiveState",
    "ConfidenceLevel",
    "KnowledgeGap",
    "Mistake",
    "ConfidenceCalibrator",
    "UncertaintyQuantifier",
    "StrategySelector",
    "get_metacognition",
    "reset_metacognition",
    
    # Parallel Processing
    "ParallelExecutor",
    "MultiLevelCache",
    "LRUCache",
    "BatchProcessor",
    "CompactDict",
    "RingBuffer",
    "TaskPriority",
    "get_parallel_executor",
    "get_cache",
    "shutdown_parallel",
    "parallel_map",
    "cached",
    "compute_bound",
    "io_bound",
    
    # System info
    "PHYSICAL_CORES",
    "LOGICAL_THREADS",
    "TOTAL_MEMORY_GB"
]

__version__ = "1.0.0"
