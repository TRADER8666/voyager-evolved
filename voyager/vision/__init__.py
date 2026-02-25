"""Vision module for Voyager Evolved.

GPU-accelerated computer vision system for Minecraft AI.
Optimized for multi-GPU setups (4x P104 GPUs).
"""

from voyager.vision.gpu_manager import GPUManager, get_gpu_manager
from voyager.vision.capture import ScreenCapture, FrameBuffer
from voyager.vision.detector import (
    MinecraftDetector,
    DetectedObject,
    ObjectCategory,
)
from voyager.vision.attention import (
    VisualAttention,
    AttentionFocus,
    SaliencyMap,
)
from voyager.vision.embeddings import (
    ImageEmbedder,
    EmbeddingCache,
)
from voyager.vision.tracker import (
    ObjectTracker,
    TrackedObject,
    TrackingState,
)
from voyager.vision.config import VisionConfig
from voyager.vision.integration import (
    VisionSystem,
    VisualObservation,
    VisualPlayerObserver,
    create_vision_system,
)

__all__ = [
    # GPU Management
    'GPUManager',
    'get_gpu_manager',
    # Capture
    'ScreenCapture',
    'FrameBuffer',
    # Detection
    'MinecraftDetector',
    'DetectedObject',
    'ObjectCategory',
    # Attention
    'VisualAttention',
    'AttentionFocus',
    'SaliencyMap',
    # Embeddings
    'ImageEmbedder',
    'EmbeddingCache',
    # Tracking
    'ObjectTracker',
    'TrackedObject',
    'TrackingState',
    # Config
    'VisionConfig',
    # Integration
    'VisionSystem',
    'VisualObservation',
    'VisualPlayerObserver',
    'create_vision_system',
]

__version__ = '1.0.0'
