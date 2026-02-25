"""Vision system configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import os


class GPUMode(Enum):
    """GPU processing mode."""
    SINGLE = "single"        # Use single GPU
    MULTI = "multi"          # Distribute across GPUs
    AUTO = "auto"            # Auto-detect best mode
    CPU_ONLY = "cpu_only"    # Fallback to CPU


@dataclass
class GPUConfig:
    """GPU-specific configuration."""
    mode: GPUMode = GPUMode.AUTO
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    memory_fraction: float = 0.8  # Use 80% of available VRAM
    enable_cuda_graphs: bool = True
    enable_tensor_cores: bool = True
    batch_size: int = 8
    mixed_precision: bool = True  # FP16 for faster inference


@dataclass
class CaptureConfig:
    """Screenshot capture configuration."""
    fps: int = 30
    resolution: tuple = (1920, 1080)
    buffer_size: int = 60  # Frames to keep in buffer
    capture_method: str = "mss"  # mss, pyautogui, x11
    compress_buffer: bool = True
    jpeg_quality: int = 85


@dataclass
class DetectorConfig:
    """Object detection configuration."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    enable_entity_detection: bool = True
    enable_block_detection: bool = True
    enable_player_detection: bool = True
    enable_item_detection: bool = True
    model_size: str = "medium"  # small, medium, large


@dataclass
class AttentionConfig:
    """Visual attention configuration."""
    saliency_method: str = "spectral"  # spectral, itti-koch, deep
    attention_decay: float = 0.95
    focus_threshold: float = 0.3
    max_focus_points: int = 10
    enable_motion_attention: bool = True
    enable_novelty_attention: bool = True
    peripheral_weight: float = 0.3


@dataclass
class EmbeddingConfig:
    """Image embedding configuration."""
    model_name: str = "resnet50"  # resnet50, efficientnet, vit
    embedding_dim: int = 512
    cache_size: int = 10000
    normalize: bool = True
    enable_scene_embedding: bool = True
    enable_object_embedding: bool = True


@dataclass
class TrackerConfig:
    """Object tracking configuration."""
    tracker_type: str = "deep_sort"  # deep_sort, byte_track, iou
    max_age: int = 30  # Frames to keep lost track
    min_hits: int = 3  # Min detections to confirm track
    iou_threshold: float = 0.3
    feature_distance_threshold: float = 0.5
    enable_reid: bool = True  # Re-identification


@dataclass
class VisionConfig:
    """Main vision system configuration."""
    enabled: bool = True
    gpu: GPUConfig = field(default_factory=GPUConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    
    # Performance settings
    async_processing: bool = True
    processing_threads: int = 4
    max_memory_mb: int = 4096  # Max memory for vision system
    
    # Persistence
    cache_dir: str = "vision_cache"
    save_debug_frames: bool = False
    debug_frame_interval: int = 100
    
    def validate(self) -> List[str]:
        """Validate configuration settings."""
        errors = []
        
        if self.gpu.memory_fraction <= 0 or self.gpu.memory_fraction > 1:
            errors.append("GPU memory_fraction must be between 0 and 1")
        
        if self.capture.fps <= 0 or self.capture.fps > 120:
            errors.append("Capture FPS must be between 1 and 120")
        
        if self.detector.confidence_threshold < 0 or self.detector.confidence_threshold > 1:
            errors.append("Detection confidence_threshold must be between 0 and 1")
        
        if self.embedding.embedding_dim not in [128, 256, 512, 1024, 2048]:
            errors.append("Embedding dimension should be 128, 256, 512, 1024, or 2048")
        
        return errors
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = self._to_dict()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'VisionConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls._from_dict(config_dict)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'gpu': {
                'mode': self.gpu.mode.value,
                'device_ids': self.gpu.device_ids,
                'memory_fraction': self.gpu.memory_fraction,
                'enable_cuda_graphs': self.gpu.enable_cuda_graphs,
                'enable_tensor_cores': self.gpu.enable_tensor_cores,
                'batch_size': self.gpu.batch_size,
                'mixed_precision': self.gpu.mixed_precision,
            },
            'capture': {
                'fps': self.capture.fps,
                'resolution': self.capture.resolution,
                'buffer_size': self.capture.buffer_size,
                'capture_method': self.capture.capture_method,
                'compress_buffer': self.capture.compress_buffer,
                'jpeg_quality': self.capture.jpeg_quality,
            },
            'detector': {
                'confidence_threshold': self.detector.confidence_threshold,
                'nms_threshold': self.detector.nms_threshold,
                'max_detections': self.detector.max_detections,
                'enable_entity_detection': self.detector.enable_entity_detection,
                'enable_block_detection': self.detector.enable_block_detection,
                'enable_player_detection': self.detector.enable_player_detection,
                'enable_item_detection': self.detector.enable_item_detection,
                'model_size': self.detector.model_size,
            },
            'attention': {
                'saliency_method': self.attention.saliency_method,
                'attention_decay': self.attention.attention_decay,
                'focus_threshold': self.attention.focus_threshold,
                'max_focus_points': self.attention.max_focus_points,
                'enable_motion_attention': self.attention.enable_motion_attention,
                'enable_novelty_attention': self.attention.enable_novelty_attention,
                'peripheral_weight': self.attention.peripheral_weight,
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'embedding_dim': self.embedding.embedding_dim,
                'cache_size': self.embedding.cache_size,
                'normalize': self.embedding.normalize,
                'enable_scene_embedding': self.embedding.enable_scene_embedding,
                'enable_object_embedding': self.embedding.enable_object_embedding,
            },
            'tracker': {
                'tracker_type': self.tracker.tracker_type,
                'max_age': self.tracker.max_age,
                'min_hits': self.tracker.min_hits,
                'iou_threshold': self.tracker.iou_threshold,
                'feature_distance_threshold': self.tracker.feature_distance_threshold,
                'enable_reid': self.tracker.enable_reid,
            },
            'async_processing': self.async_processing,
            'processing_threads': self.processing_threads,
            'max_memory_mb': self.max_memory_mb,
            'cache_dir': self.cache_dir,
            'save_debug_frames': self.save_debug_frames,
            'debug_frame_interval': self.debug_frame_interval,
        }
    
    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> 'VisionConfig':
        """Create from dictionary."""
        gpu_config = GPUConfig(
            mode=GPUMode(d.get('gpu', {}).get('mode', 'auto')),
            device_ids=d.get('gpu', {}).get('device_ids', [0, 1, 2, 3]),
            memory_fraction=d.get('gpu', {}).get('memory_fraction', 0.8),
            enable_cuda_graphs=d.get('gpu', {}).get('enable_cuda_graphs', True),
            enable_tensor_cores=d.get('gpu', {}).get('enable_tensor_cores', True),
            batch_size=d.get('gpu', {}).get('batch_size', 8),
            mixed_precision=d.get('gpu', {}).get('mixed_precision', True),
        )
        
        capture_config = CaptureConfig(
            fps=d.get('capture', {}).get('fps', 30),
            resolution=tuple(d.get('capture', {}).get('resolution', [1920, 1080])),
            buffer_size=d.get('capture', {}).get('buffer_size', 60),
            capture_method=d.get('capture', {}).get('capture_method', 'mss'),
            compress_buffer=d.get('capture', {}).get('compress_buffer', True),
            jpeg_quality=d.get('capture', {}).get('jpeg_quality', 85),
        )
        
        detector_config = DetectorConfig(
            confidence_threshold=d.get('detector', {}).get('confidence_threshold', 0.5),
            nms_threshold=d.get('detector', {}).get('nms_threshold', 0.4),
            max_detections=d.get('detector', {}).get('max_detections', 100),
            enable_entity_detection=d.get('detector', {}).get('enable_entity_detection', True),
            enable_block_detection=d.get('detector', {}).get('enable_block_detection', True),
            enable_player_detection=d.get('detector', {}).get('enable_player_detection', True),
            enable_item_detection=d.get('detector', {}).get('enable_item_detection', True),
            model_size=d.get('detector', {}).get('model_size', 'medium'),
        )
        
        attention_config = AttentionConfig(
            saliency_method=d.get('attention', {}).get('saliency_method', 'spectral'),
            attention_decay=d.get('attention', {}).get('attention_decay', 0.95),
            focus_threshold=d.get('attention', {}).get('focus_threshold', 0.3),
            max_focus_points=d.get('attention', {}).get('max_focus_points', 10),
            enable_motion_attention=d.get('attention', {}).get('enable_motion_attention', True),
            enable_novelty_attention=d.get('attention', {}).get('enable_novelty_attention', True),
            peripheral_weight=d.get('attention', {}).get('peripheral_weight', 0.3),
        )
        
        embedding_config = EmbeddingConfig(
            model_name=d.get('embedding', {}).get('model_name', 'resnet50'),
            embedding_dim=d.get('embedding', {}).get('embedding_dim', 512),
            cache_size=d.get('embedding', {}).get('cache_size', 10000),
            normalize=d.get('embedding', {}).get('normalize', True),
            enable_scene_embedding=d.get('embedding', {}).get('enable_scene_embedding', True),
            enable_object_embedding=d.get('embedding', {}).get('enable_object_embedding', True),
        )
        
        tracker_config = TrackerConfig(
            tracker_type=d.get('tracker', {}).get('tracker_type', 'deep_sort'),
            max_age=d.get('tracker', {}).get('max_age', 30),
            min_hits=d.get('tracker', {}).get('min_hits', 3),
            iou_threshold=d.get('tracker', {}).get('iou_threshold', 0.3),
            feature_distance_threshold=d.get('tracker', {}).get('feature_distance_threshold', 0.5),
            enable_reid=d.get('tracker', {}).get('enable_reid', True),
        )
        
        return cls(
            enabled=d.get('enabled', True),
            gpu=gpu_config,
            capture=capture_config,
            detector=detector_config,
            attention=attention_config,
            embedding=embedding_config,
            tracker=tracker_config,
            async_processing=d.get('async_processing', True),
            processing_threads=d.get('processing_threads', 4),
            max_memory_mb=d.get('max_memory_mb', 4096),
            cache_dir=d.get('cache_dir', 'vision_cache'),
            save_debug_frames=d.get('save_debug_frames', False),
            debug_frame_interval=d.get('debug_frame_interval', 100),
        )
    
    @classmethod
    def create_high_performance(cls) -> 'VisionConfig':
        """Create high-performance config for 4x P104 setup."""
        config = cls()
        config.gpu.mode = GPUMode.MULTI
        config.gpu.device_ids = [0, 1, 2, 3]
        config.gpu.batch_size = 16
        config.gpu.mixed_precision = True
        config.capture.fps = 60
        config.capture.buffer_size = 120
        config.detector.max_detections = 200
        config.embedding.cache_size = 50000
        config.processing_threads = 8
        config.max_memory_mb = 8192
        return config
    
    @classmethod
    def create_balanced(cls) -> 'VisionConfig':
        """Create balanced config for mixed workloads."""
        return cls()  # Default settings are balanced
    
    @classmethod
    def create_low_memory(cls) -> 'VisionConfig':
        """Create low-memory config for constrained systems."""
        config = cls()
        config.gpu.mode = GPUMode.SINGLE
        config.gpu.memory_fraction = 0.5
        config.gpu.batch_size = 4
        config.capture.fps = 15
        config.capture.buffer_size = 30
        config.capture.compress_buffer = True
        config.detector.max_detections = 50
        config.embedding.cache_size = 5000
        config.processing_threads = 2
        config.max_memory_mb = 2048
        return config
