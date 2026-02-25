"""Visual attention mechanism for Minecraft AI.

Implements saliency-based attention to determine what the agent notices.
"""

import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

import numpy as np

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of visual attention."""
    SALIENCY = "saliency"      # Bottom-up attention from visual features
    MOTION = "motion"          # Motion-based attention
    NOVELTY = "novelty"        # Attention to novel/unusual elements
    TASK = "task"              # Top-down task-driven attention
    THREAT = "threat"          # Attention to potential threats
    SOCIAL = "social"          # Attention to other players


@dataclass
class AttentionFocus:
    """A single point of visual attention."""
    x: float  # Screen x coordinate (0-1 normalized)
    y: float  # Screen y coordinate (0-1 normalized)
    strength: float  # Attention strength (0-1)
    attention_type: AttentionType
    label: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.95
    
    @property
    def pixel_coords(self, width: int = 1920, height: int = 1080) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))
    
    def update(self, strength_boost: float = 0.0) -> None:
        """Update attention with decay and optional boost."""
        self.strength = min(1.0, self.strength * self.decay_rate + strength_boost)
        self.timestamp = time.time()
    
    def is_expired(self, min_strength: float = 0.1) -> bool:
        """Check if attention has decayed below threshold."""
        return self.strength < min_strength


@dataclass
class SaliencyMap:
    """Visual saliency map."""
    map: np.ndarray  # 2D array of saliency values (0-1)
    timestamp: float
    resolution: Tuple[int, int]
    max_points: List[Tuple[int, int, float]] = field(default_factory=list)  # (x, y, value)
    
    def get_top_k_points(self, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k most salient points."""
        if self.max_points:
            return sorted(self.max_points, key=lambda p: p[2], reverse=True)[:k]
        
        # Find local maxima
        from scipy import ndimage
        try:
            local_max = ndimage.maximum_filter(self.map, size=20)
            maxima = (self.map == local_max) & (self.map > 0.3)
            coords = np.argwhere(maxima)
            
            points = [
                (int(c[1]), int(c[0]), float(self.map[c[0], c[1]]))
                for c in coords
            ]
            return sorted(points, key=lambda p: p[2], reverse=True)[:k]
        except ImportError:
            # Fallback without scipy
            flat_idx = np.argsort(self.map.flatten())[-k:][::-1]
            points = []
            for idx in flat_idx:
                y, x = np.unravel_index(idx, self.map.shape)
                points.append((int(x), int(y), float(self.map[y, x])))
            return points
    
    def normalize(self) -> None:
        """Normalize saliency map to 0-1 range."""
        min_val = self.map.min()
        max_val = self.map.max()
        if max_val > min_val:
            self.map = (self.map - min_val) / (max_val - min_val)


class VisualAttention:
    """Visual attention system for determining what the agent notices.
    
    Features:
    - Multiple saliency computation methods
    - Motion-based attention
    - Novelty detection
    - Attention decay over time
    - Task-driven attention modulation
    - Threat prioritization
    """
    
    def __init__(
        self,
        saliency_method: str = "spectral",
        attention_decay: float = 0.95,
        focus_threshold: float = 0.3,
        max_focus_points: int = 10,
        enable_motion_attention: bool = True,
        enable_novelty_attention: bool = True,
        peripheral_weight: float = 0.3,
        device: Optional[str] = None,
    ):
        self.saliency_method = saliency_method
        self.attention_decay = attention_decay
        self.focus_threshold = focus_threshold
        self.max_focus_points = max_focus_points
        self.enable_motion_attention = enable_motion_attention
        self.enable_novelty_attention = enable_novelty_attention
        self.peripheral_weight = peripheral_weight
        
        self._device = None
        self._cv2_available = False
        self._torch_available = False
        
        # Attention state
        self._focus_points: List[AttentionFocus] = []
        self._previous_frame: Optional[np.ndarray] = None
        self._frame_history: deque = deque(maxlen=30)
        self._novelty_baseline: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        
        # Task-driven attention targets
        self._task_targets: List[Dict[str, Any]] = []
        
        self._initialize(device)
    
    def _initialize(self, device: Optional[str]) -> None:
        """Initialize attention system."""
        try:
            import cv2
            self._cv2_available = True
        except ImportError:
            logger.warning("OpenCV not available for attention")
        
        try:
            import torch
            self._torch_available = True
            if device:
                self._device = torch.device(device)
            elif torch.cuda.is_available():
                self._device = torch.device('cuda:0')
            else:
                self._device = torch.device('cpu')
        except ImportError:
            logger.warning("PyTorch not available for attention")
    
    def compute_saliency(self, image: np.ndarray) -> SaliencyMap:
        """Compute visual saliency map."""
        if self.saliency_method == "spectral":
            return self._spectral_saliency(image)
        elif self.saliency_method == "itti-koch":
            return self._itti_koch_saliency(image)
        elif self.saliency_method == "deep":
            return self._deep_saliency(image)
        else:
            return self._spectral_saliency(image)
    
    def _spectral_saliency(self, image: np.ndarray) -> SaliencyMap:
        """Compute spectral residual saliency.
        
        Fast and effective method based on FFT.
        """
        if not self._cv2_available:
            return self._empty_saliency(image)
        
        import cv2
        
        # Convert to grayscale and resize for speed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        
        # Compute FFT
        fft = np.fft.fft2(small.astype(np.float32))
        
        # Get log magnitude
        magnitude = np.log(np.abs(fft) + 1e-10)
        phase = np.angle(fft)
        
        # Compute spectral residual
        kernel = np.ones((3, 3)) / 9
        smoothed = cv2.filter2D(magnitude, -1, kernel)
        residual = magnitude - smoothed
        
        # Reconstruct saliency map
        saliency_fft = np.exp(residual) * np.exp(1j * phase)
        saliency_small = np.abs(np.fft.ifft2(saliency_fft)) ** 2
        
        # Resize back to original size
        h, w = image.shape[:2]
        saliency = cv2.resize(saliency_small, (w, h))
        
        # Gaussian blur and normalize
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return SaliencyMap(
            map=saliency.astype(np.float32),
            timestamp=time.time(),
            resolution=(w, h),
        )
    
    def _itti_koch_saliency(self, image: np.ndarray) -> SaliencyMap:
        """Compute Itti-Koch style saliency.
        
        Combines color, intensity, and orientation features.
        """
        if not self._cv2_available:
            return self._empty_saliency(image)
        
        import cv2
        
        h, w = image.shape[:2]
        
        # Create feature maps
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Intensity channel
        intensity = gray / 255.0
        
        # Color channels (RG, BY opponent colors)
        b, g, r = [image[:, :, i].astype(np.float32) / 255.0 for i in range(3)]
        rg = r - g
        by = (r + g) / 2 - np.abs(r - g) / 2 - b
        
        # Orientation using Gabor filters
        orientations = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=np.radians(theta),
                lambd=10.0, gamma=0.5
            )
            orientation = cv2.filter2D(gray, cv2.CV_32F, kernel)
            orientations.append(np.abs(orientation))
        
        # Combine feature maps
        features = [intensity, np.abs(rg), np.abs(by)] + orientations
        
        # Multi-scale center-surround
        saliency = np.zeros((h, w), dtype=np.float32)
        for feature in features:
            for scale in [2, 4, 8]:
                center = cv2.resize(feature, (w // scale, h // scale))
                surround = cv2.GaussianBlur(center, (5, 5), 0)
                diff = np.abs(center - surround)
                diff = cv2.resize(diff, (w, h))
                saliency += diff
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return SaliencyMap(
            map=saliency,
            timestamp=time.time(),
            resolution=(w, h),
        )
    
    def _deep_saliency(self, image: np.ndarray) -> SaliencyMap:
        """Compute deep learning based saliency."""
        # Use a lightweight CNN if available, otherwise fall back
        if not self._torch_available:
            return self._spectral_saliency(image)
        
        # For now, use spectral as placeholder
        # In production, this would use a trained saliency model
        return self._spectral_saliency(image)
    
    def _empty_saliency(self, image: np.ndarray) -> SaliencyMap:
        """Create an empty saliency map."""
        h, w = image.shape[:2]
        return SaliencyMap(
            map=np.zeros((h, w), dtype=np.float32),
            timestamp=time.time(),
            resolution=(w, h),
        )
    
    def compute_motion_attention(self, image: np.ndarray) -> Optional[SaliencyMap]:
        """Compute motion-based attention from optical flow."""
        if not self._cv2_available or not self.enable_motion_attention:
            return None
        
        import cv2
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self._previous_frame is None:
            self._previous_frame = gray
            return None
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._previous_frame, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Convert flow to magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        
        # Normalize
        motion_saliency = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)
        
        self._previous_frame = gray
        
        return SaliencyMap(
            map=motion_saliency.astype(np.float32),
            timestamp=time.time(),
            resolution=(image.shape[1], image.shape[0]),
        )
    
    def compute_novelty_attention(self, image: np.ndarray) -> Optional[SaliencyMap]:
        """Compute novelty-based attention by comparing to learned baseline."""
        if not self._cv2_available or not self.enable_novelty_attention:
            return None
        
        import cv2
        
        # Resize for comparison
        small = cv2.resize(image, (128, 128))
        small_float = small.astype(np.float32) / 255.0
        
        # Update baseline with exponential moving average
        if self._novelty_baseline is None:
            self._novelty_baseline = small_float
            return None
        
        # Compute difference from baseline
        diff = np.abs(small_float - self._novelty_baseline)
        novelty = np.mean(diff, axis=2)  # Combine color channels
        
        # Update baseline slowly
        self._novelty_baseline = 0.99 * self._novelty_baseline + 0.01 * small_float
        
        # Resize back
        h, w = image.shape[:2]
        novelty = cv2.resize(novelty, (w, h))
        
        return SaliencyMap(
            map=novelty.astype(np.float32),
            timestamp=time.time(),
            resolution=(w, h),
        )
    
    def update_attention(
        self,
        image: np.ndarray,
        detections: Optional[List] = None,
    ) -> List[AttentionFocus]:
        """Update attention based on visual input.
        
        Args:
            image: Current frame
            detections: Optional list of detected objects
        
        Returns:
            List of current attention focus points
        """
        with self._lock:
            h, w = image.shape[:2]
            
            # Decay existing attention
            self._decay_attention()
            
            # Compute saliency
            saliency = self.compute_saliency(image)
            
            # Add saliency-based attention
            top_points = saliency.get_top_k_points(5)
            for x, y, strength in top_points:
                if strength > self.focus_threshold:
                    self._add_or_update_focus(
                        x / w, y / h, strength,
                        AttentionType.SALIENCY,
                        "salient_region"
                    )
            
            # Add motion attention
            motion = self.compute_motion_attention(image)
            if motion:
                motion_points = motion.get_top_k_points(3)
                for x, y, strength in motion_points:
                    if strength > 0.4:
                        self._add_or_update_focus(
                            x / w, y / h, strength * 0.8,
                            AttentionType.MOTION,
                            "moving_object"
                        )
            
            # Add novelty attention
            novelty = self.compute_novelty_attention(image)
            if novelty:
                novelty_points = novelty.get_top_k_points(3)
                for x, y, strength in novelty_points:
                    if strength > 0.5:
                        self._add_or_update_focus(
                            x / w, y / h, strength * 0.7,
                            AttentionType.NOVELTY,
                            "novel_element"
                        )
            
            # Add attention for detected objects
            if detections:
                self._process_detections(detections, w, h)
            
            # Limit focus points
            self._prune_focus_points()
            
            return list(self._focus_points)
    
    def _decay_attention(self) -> None:
        """Decay all attention points."""
        for focus in self._focus_points:
            focus.update()
        
        # Remove expired
        self._focus_points = [
            f for f in self._focus_points if not f.is_expired()
        ]
    
    def _add_or_update_focus(
        self,
        x: float,
        y: float,
        strength: float,
        attention_type: AttentionType,
        label: str,
    ) -> None:
        """Add new focus or update existing nearby focus."""
        # Check for existing nearby focus
        for focus in self._focus_points:
            distance = ((focus.x - x) ** 2 + (focus.y - y) ** 2) ** 0.5
            if distance < 0.1:  # Within 10% of screen
                focus.update(strength * 0.3)  # Boost existing
                return
        
        # Add new focus
        self._focus_points.append(AttentionFocus(
            x=x,
            y=y,
            strength=strength,
            attention_type=attention_type,
            label=label,
            decay_rate=self.attention_decay,
        ))
    
    def _process_detections(
        self,
        detections: List,
        width: int,
        height: int,
    ) -> None:
        """Process detected objects for attention."""
        from voyager.vision.detector import ObjectCategory
        
        for det in detections:
            cx, cy = det.bbox.center
            nx, ny = cx / width, cy / height
            
            # Determine attention type based on category
            if det.category == ObjectCategory.PLAYER:
                attention_type = AttentionType.SOCIAL
                strength = det.confidence * 0.9
            elif det.category == ObjectCategory.HOSTILE_MOB:
                attention_type = AttentionType.THREAT
                strength = det.confidence * 1.0  # High priority
            elif det.category in [ObjectCategory.ITEM]:
                attention_type = AttentionType.SALIENCY
                strength = det.confidence * 0.6
            else:
                attention_type = AttentionType.SALIENCY
                strength = det.confidence * 0.5
            
            self._add_or_update_focus(
                nx, ny, strength, attention_type, det.label
            )
    
    def _prune_focus_points(self) -> None:
        """Limit focus points to max allowed."""
        if len(self._focus_points) > self.max_focus_points:
            # Keep strongest points
            self._focus_points = sorted(
                self._focus_points,
                key=lambda f: f.strength,
                reverse=True
            )[:self.max_focus_points]
    
    def add_task_target(
        self,
        target_type: str,
        description: str,
        screen_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Add a task-driven attention target."""
        self._task_targets.append({
            'type': target_type,
            'description': description,
            'region': screen_region,
            'timestamp': time.time(),
        })
    
    def clear_task_targets(self) -> None:
        """Clear all task targets."""
        self._task_targets.clear()
    
    def get_primary_focus(self) -> Optional[AttentionFocus]:
        """Get the strongest attention focus."""
        if not self._focus_points:
            return None
        return max(self._focus_points, key=lambda f: f.strength)
    
    def get_focus_by_type(
        self,
        attention_type: AttentionType
    ) -> List[AttentionFocus]:
        """Get all focus points of a specific type."""
        return [
            f for f in self._focus_points
            if f.attention_type == attention_type
        ]
    
    def get_attention_heatmap(
        self,
        width: int = 1920,
        height: int = 1080,
    ) -> np.ndarray:
        """Generate attention heatmap for visualization."""
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        if not self._cv2_available:
            return heatmap
        
        import cv2
        
        for focus in self._focus_points:
            x = int(focus.x * width)
            y = int(focus.y * height)
            
            # Create Gaussian blob
            radius = int(50 * focus.strength + 10)
            cv2.circle(heatmap, (x, y), radius, focus.strength, -1)
        
        # Blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        return heatmap
