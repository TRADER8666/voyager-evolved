"""Vision system integration with Voyager Evolved.

Connects vision and memory systems with existing player observation,
evolutionary goals, and decision-making systems.

"""

import logging
import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from voyager.vision.config import VisionConfig
from voyager.vision.gpu_manager import GPUManager, get_gpu_manager
from voyager.vision.capture import ScreenCapture, Frame, FrameBuffer
from voyager.vision.detector import MinecraftDetector, DetectedObject, ObjectCategory
from voyager.vision.attention import VisualAttention, AttentionFocus, AttentionType
from voyager.vision.embeddings import ImageEmbedder, ImageEmbedding
from voyager.vision.tracker import ObjectTracker, TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class VisualObservation:
    """A complete visual observation of the game state."""
    timestamp: float
    frame_number: int
    
    # Detection results
    detected_objects: List[DetectedObject] = field(default_factory=list)
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    
    # Attention
    attention_points: List[AttentionFocus] = field(default_factory=list)
    primary_focus: Optional[AttentionFocus] = None
    
    # Scene understanding
    scene_embedding: Optional[ImageEmbedding] = None
    scene_description: str = ""
    
    # Specific detections
    players_visible: List[str] = field(default_factory=list)
    hostile_mobs: List[str] = field(default_factory=list)
    passive_mobs: List[str] = field(default_factory=list)
    items_visible: List[str] = field(default_factory=list)
    
    # Statistics
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'num_detected': len(self.detected_objects),
            'num_tracked': len(self.tracked_objects),
            'num_attention_points': len(self.attention_points),
            'primary_focus': {
                'x': self.primary_focus.x,
                'y': self.primary_focus.y,
                'type': self.primary_focus.attention_type.value,
            } if self.primary_focus else None,
            'players_visible': self.players_visible,
            'hostile_mobs': self.hostile_mobs,
            'passive_mobs': self.passive_mobs,
            'items_visible': self.items_visible,
            'processing_time': self.processing_time,
        }


class VisionSystem:
    """Integrated vision system for Voyager Evolved.
    
    Features:
    - GPU-accelerated processing across 4x P104 GPUs
    - Real-time object detection and tracking
    - Visual attention mechanism
    - Scene understanding and embedding
    - Integration with player observation system
    - Visual memory storage
    """
    
    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        memory_store=None,
    ):
        self.config = config or VisionConfig()
        self.memory_store = memory_store
        
        self._initialized = False
        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Components
        self._gpu_manager: Optional[GPUManager] = None
        self._screen_capture: Optional[ScreenCapture] = None
        self._detector: Optional[MinecraftDetector] = None
        self._attention: Optional[VisualAttention] = None
        self._embedder: Optional[ImageEmbedder] = None
        self._tracker: Optional[ObjectTracker] = None
        
        # State
        self._latest_observation: Optional[VisualObservation] = None
        self._observation_callbacks: List[Callable[[VisualObservation], None]] = []
        self._frame_count = 0
        
        # Processing pool
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Statistics
        self._stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'players_seen': set(),
        }
        
        if self.config.enabled:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize all vision components."""
        logger.info("Initializing vision system...")
        
        try:
            # Initialize GPU manager
            self._gpu_manager = get_gpu_manager(
                device_ids=self.config.gpu.device_ids,
                memory_fraction=self.config.gpu.memory_fraction,
                mixed_precision=self.config.gpu.mixed_precision,
            )
            
            # Get primary device
            device = str(self._gpu_manager.get_device())
            logger.info(f"Using device: {device}")
            
            # Initialize screen capture
            self._screen_capture = ScreenCapture(
                fps=self.config.capture.fps,
                resolution=self.config.capture.resolution,
                capture_method=self.config.capture.capture_method,
                buffer_size=self.config.capture.buffer_size,
                compress_buffer=self.config.capture.compress_buffer,
            )
            
            # Initialize detector
            self._detector = MinecraftDetector(
                confidence_threshold=self.config.detector.confidence_threshold,
                nms_threshold=self.config.detector.nms_threshold,
                max_detections=self.config.detector.max_detections,
                enable_entity_detection=self.config.detector.enable_entity_detection,
                enable_player_detection=self.config.detector.enable_player_detection,
                model_size=self.config.detector.model_size,
                device=device,
            )
            
            # Initialize attention
            self._attention = VisualAttention(
                saliency_method=self.config.attention.saliency_method,
                attention_decay=self.config.attention.attention_decay,
                focus_threshold=self.config.attention.focus_threshold,
                max_focus_points=self.config.attention.max_focus_points,
                enable_motion_attention=self.config.attention.enable_motion_attention,
                enable_novelty_attention=self.config.attention.enable_novelty_attention,
                device=device,
            )
            
            # Initialize embedder
            self._embedder = ImageEmbedder(
                model_name=self.config.embedding.model_name,
                embedding_dim=self.config.embedding.embedding_dim,
                cache_size=self.config.embedding.cache_size,
                normalize=self.config.embedding.normalize,
                device=device,
            )
            
            # Initialize tracker
            self._tracker = ObjectTracker(
                tracker_type=self.config.tracker.tracker_type,
                max_age=self.config.tracker.max_age,
                min_hits=self.config.tracker.min_hits,
                iou_threshold=self.config.tracker.iou_threshold,
                enable_reid=self.config.tracker.enable_reid,
                embedder=self._embedder,
            )
            
            # Initialize thread pool
            if self.config.async_processing:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.config.processing_threads
                )
            
            self._initialized = True
            logger.info("Vision system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision system: {e}")
            self._initialized = False
    
    def start(self) -> None:
        """Start the vision processing system."""
        if not self._initialized:
            logger.warning("Vision system not initialized")
            return
        
        if self._running:
            return
        
        self._running = True
        
        # Start screen capture
        self._screen_capture.start()
        
        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="VisionProcessor"
        )
        self._process_thread.start()
        
        logger.info("Vision system started")
    
    def stop(self) -> None:
        """Stop the vision processing system."""
        self._running = False
        
        if self._screen_capture:
            self._screen_capture.stop()
        
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
        
        if self._executor:
            self._executor.shutdown(wait=False)
        
        logger.info("Vision system stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Get latest frame
                frame = self._screen_capture.latest_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                observation = self.process_frame(frame)
                
                if observation:
                    with self._lock:
                        self._latest_observation = observation
                    
                    # Notify callbacks
                    for callback in self._observation_callbacks:
                        try:
                            callback(observation)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
    
    def process_frame(self, frame: Frame) -> Optional[VisualObservation]:
        """Process a single frame.
        
        Args:
            frame: Captured frame
        
        Returns:
            VisualObservation with all results
        """
        if not self._initialized:
            return None
        
        start_time = time.time()
        
        # Get image
        if frame.compressed:
            # Decompress if needed
            import cv2
            image_array = np.frombuffer(frame.image, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = frame.image
        
        # Run detection
        detections = self._detector.detect(image, frame.frame_number)
        
        # Update tracker
        tracked = self._tracker.update(detections, image)
        
        # Update attention
        attention_points = self._attention.update_attention(image, detections)
        primary_focus = self._attention.get_primary_focus()
        
        # Generate scene embedding
        scene_embedding = self._embedder.embed(image)
        
        # Categorize detections
        players = []
        hostile = []
        passive = []
        items = []
        
        for det in detections:
            if det.category == ObjectCategory.PLAYER:
                players.append(det.label)
                self._stats['players_seen'].add(det.label)
            elif det.category == ObjectCategory.HOSTILE_MOB:
                hostile.append(det.label)
            elif det.category == ObjectCategory.PASSIVE_MOB:
                passive.append(det.label)
            elif det.category == ObjectCategory.ITEM:
                items.append(det.label)
        
        processing_time = time.time() - start_time
        
        # Create observation
        observation = VisualObservation(
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
            detected_objects=detections,
            tracked_objects=tracked,
            attention_points=attention_points,
            primary_focus=primary_focus,
            scene_embedding=scene_embedding,
            players_visible=players,
            hostile_mobs=hostile,
            passive_mobs=passive,
            items_visible=items,
            processing_time=processing_time,
        )
        
        # Update stats
        self._stats['total_frames'] += 1
        self._stats['total_detections'] += len(detections)
        self._stats['avg_processing_time'] = (
            0.9 * self._stats['avg_processing_time'] + 0.1 * processing_time
        )
        
        self._frame_count += 1
        
        return observation
    
    def get_latest_observation(self) -> Optional[VisualObservation]:
        """Get the most recent visual observation."""
        with self._lock:
            return self._latest_observation
    
    def register_callback(
        self,
        callback: Callable[[VisualObservation], None]
    ) -> None:
        """Register a callback for new observations."""
        self._observation_callbacks.append(callback)
    
    def unregister_callback(
        self,
        callback: Callable[[VisualObservation], None]
    ) -> None:
        """Unregister an observation callback."""
        if callback in self._observation_callbacks:
            self._observation_callbacks.remove(callback)
    
    def get_attention_heatmap(self) -> Optional[np.ndarray]:
        """Get current attention heatmap."""
        if self._attention:
            return self._attention.get_attention_heatmap(
                self.config.capture.resolution[0],
                self.config.capture.resolution[1]
            )
        return None
    
    def find_similar_scenes(
        self,
        embedding: ImageEmbedding,
        n_results: int = 5,
    ) -> List[Tuple[ImageEmbedding, float]]:
        """Find similar scenes from memory."""
        if not self.memory_store:
            return []
        
        # This would search the memory store for similar scenes
        # Implementation depends on memory system integration
        return []
    
    def capture_snapshot(self) -> Optional[Frame]:
        """Manually capture a single frame."""
        if self._screen_capture:
            return self._screen_capture.capture_frame()
        return None
    
    def get_tracked_players(self) -> List[TrackedObject]:
        """Get all tracked player objects."""
        if self._tracker:
            return self._tracker.get_tracks_by_label("player")
        return []
    
    def get_threats(self) -> List[DetectedObject]:
        """Get all detected threats (hostile mobs)."""
        obs = self.get_latest_observation()
        if obs:
            return [
                det for det in obs.detected_objects
                if det.category == ObjectCategory.HOSTILE_MOB
            ]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vision system statistics."""
        return {
            **self._stats,
            'initialized': self._initialized,
            'running': self._running,
            'gpu_available': self._gpu_manager.is_available if self._gpu_manager else False,
            'capture_stats': self._screen_capture.get_stats() if self._screen_capture else {},
            'tracker_stats': self._tracker.get_stats() if self._tracker else {},
            'embedder_stats': self._embedder.get_stats() if self._embedder else {},
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class VisualPlayerObserver:
    """Enhanced player observer with visual capabilities.
    
    Extends the existing PlayerObserver with visual detection
    and recognition capabilities.
    """
    
    def __init__(
        self,
        vision_system: VisionSystem,
        player_observer=None,  # Original PlayerObserver
    ):
        self.vision = vision_system
        self.player_observer = player_observer
        
        # Player visual memory
        self._player_embeddings: Dict[str, List[ImageEmbedding]] = {}
        self._player_appearances: Dict[str, Dict[str, Any]] = {}
        
        # Register callback
        self.vision.register_callback(self._on_visual_observation)
    
    def _on_visual_observation(self, observation: VisualObservation) -> None:
        """Handle new visual observation."""
        # Process detected players
        for det in observation.detected_objects:
            if det.category == ObjectCategory.PLAYER:
                self._update_player_visual(det, observation)
        
        # If we have an existing player observer, update it
        if self.player_observer:
            # Convert visual detections to player observer format
            visual_players = self._extract_visual_player_data(observation)
            # This would feed into the existing player observation system
    
    def _update_player_visual(
        self,
        detection: DetectedObject,
        observation: VisualObservation,
    ) -> None:
        """Update visual information about a player."""
        player_name = detection.label
        
        if player_name not in self._player_embeddings:
            self._player_embeddings[player_name] = []
        
        # Store appearance embedding if available
        if detection.features is not None:
            from voyager.vision.embeddings import ImageEmbedding
            embedding = ImageEmbedding(
                embedding=detection.features,
                timestamp=observation.timestamp,
                source_hash=f"player_{player_name}_{observation.frame_number}",
                embedding_type="player_appearance",
            )
            self._player_embeddings[player_name].append(embedding)
            
            # Keep only recent embeddings
            if len(self._player_embeddings[player_name]) > 10:
                self._player_embeddings[player_name].pop(0)
        
        # Update appearance info
        self._player_appearances[player_name] = {
            'last_seen': observation.timestamp,
            'last_position': detection.bbox.center,
            'confidence': detection.confidence,
        }
    
    def _extract_visual_player_data(
        self,
        observation: VisualObservation,
    ) -> List[Dict[str, Any]]:
        """Extract player data from visual observation."""
        players = []
        
        for det in observation.detected_objects:
            if det.category == ObjectCategory.PLAYER:
                players.append({
                    'name': det.label,
                    'position': det.bbox.center,
                    'bbox': det.bbox.to_xywh(),
                    'confidence': det.confidence,
                })
        
        return players
    
    def recognize_player(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
    ) -> Optional[str]:
        """Try to recognize a player by their appearance."""
        best_match = None
        best_similarity = threshold
        
        for player_name, embeddings in self._player_embeddings.items():
            if not embeddings:
                continue
            
            # Average similarity with stored embeddings
            similarities = []
            for stored in embeddings:
                sim = np.dot(embedding, stored.embedding)
                norm = np.linalg.norm(embedding) * np.linalg.norm(stored.embedding)
                if norm > 0:
                    similarities.append(sim / norm)
            
            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_match = player_name
        
        return best_match
    
    def get_player_last_seen(self, player_name: str) -> Optional[float]:
        """Get timestamp when player was last visually seen."""
        if player_name in self._player_appearances:
            return self._player_appearances[player_name].get('last_seen')
        return None
    
    def get_visible_players(self) -> List[str]:
        """Get list of currently visible players."""
        obs = self.vision.get_latest_observation()
        if obs:
            return obs.players_visible
        return []


# Factory function
def create_vision_system(
    config: Optional[VisionConfig] = None,
    memory_store=None,
) -> VisionSystem:
    """Create a configured vision system."""
    return VisionSystem(config=config, memory_store=memory_store)
