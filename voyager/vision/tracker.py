"""Object tracking across frames.

Implements multi-object tracking for Minecraft entities.
"""

import logging
import time
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

import numpy as np

logger = logging.getLogger(__name__)


class TrackingState(Enum):
    """State of a tracked object."""
    TENTATIVE = "tentative"    # Not yet confirmed
    CONFIRMED = "confirmed"    # Actively tracked
    LOST = "lost"             # Temporarily lost
    DELETED = "deleted"       # Track ended


@dataclass
class TrackedObject:
    """A tracked object across frames."""
    track_id: int
    state: TrackingState
    bbox: 'BoundingBox'
    label: str
    confidence: float
    
    # Tracking history
    first_seen: float
    last_seen: float
    frame_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    # Velocity and prediction
    velocity: Tuple[float, float] = (0.0, 0.0)
    predicted_bbox: Optional['BoundingBox'] = None
    
    # Features for re-identification
    features: Optional[np.ndarray] = None
    feature_history: List[np.ndarray] = field(default_factory=list)
    
    # Metadata
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(
        self,
        bbox: 'BoundingBox',
        confidence: float,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Update track with new detection."""
        # Update velocity estimate
        old_cx, old_cy = self.bbox.center
        new_cx, new_cy = bbox.center
        self.velocity = (new_cx - old_cx, new_cy - old_cy)
        
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frame_count += 1
        self.hit_count += 1
        self.miss_count = 0
        
        if features is not None:
            self.features = features
            if len(self.feature_history) < 10:
                self.feature_history.append(features)
            else:
                self.feature_history.pop(0)
                self.feature_history.append(features)
        
        # State transition
        if self.state == TrackingState.TENTATIVE and self.hit_count >= 3:
            self.state = TrackingState.CONFIRMED
        elif self.state == TrackingState.LOST:
            self.state = TrackingState.CONFIRMED
    
    def mark_missed(self) -> None:
        """Mark a frame where object was not detected."""
        self.miss_count += 1
        self.frame_count += 1
        
        if self.state == TrackingState.CONFIRMED:
            self.state = TrackingState.LOST
    
    def predict(self) -> 'BoundingBox':
        """Predict next position using velocity."""
        from voyager.vision.detector import BoundingBox
        
        vx, vy = self.velocity
        
        self.predicted_bbox = BoundingBox(
            x1=self.bbox.x1 + vx,
            y1=self.bbox.y1 + vy,
            x2=self.bbox.x2 + vx,
            y2=self.bbox.y2 + vy,
        )
        
        return self.predicted_bbox
    
    @property
    def age(self) -> float:
        """Time since first seen."""
        return time.time() - self.first_seen
    
    @property
    def time_since_update(self) -> float:
        """Time since last update."""
        return time.time() - self.last_seen
    
    def get_average_features(self) -> Optional[np.ndarray]:
        """Get average features from history."""
        if not self.feature_history:
            return self.features
        return np.mean(self.feature_history, axis=0)


class KalmanFilter:
    """Simple Kalman filter for position tracking."""
    
    def __init__(self, bbox: 'BoundingBox'):
        """Initialize filter with bounding box."""
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height
        
        # State: [cx, cy, vx, vy, w, h]
        self.state = np.array([cx, cy, 0, 0, w, h], dtype=np.float32)
        
        # Covariance
        self.P = np.eye(6, dtype=np.float32) * 100
        self.P[2:4, 2:4] *= 10  # Higher uncertainty for velocity
        
        # Transition matrix
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 2] = 1  # x += vx
        self.F[1, 3] = 1  # y += vy
        
        # Observation matrix
        self.H = np.eye(4, 6, dtype=np.float32)  # Observe [cx, cy, w, h]
        
        # Process noise
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.Q[2:4, 2:4] *= 2  # More noise for velocity
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0
    
    def predict(self) -> 'BoundingBox':
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self._state_to_bbox()
    
    def update(self, bbox: 'BoundingBox') -> 'BoundingBox':
        """Update with measurement."""
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self._state_to_bbox()
    
    def _state_to_bbox(self) -> 'BoundingBox':
        """Convert state to bounding box."""
        from voyager.vision.detector import BoundingBox
        
        cx, cy, _, _, w, h = self.state
        
        return BoundingBox(
            x1=cx - w / 2,
            y1=cy - h / 2,
            x2=cx + w / 2,
            y2=cy + h / 2,
        )


class ObjectTracker:
    """Multi-object tracker for Minecraft.
    
    Features:
    - IOU-based and appearance-based matching
    - Kalman filter prediction
    - Track lifecycle management
    - Re-identification for lost tracks
    - GPU-accelerated feature extraction
    """
    
    def __init__(
        self,
        tracker_type: str = "deep_sort",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_distance_threshold: float = 0.5,
        enable_reid: bool = True,
        embedder: Optional['ImageEmbedder'] = None,
    ):
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_distance_threshold = feature_distance_threshold
        self.enable_reid = enable_reid
        self.embedder = embedder
        
        self._tracks: Dict[int, TrackedObject] = {}
        self._kalman_filters: Dict[int, KalmanFilter] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        
        # Track statistics
        self._stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
        }
    
    def update(
        self,
        detections: List['DetectedObject'],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected objects
            frame: Optional frame for feature extraction
        
        Returns:
            List of active tracked objects
        """
        with self._lock:
            # Predict new positions for all tracks
            self._predict_all()
            
            # Extract features if enabled
            if self.enable_reid and frame is not None and self.embedder:
                self._extract_features(detections, frame)
            
            # Match detections to tracks
            matches, unmatched_dets, unmatched_tracks = self._match(
                detections
            )
            
            # Update matched tracks
            for track_id, det in matches:
                self._update_track(track_id, det)
            
            # Create new tracks for unmatched detections
            for det in unmatched_dets:
                self._create_track(det)
            
            # Mark unmatched tracks as missed
            for track_id in unmatched_tracks:
                self._mark_missed(track_id)
            
            # Remove dead tracks
            self._remove_dead_tracks()
            
            # Return active tracks
            return self.get_active_tracks()
    
    def _predict_all(self) -> None:
        """Predict next position for all tracks."""
        for track_id, kf in self._kalman_filters.items():
            if track_id in self._tracks:
                predicted = kf.predict()
                self._tracks[track_id].predicted_bbox = predicted
    
    def _extract_features(
        self,
        detections: List['DetectedObject'],
        frame: np.ndarray,
    ) -> None:
        """Extract features for detections."""
        for det in detections:
            bbox = (
                int(det.bbox.x1), int(det.bbox.y1),
                int(det.bbox.x2), int(det.bbox.y2)
            )
            
            # Ensure valid bbox
            h, w = frame.shape[:2]
            bbox = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(w, bbox[2]), min(h, bbox[3])
            )
            
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                embedding = self.embedder.embed_region(frame, bbox)
                det.features = embedding.embedding
    
    def _match(
        self,
        detections: List['DetectedObject'],
    ) -> Tuple[
        List[Tuple[int, 'DetectedObject']],
        List['DetectedObject'],
        List[int]
    ]:
        """Match detections to existing tracks."""
        if not self._tracks or not detections:
            return [], detections, list(self._tracks.keys())
        
        # Compute IOU cost matrix
        track_ids = list(self._tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            pred_bbox = track.predicted_bbox or track.bbox
            
            for j, det in enumerate(detections):
                iou_matrix[i, j] = pred_bbox.iou(det.bbox)
        
        # If using appearance features, compute combined cost
        if self.enable_reid:
            feature_matrix = np.zeros_like(iou_matrix)
            
            for i, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                track_features = track.get_average_features()
                
                if track_features is not None:
                    for j, det in enumerate(detections):
                        if det.features is not None:
                            # Cosine distance
                            sim = np.dot(track_features, det.features)
                            norm = np.linalg.norm(track_features) * np.linalg.norm(det.features)
                            if norm > 0:
                                feature_matrix[i, j] = sim / norm
            
            # Combined cost (lower is better)
            cost_matrix = 1 - (0.5 * iou_matrix + 0.5 * feature_matrix)
        else:
            cost_matrix = 1 - iou_matrix
        
        # Hungarian algorithm for optimal assignment
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1 - self.iou_threshold:
                    matches.append((track_ids[row], detections[col]))
                    if col in unmatched_dets:
                        unmatched_dets.remove(col)
                    if row in unmatched_tracks:
                        unmatched_tracks.remove(row)
        except ImportError:
            # Greedy matching fallback
            while True:
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_val = cost_matrix[min_idx]
                
                if min_val >= 1 - self.iou_threshold:
                    break
                
                row, col = min_idx
                if row in unmatched_tracks and col in unmatched_dets:
                    matches.append((track_ids[row], detections[col]))
                    unmatched_dets.remove(col)
                    unmatched_tracks.remove(row)
                
                cost_matrix[row, :] = float('inf')
                cost_matrix[:, col] = float('inf')
        
        # Convert indices to actual values
        unmatched_dets = [detections[i] for i in unmatched_dets]
        unmatched_tracks = [track_ids[i] for i in unmatched_tracks]
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _update_track(
        self,
        track_id: int,
        detection: 'DetectedObject',
    ) -> None:
        """Update track with matched detection."""
        track = self._tracks[track_id]
        
        # Update Kalman filter
        if track_id in self._kalman_filters:
            filtered_bbox = self._kalman_filters[track_id].update(detection.bbox)
        else:
            filtered_bbox = detection.bbox
        
        track.update(
            bbox=filtered_bbox,
            confidence=detection.confidence,
            features=detection.features,
        )
    
    def _create_track(self, detection: 'DetectedObject') -> TrackedObject:
        """Create new track from detection."""
        from voyager.vision.detector import BoundingBox
        
        track_id = self._next_id
        self._next_id += 1
        
        track = TrackedObject(
            track_id=track_id,
            state=TrackingState.TENTATIVE,
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            first_seen=time.time(),
            last_seen=time.time(),
            frame_count=1,
            hit_count=1,
            features=detection.features,
            category=detection.category.value if detection.category else None,
        )
        
        if detection.features is not None:
            track.feature_history.append(detection.features)
        
        self._tracks[track_id] = track
        self._kalman_filters[track_id] = KalmanFilter(detection.bbox)
        
        self._stats['total_tracks'] += 1
        
        return track
    
    def _mark_missed(self, track_id: int) -> None:
        """Mark track as missed in this frame."""
        if track_id in self._tracks:
            self._tracks[track_id].mark_missed()
            
            # Update Kalman with prediction only
            if track_id in self._kalman_filters:
                self._kalman_filters[track_id].predict()
    
    def _remove_dead_tracks(self) -> None:
        """Remove tracks that have been lost too long."""
        to_remove = []
        
        for track_id, track in self._tracks.items():
            if track.miss_count > self.max_age:
                track.state = TrackingState.DELETED
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracks[track_id]
            if track_id in self._kalman_filters:
                del self._kalman_filters[track_id]
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracks."""
        return [
            track for track in self._tracks.values()
            if track.state in [TrackingState.CONFIRMED, TrackingState.LOST]
        ]
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get only confirmed tracks."""
        return [
            track for track in self._tracks.values()
            if track.state == TrackingState.CONFIRMED
        ]
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID."""
        return self._tracks.get(track_id)
    
    def get_tracks_by_label(self, label: str) -> List[TrackedObject]:
        """Get tracks matching a label."""
        return [
            track for track in self._tracks.values()
            if track.label == label and track.state != TrackingState.DELETED
        ]
    
    def reset(self) -> None:
        """Reset all tracks."""
        with self._lock:
            self._tracks.clear()
            self._kalman_filters.clear()
            self._next_id = 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        self._stats['active_tracks'] = len(self.get_active_tracks())
        self._stats['lost_tracks'] = len([
            t for t in self._tracks.values()
            if t.state == TrackingState.LOST
        ])
        return self._stats.copy()
