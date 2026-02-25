"""Enhanced Player Observation System for Voyager Evolved (Linux Optimized).

This module provides advanced functionality to detect, track, and observe other players
on the Minecraft server, with pattern recognition, behavior clustering, and memory decay.

Optimizations:
- Linux-specific memory management using psutil
- Batch processing for observation updates
- Efficient numpy-based distance calculations
- Pattern recognition with sklearn clustering
"""

import time
import json
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from functools import lru_cache
import voyager.utils as U

# Linux optimizations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PlayerActivity(Enum):
    """Types of activities that can be observed from players."""
    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    MINING = "mining"
    BUILDING = "building"
    CRAFTING = "crafting"
    FIGHTING = "fighting"
    FARMING = "farming"
    TRADING = "trading"
    EXPLORING = "exploring"
    GATHERING = "gathering"
    SWIMMING = "swimming"
    CLIMBING = "climbing"
    SNEAKING = "sneaking"
    JUMPING = "jumping"
    UNKNOWN = "unknown"


@dataclass
class PlayerSnapshot:
    """A snapshot of a player's state at a moment in time."""
    timestamp: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0, 0, 0)
    yaw: float = 0
    pitch: float = 0
    held_item: Optional[str] = None
    health: float = 20.0
    is_sneaking: bool = False
    is_sprinting: bool = False
    is_on_ground: bool = True
    nearby_blocks: List[str] = field(default_factory=list)
    nearby_entities: List[str] = field(default_factory=list)


@dataclass
class BehaviorPattern:
    """A recognized pattern of behavior."""
    pattern_id: str
    activity_sequence: List[PlayerActivity]
    avg_duration: float
    occurrences: int
    success_rate: float
    context_requirements: Dict[str, Any]
    confidence: float
    tools_typically_used: List[str]
    
    def similarity(self, other: 'BehaviorPattern') -> float:
        """Calculate similarity between two patterns."""
        if len(self.activity_sequence) != len(other.activity_sequence):
            return 0.0
        matches = sum(1 for a, b in zip(self.activity_sequence, other.activity_sequence) if a == b)
        return matches / len(self.activity_sequence)


@dataclass
class ObservedBehavior:
    """A recorded behavior pattern observed from a player."""
    player_name: str
    activity: PlayerActivity
    start_time: float
    end_time: float
    success: bool
    context: Dict[str, Any]
    tools_used: List[str]
    blocks_involved: List[str]
    items_gained: List[str]
    items_lost: List[str]
    sequence: List[PlayerSnapshot]
    confidence: float
    pattern_id: Optional[str] = None
    decay_factor: float = 1.0  # Memory decay
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def apply_decay(self, decay_rate: float):
        """Apply memory decay to this observation."""
        age = time.time() - self.end_time
        self.decay_factor = max(0.1, 1.0 - (age * decay_rate / 3600))  # Per hour decay
    
    def relevance_score(self) -> float:
        """Get relevance score considering confidence and decay."""
        return self.confidence * self.decay_factor
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['activity'] = self.activity.value
        data['sequence'] = [asdict(s) for s in self.sequence]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObservedBehavior':
        data['activity'] = PlayerActivity(data['activity'])
        data['sequence'] = [PlayerSnapshot(**s) for s in data['sequence']]
        return cls(**data)


@dataclass
class PlayerProfile:
    """Enhanced profile of an observed player built over time."""
    name: str
    first_seen: float
    last_seen: float
    total_observation_time: float = 0.0
    activity_counts: Dict[str, int] = field(default_factory=dict)
    activity_transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    preferred_tools: Dict[str, int] = field(default_factory=dict)
    skill_demonstrations: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.5
    behavior_patterns: List[str] = field(default_factory=list)
    locations_frequented: List[Tuple[float, float, float]] = field(default_factory=list)
    play_style: str = "unknown"  # miner, builder, explorer, etc.
    skill_level: float = 0.5  # 0 to 1 estimated skill
    predictability: float = 0.5  # How predictable their actions are
    
    def update_activity(self, activity: PlayerActivity, previous: Optional[PlayerActivity] = None):
        """Update activity tracking with transition data."""
        activity_name = activity.value
        self.activity_counts[activity_name] = self.activity_counts.get(activity_name, 0) + 1
        
        # Track transitions for pattern recognition
        if previous:
            prev_name = previous.value
            if prev_name not in self.activity_transitions:
                self.activity_transitions[prev_name] = {}
            self.activity_transitions[prev_name][activity_name] = \
                self.activity_transitions[prev_name].get(activity_name, 0) + 1
    
    def get_primary_activity(self) -> Optional[str]:
        if not self.activity_counts:
            return None
        return max(self.activity_counts.items(), key=lambda x: x[1])[0]
    
    def predict_next_activity(self, current: PlayerActivity) -> Optional[PlayerActivity]:
        """Predict likely next activity based on transitions."""
        current_name = current.value
        if current_name not in self.activity_transitions:
            return None
        transitions = self.activity_transitions[current_name]
        if not transitions:
            return None
        predicted = max(transitions.items(), key=lambda x: x[1])[0]
        return PlayerActivity(predicted)
    
    def infer_play_style(self):
        """Infer the player's overall play style."""
        if not self.activity_counts:
            return
        
        total = sum(self.activity_counts.values())
        if total < 10:
            return
        
        mining = self.activity_counts.get('mining', 0) / total
        building = self.activity_counts.get('building', 0) / total
        exploring = self.activity_counts.get('exploring', 0) / total
        fighting = self.activity_counts.get('fighting', 0) / total
        farming = self.activity_counts.get('farming', 0) / total
        
        styles = [
            ('miner', mining),
            ('builder', building),
            ('explorer', exploring),
            ('warrior', fighting),
            ('farmer', farming)
        ]
        self.play_style = max(styles, key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerProfile':
        # Convert tuples from lists
        if 'locations_frequented' in data:
            data['locations_frequented'] = [tuple(loc) for loc in data['locations_frequented']]
        return cls(**data)


class BehaviorClusterer:
    """Clusters similar behaviors using DBSCAN for pattern recognition."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.clusters: Dict[int, List[ObservedBehavior]] = {}
        self.pattern_cache: Dict[str, BehaviorPattern] = {}
    
    def _behavior_to_vector(self, behavior: ObservedBehavior) -> List[float]:
        """Convert behavior to feature vector for clustering."""
        # Activity encoding
        activity_encoding = list(PlayerActivity).index(behavior.activity)
        
        # Duration features
        duration = behavior.duration()
        
        # Tool usage (hash of tools used)
        tool_hash = hash(tuple(sorted(set(behavior.tools_used)))) % 100 / 100.0
        
        # Position change from sequence
        if len(behavior.sequence) >= 2:
            start_pos = behavior.sequence[0].position
            end_pos = behavior.sequence[-1].position
            distance = ((end_pos[0] - start_pos[0])**2 + 
                       (end_pos[1] - start_pos[1])**2 + 
                       (end_pos[2] - start_pos[2])**2)**0.5
        else:
            distance = 0
        
        return [
            activity_encoding / len(PlayerActivity),  # Normalized activity
            min(1.0, duration / 60.0),  # Duration capped at 60s
            tool_hash,
            min(1.0, distance / 50.0),  # Distance capped at 50 blocks
            behavior.confidence
        ]
    
    def cluster_behaviors(self, behaviors: List[ObservedBehavior]) -> Dict[int, List[ObservedBehavior]]:
        """Cluster behaviors into patterns."""
        if len(behaviors) < self.min_samples:
            return {}
        
        if not HAS_SKLEARN or not HAS_NUMPY:
            # Fallback: simple activity-based grouping
            clusters = defaultdict(list)
            for b in behaviors:
                clusters[b.activity.value].append(b)
            return {i: v for i, v in enumerate(clusters.values()) if len(v) >= self.min_samples}
        
        # Convert to vectors
        vectors = np.array([self._behavior_to_vector(b) for b in behaviors])
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(vectors)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # -1 is noise
                clusters[label].append(behaviors[idx])
        
        self.clusters = dict(clusters)
        return self.clusters
    
    def extract_patterns(self, behaviors: List[ObservedBehavior]) -> List[BehaviorPattern]:
        """Extract behavior patterns from clusters."""
        clusters = self.cluster_behaviors(behaviors)
        patterns = []
        
        for cluster_id, cluster_behaviors in clusters.items():
            if len(cluster_behaviors) < self.min_samples:
                continue
            
            # Analyze cluster
            activities = [b.activity for b in cluster_behaviors]
            durations = [b.duration() for b in cluster_behaviors]
            tools = [t for b in cluster_behaviors for t in b.tools_used]
            successes = [b.success for b in cluster_behaviors]
            
            # Find most common activity sequence
            activity_seq = [max(set(activities), key=activities.count)]
            
            pattern = BehaviorPattern(
                pattern_id=f"pattern_{cluster_id}_{hash(tuple(activity_seq)) % 10000}",
                activity_sequence=activity_seq,
                avg_duration=sum(durations) / len(durations),
                occurrences=len(cluster_behaviors),
                success_rate=sum(successes) / len(successes),
                context_requirements={},
                confidence=sum(b.confidence for b in cluster_behaviors) / len(cluster_behaviors),
                tools_typically_used=list(set(tools))[:5]
            )
            patterns.append(pattern)
            self.pattern_cache[pattern.pattern_id] = pattern
        
        return patterns


class PlayerObserver:
    """Enhanced player observation system with pattern recognition and memory decay.
    
    Optimizations for Linux:
    - Batch processing of observations
    - Efficient distance calculations using numpy
    - Memory decay for old observations
    - Pattern recognition with clustering
    """
    
    def __init__(self, config, ckpt_dir: str = "ckpt", resume: bool = False):
        self.config = config.observation
        self.ckpt_dir = ckpt_dir
        
        # Tracking state
        self.tracked_players: Dict[str, List[PlayerSnapshot]] = defaultdict(list)
        self.player_profiles: Dict[str, PlayerProfile] = {}
        self.observed_behaviors: List[ObservedBehavior] = []
        self.last_update_time: float = 0
        self.last_activities: Dict[str, PlayerActivity] = {}
        
        # Pattern recognition
        self.clusterer = BehaviorClusterer(eps=0.5, min_samples=3)
        self.known_patterns: List[BehaviorPattern] = []
        
        # Context tracking
        self.current_biome: str = "unknown"
        self.time_of_day: str = "day"
        
        # Performance optimization: batch processing queue
        self.observation_queue: List[Tuple[str, Dict]] = []
        self.batch_size = 10
        
        # Memory management
        self.max_snapshots_per_player = 100
        
        # Setup persistence
        U.f_mkdir(f"{ckpt_dir}/observation")
        
        if resume:
            self._load_state()
    
    def _load_state(self):
        """Load saved observation state."""
        try:
            profiles_path = f"{self.ckpt_dir}/observation/player_profiles.json"
            if U.f_exists(profiles_path):
                data = U.load_json(profiles_path)
                self.player_profiles = {
                    name: PlayerProfile.from_dict(p) 
                    for name, p in data.items()
                }
            
            behaviors_path = f"{self.ckpt_dir}/observation/observed_behaviors.json"
            if U.f_exists(behaviors_path):
                data = U.load_json(behaviors_path)
                self.observed_behaviors = [
                    ObservedBehavior.from_dict(b) for b in data
                ]
                # Apply memory decay and trim
                self._apply_memory_decay()
                self._trim_observations()
            
            patterns_path = f"{self.ckpt_dir}/observation/patterns.json"
            if U.f_exists(patterns_path):
                data = U.load_json(patterns_path)
                # Reconstruct patterns
                for p_data in data:
                    p_data['activity_sequence'] = [PlayerActivity(a) for a in p_data['activity_sequence']]
                    self.known_patterns.append(BehaviorPattern(**p_data))
                    
            print(f"\033[36mLoaded {len(self.player_profiles)} player profiles, "
                  f"{len(self.observed_behaviors)} behaviors, "
                  f"{len(self.known_patterns)} patterns\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load observation state: {e}\033[0m")
    
    def _apply_memory_decay(self):
        """Apply memory decay to all observations."""
        decay_rate = self.config.observation_decay_rate
        for behavior in self.observed_behaviors:
            behavior.apply_decay(decay_rate)
    
    def _trim_observations(self):
        """Trim observations to max memory, preferring high-relevance ones."""
        if len(self.observed_behaviors) > self.config.max_observation_memory:
            # Sort by relevance (confidence * decay)
            self.observed_behaviors.sort(key=lambda b: b.relevance_score(), reverse=True)
            self.observed_behaviors = self.observed_behaviors[:self.config.max_observation_memory]
    
    def save_state(self):
        """Save observation state to disk."""
        profiles_data = {name: p.to_dict() for name, p in self.player_profiles.items()}
        U.dump_json(profiles_data, f"{self.ckpt_dir}/observation/player_profiles.json")
        
        behaviors_data = [b.to_dict() for b in self.observed_behaviors]
        U.dump_json(behaviors_data, f"{self.ckpt_dir}/observation/observed_behaviors.json")
        
        patterns_data = []
        for p in self.known_patterns:
            p_dict = asdict(p)
            p_dict['activity_sequence'] = [a.value for a in p.activity_sequence]
            patterns_data.append(p_dict)
        U.dump_json(patterns_data, f"{self.ckpt_dir}/observation/patterns.json")
    
    def process_events(self, events: List) -> Dict[str, Any]:
        """Process game events with batch optimization.
        
        Args:
            events: List of game events from the environment
            
        Returns:
            Dictionary containing observation results
        """
        current_time = time.time()
        
        # Check if enough time has passed for an update
        if current_time - self.last_update_time < self.config.update_frequency:
            return {"updated": False}
        
        self.last_update_time = current_time
        
        results = {
            "updated": True,
            "players_detected": [],
            "new_behaviors": [],
            "activity_updates": {},
            "patterns_recognized": []
        }
        
        # Extract context from events
        self._update_context(events)
        
        # Batch collect player data
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                event_data = event[1]
                
                if "nearbyPlayers" in event_data:
                    for player in event_data["nearbyPlayers"]:
                        self.observation_queue.append((current_time, player))
                
                if "status" in event_data and "nearbyPlayers" in event_data.get("status", {}):
                    for player in event_data["status"]["nearbyPlayers"]:
                        self.observation_queue.append((current_time, player))
        
        # Process batch
        if len(self.observation_queue) >= self.batch_size or current_time - self.last_update_time > 5.0:
            self._process_observation_batch(results)
        
        # Periodic pattern analysis
        if len(self.observed_behaviors) >= 20 and random.random() < 0.1:
            self._update_patterns(results)
        
        # Apply memory decay periodically
        if random.random() < 0.05:
            self._apply_memory_decay()
        
        return results
    
    def _update_context(self, events: List):
        """Update context from events."""
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                if "biome" in event[1]:
                    self.current_biome = event[1]["biome"]
                status = event[1].get("status", {})
                if "timeOfDay" in status:
                    self.time_of_day = "night" if "night" in status["timeOfDay"].lower() else "day"
    
    def _process_observation_batch(self, results: Dict):
        """Process queued observations in batch."""
        current_time = time.time()
        
        for timestamp, player_data in self.observation_queue:
            if not isinstance(player_data, dict):
                continue
                
            name = player_data.get("name") or player_data.get("username")
            if not name:
                continue
            
            # Don't track more than max players
            if (name not in self.tracked_players and 
                len(self.tracked_players) >= self.config.max_tracked_players):
                continue
            
            # Create snapshot
            position = player_data.get("position", {})
            velocity = player_data.get("velocity", {})
            
            snapshot = PlayerSnapshot(
                timestamp=timestamp,
                position=(
                    position.get("x", 0),
                    position.get("y", 0),
                    position.get("z", 0)
                ),
                velocity=(
                    velocity.get("x", 0) if isinstance(velocity, dict) else 0,
                    velocity.get("y", 0) if isinstance(velocity, dict) else 0,
                    velocity.get("z", 0) if isinstance(velocity, dict) else 0
                ),
                yaw=player_data.get("yaw", 0),
                pitch=player_data.get("pitch", 0),
                held_item=player_data.get("heldItem"),
                health=player_data.get("health", 20.0),
                is_sneaking=player_data.get("isSneaking", False),
                is_sprinting=player_data.get("isSprinting", False),
                is_on_ground=player_data.get("onGround", True),
                nearby_blocks=player_data.get("nearbyBlocks", []),
                nearby_entities=player_data.get("nearbyEntities", [])
            )
            
            # Track player
            self.tracked_players[name].append(snapshot)
            
            # Trim old snapshots (memory management)
            if len(self.tracked_players[name]) > self.max_snapshots_per_player:
                self.tracked_players[name] = self.tracked_players[name][-self.max_snapshots_per_player:]
            
            # Update or create profile
            if name not in self.player_profiles:
                self.player_profiles[name] = PlayerProfile(
                    name=name,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
            else:
                self.player_profiles[name].last_seen = timestamp
                self.player_profiles[name].total_observation_time += self.config.update_frequency
            
            # Classify current activity with previous context
            previous_activity = self.last_activities.get(name)
            activity = self._classify_activity_enhanced(name, snapshot)
            self.player_profiles[name].update_activity(activity, previous_activity)
            self.last_activities[name] = activity
            
            results["players_detected"].append(name)
            results["activity_updates"][name] = activity.value
            
            # Track tool usage
            if snapshot.held_item:
                profile = self.player_profiles[name]
                profile.preferred_tools[snapshot.held_item] = \
                    profile.preferred_tools.get(snapshot.held_item, 0) + 1
            
            # Update location history
            if len(self.player_profiles[name].locations_frequented) < 100:
                self.player_profiles[name].locations_frequented.append(snapshot.position)
        
        # Clear queue
        self.observation_queue = []
        
        # Analyze for completed behaviors
        self._analyze_activity_buffers_enhanced(current_time, results)
    
    @lru_cache(maxsize=256)
    def _calc_distance(self, pos1: Tuple[float, float, float], 
                       pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance with caching."""
        if HAS_NUMPY:
            return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
        return ((pos1[0] - pos2[0])**2 + 
                (pos1[1] - pos2[1])**2 + 
                (pos1[2] - pos2[2])**2)**0.5
    
    def _classify_activity_enhanced(self, player_name: str, 
                                    current_snapshot: PlayerSnapshot) -> PlayerActivity:
        """Enhanced activity classification with better heuristics."""
        snapshots = self.tracked_players.get(player_name, [])
        
        if len(snapshots) < 2:
            return PlayerActivity.IDLE
        
        recent_snapshots = snapshots[-10:]  # More context
        
        # Calculate movement metrics
        total_distance = 0
        y_changes = []
        
        for i in range(1, len(recent_snapshots)):
            prev = recent_snapshots[i-1]
            curr = recent_snapshots[i]
            total_distance += self._calc_distance(prev.position, curr.position)
            y_changes.append(curr.position[1] - prev.position[1])
        
        avg_speed = total_distance / max(len(recent_snapshots), 1)
        avg_y_change = sum(y_changes) / len(y_changes) if y_changes else 0
        
        # Check states first
        if current_snapshot.is_sneaking:
            return PlayerActivity.SNEAKING
        
        if current_snapshot.is_sprinting or avg_speed > 6.0:
            return PlayerActivity.RUNNING
        
        # Check held item for activity hints
        held = (current_snapshot.held_item or "").lower()
        
        # Mining indicators
        if "pickaxe" in held:
            if avg_speed < 0.5 and avg_y_change < -0.1:
                return PlayerActivity.MINING
        
        # Combat indicators
        if any(weapon in held for weapon in ["sword", "bow", "crossbow", "axe"]) and "pickaxe" not in held:
            # Check for nearby hostile entities
            if current_snapshot.nearby_entities:
                return PlayerActivity.FIGHTING
        
        # Farming indicators
        if "hoe" in held:
            return PlayerActivity.FARMING
        
        # Building indicators
        building_blocks = ["_planks", "_log", "cobblestone", "stone", "brick", "concrete", "glass"]
        if any(block in held for block in building_blocks):
            if avg_speed < 1.0:
                return PlayerActivity.BUILDING
        
        # Swimming (underwater or in water with low y)
        if current_snapshot.position[1] < 62 and not current_snapshot.is_on_ground:
            return PlayerActivity.SWIMMING
        
        # Climbing (going up without ground)
        if avg_y_change > 0.3 and not current_snapshot.is_on_ground:
            return PlayerActivity.CLIMBING
        
        # Jumping
        if not current_snapshot.is_on_ground and avg_y_change > 0.2:
            return PlayerActivity.JUMPING
        
        # Gathering (with axe but not pickaxe)
        if "axe" in held and "pickaxe" not in held:
            return PlayerActivity.GATHERING
        
        # Movement-based classification
        if avg_speed > 3.0:
            if total_distance > 20:
                return PlayerActivity.EXPLORING
            return PlayerActivity.WALKING
        
        if avg_speed > 0.5:
            return PlayerActivity.WALKING
        
        return PlayerActivity.IDLE
    
    def _analyze_activity_buffers_enhanced(self, current_time: float, results: Dict):
        """Enhanced analysis with pattern matching."""
        window = self.config.activity_detection_window
        
        for player_name, snapshots in list(self.tracked_players.items()):
            if len(snapshots) < 5:
                continue
            
            recent = [s for s in snapshots if s.timestamp > current_time - window]
            if len(recent) < 3:
                continue
            
            activities = [self._classify_activity_enhanced(player_name, s) for s in recent]
            
            # Find dominant activity
            activity_counts = defaultdict(int)
            for a in activities:
                activity_counts[a] += 1
            
            if not activity_counts:
                continue
                
            dominant_activity, count = max(activity_counts.items(), key=lambda x: x[1])
            confidence = count / len(activities)
            
            if (confidence >= self.config.min_observation_confidence and 
                dominant_activity != PlayerActivity.IDLE):
                
                # Match to known pattern
                pattern_id = self._match_to_pattern(dominant_activity, recent)
                
                behavior = ObservedBehavior(
                    player_name=player_name,
                    activity=dominant_activity,
                    start_time=recent[0].timestamp,
                    end_time=recent[-1].timestamp,
                    success=True,
                    context={
                        "biome": self.current_biome,
                        "time_of_day": self.time_of_day,
                        "player_health": recent[-1].health
                    },
                    tools_used=list(set(s.held_item for s in recent if s.held_item)),
                    blocks_involved=list(set(b for s in recent for b in s.nearby_blocks))[:10],
                    items_gained=[],
                    items_lost=[],
                    sequence=recent,
                    confidence=confidence,
                    pattern_id=pattern_id
                )
                
                if not self._is_duplicate_behavior(behavior):
                    self.observed_behaviors.append(behavior)
                    results["new_behaviors"].append(behavior.to_dict())
                    
                    if pattern_id:
                        results["patterns_recognized"].append(pattern_id)
                    
                    self._trim_observations()
    
    def _match_to_pattern(self, activity: PlayerActivity, 
                          snapshots: List[PlayerSnapshot]) -> Optional[str]:
        """Try to match behavior to a known pattern."""
        for pattern in self.known_patterns:
            if activity in pattern.activity_sequence:
                return pattern.pattern_id
        return None
    
    def _update_patterns(self, results: Dict):
        """Update recognized patterns from recent observations."""
        recent_behaviors = [b for b in self.observed_behaviors 
                          if b.relevance_score() > 0.5]
        
        if len(recent_behaviors) < 10:
            return
        
        new_patterns = self.clusterer.extract_patterns(recent_behaviors)
        
        # Merge with existing patterns
        for new_pattern in new_patterns:
            existing = next((p for p in self.known_patterns 
                           if p.similarity(new_pattern) > 0.8), None)
            if existing:
                # Update existing
                existing.occurrences += new_pattern.occurrences
                existing.avg_duration = (existing.avg_duration + new_pattern.avg_duration) / 2
            else:
                self.known_patterns.append(new_pattern)
        
        # Trim old patterns
        self.known_patterns = sorted(self.known_patterns, 
                                     key=lambda p: p.occurrences, 
                                     reverse=True)[:50]
    
    def _is_duplicate_behavior(self, new_behavior: ObservedBehavior) -> bool:
        """Check if this behavior is a duplicate of a recent one."""
        for existing in self.observed_behaviors[-10:]:
            if (existing.player_name == new_behavior.player_name and
                existing.activity == new_behavior.activity and
                abs(existing.end_time - new_behavior.start_time) < 2.0):
                return True
        return False
    
    def get_nearby_players(self) -> List[str]:
        """Get list of currently tracked players."""
        current_time = time.time()
        active_players = []
        
        for name, snapshots in self.tracked_players.items():
            if snapshots and current_time - snapshots[-1].timestamp < 30.0:
                active_players.append(name)
        
        return active_players
    
    def get_player_activity(self, player_name: str) -> Optional[PlayerActivity]:
        """Get the current activity of a tracked player."""
        if player_name not in self.tracked_players:
            return None
        
        snapshots = self.tracked_players[player_name]
        if not snapshots:
            return None
        
        return self._classify_activity_enhanced(player_name, snapshots[-1])
    
    def get_player_profile(self, player_name: str) -> Optional[PlayerProfile]:
        """Get the profile of an observed player."""
        profile = self.player_profiles.get(player_name)
        if profile:
            profile.infer_play_style()
        return profile
    
    def get_behaviors_by_activity(self, activity: PlayerActivity, 
                                   min_relevance: float = 0.3) -> List[ObservedBehavior]:
        """Get observed behaviors of a specific activity type with relevance filtering."""
        return [b for b in self.observed_behaviors 
                if b.activity == activity and b.relevance_score() >= min_relevance]
    
    def get_most_common_strategies(self, activity: PlayerActivity, top_k: int = 5) -> List[Dict]:
        """Get the most commonly observed strategies for an activity."""
        behaviors = self.get_behaviors_by_activity(activity)
        
        if not behaviors:
            return []
        
        # Group by tools used
        tool_strategies = defaultdict(list)
        for b in behaviors:
            tools_key = tuple(sorted(set(b.tools_used)))
            tool_strategies[tools_key].append(b)
        
        # Sort by weighted frequency (count * avg relevance)
        sorted_strategies = sorted(
            tool_strategies.items(), 
            key=lambda x: len(x[1]) * sum(b.relevance_score() for b in x[1]) / len(x[1]), 
            reverse=True
        )[:top_k]
        
        return [
            {
                "tools": list(tools),
                "count": len(behaviors),
                "avg_confidence": sum(b.confidence for b in behaviors) / len(behaviors),
                "avg_duration": sum(b.duration() for b in behaviors) / len(behaviors),
                "avg_relevance": sum(b.relevance_score() for b in behaviors) / len(behaviors)
            }
            for tools, behaviors in sorted_strategies
        ]
    
    def predict_player_action(self, player_name: str) -> Optional[Dict]:
        """Predict what a player is likely to do next."""
        profile = self.player_profiles.get(player_name)
        if not profile:
            return None
        
        current_activity = self.get_player_activity(player_name)
        if not current_activity:
            return None
        
        predicted = profile.predict_next_activity(current_activity)
        if not predicted:
            return None
        
        return {
            "current_activity": current_activity.value,
            "predicted_next": predicted.value,
            "play_style": profile.play_style,
            "confidence": profile.predictability
        }
    
    def get_observation_summary(self) -> Dict:
        """Get a summary of all observations."""
        activity_counts = defaultdict(int)
        for b in self.observed_behaviors:
            activity_counts[b.activity.value] += 1
        
        return {
            "total_players_observed": len(self.player_profiles),
            "active_players": len(self.get_nearby_players()),
            "total_behaviors_recorded": len(self.observed_behaviors),
            "activity_distribution": dict(activity_counts),
            "known_patterns": len(self.known_patterns),
            "avg_observation_relevance": sum(b.relevance_score() for b in self.observed_behaviors) / max(len(self.observed_behaviors), 1),
            "player_profiles": {
                name: {
                    "primary_activity": p.get_primary_activity(),
                    "play_style": p.play_style,
                    "total_time": p.total_observation_time,
                    "success_rate": p.success_rate
                }
                for name, p in self.player_profiles.items()
            }
        }
