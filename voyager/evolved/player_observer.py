"""Player Observation System for Voyager Evolved.

This module provides functionality to detect, track, and observe other players
on the Minecraft server, extracting behavioral patterns for learning.
"""

import time
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import voyager.utils as U


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
    nearby_blocks: List[str] = field(default_factory=list)
    

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
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['activity'] = self.activity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObservedBehavior':
        data['activity'] = PlayerActivity(data['activity'])
        data['sequence'] = [PlayerSnapshot(**s) for s in data['sequence']]
        return cls(**data)


@dataclass
class PlayerProfile:
    """Profile of an observed player built over time."""
    name: str
    first_seen: float
    last_seen: float
    total_observation_time: float = 0.0
    activity_counts: Dict[str, int] = field(default_factory=dict)
    preferred_tools: Dict[str, int] = field(default_factory=dict)
    skill_demonstrations: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.5
    behavior_patterns: List[str] = field(default_factory=list)
    
    def update_activity(self, activity: PlayerActivity):
        activity_name = activity.value
        self.activity_counts[activity_name] = self.activity_counts.get(activity_name, 0) + 1
    
    def get_primary_activity(self) -> Optional[str]:
        if not self.activity_counts:
            return None
        return max(self.activity_counts.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerProfile':
        return cls(**data)


class PlayerObserver:
    """Observes and tracks other players on the server.
    
    This system detects nearby players, classifies their activities,
    extracts behavioral patterns, and stores observations for learning.
    """
    
    def __init__(self, config, ckpt_dir: str = "ckpt", resume: bool = False):
        self.config = config.observation
        self.ckpt_dir = ckpt_dir
        
        # Tracking state
        self.tracked_players: Dict[str, List[PlayerSnapshot]] = defaultdict(list)
        self.player_profiles: Dict[str, PlayerProfile] = {}
        self.observed_behaviors: List[ObservedBehavior] = []
        self.last_update_time: float = 0
        
        # Activity classification state
        self.activity_buffer: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
        
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
                # Trim to max memory
                if len(self.observed_behaviors) > self.config.max_observation_memory:
                    self.observed_behaviors = self.observed_behaviors[-self.config.max_observation_memory:]
                    
            print(f"\033[36mLoaded {len(self.player_profiles)} player profiles and "
                  f"{len(self.observed_behaviors)} observed behaviors\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load observation state: {e}\033[0m")
    
    def save_state(self):
        """Save observation state to disk."""
        profiles_data = {name: p.to_dict() for name, p in self.player_profiles.items()}
        U.dump_json(profiles_data, f"{self.ckpt_dir}/observation/player_profiles.json")
        
        behaviors_data = [b.to_dict() for b in self.observed_behaviors]
        U.dump_json(behaviors_data, f"{self.ckpt_dir}/observation/observed_behaviors.json")
    
    def process_events(self, events: List) -> Dict[str, Any]:
        """Process game events to detect and observe players.
        
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
            "activity_updates": {}
        }
        
        # Extract player information from events
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                event_data = event[1]
                
                # Look for nearby players in various event fields
                if "nearbyPlayers" in event_data:
                    self._process_nearby_players(event_data["nearbyPlayers"], current_time, results)
                
                if "status" in event_data and "nearbyPlayers" in event_data.get("status", {}):
                    self._process_nearby_players(
                        event_data["status"]["nearbyPlayers"], current_time, results
                    )
        
        # Analyze activity buffers for completed behaviors
        self._analyze_activity_buffers(current_time, results)
        
        return results
    
    def _process_nearby_players(self, players: List[Dict], current_time: float, results: Dict):
        """Process information about nearby players."""
        for player_data in players:
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
            snapshot = PlayerSnapshot(
                timestamp=current_time,
                position=(
                    position.get("x", 0),
                    position.get("y", 0),
                    position.get("z", 0)
                ),
                velocity=tuple(player_data.get("velocity", {}).values()) if isinstance(player_data.get("velocity"), dict) else (0, 0, 0),
                yaw=player_data.get("yaw", 0),
                pitch=player_data.get("pitch", 0),
                held_item=player_data.get("heldItem"),
                health=player_data.get("health", 20.0),
                is_sneaking=player_data.get("isSneaking", False),
                is_sprinting=player_data.get("isSprinting", False),
                nearby_blocks=player_data.get("nearbyBlocks", [])
            )
            
            # Track player
            self.tracked_players[name].append(snapshot)
            
            # Trim old snapshots
            cutoff_time = current_time - self.config.activity_detection_window * 2
            self.tracked_players[name] = [
                s for s in self.tracked_players[name] 
                if s.timestamp > cutoff_time
            ]
            
            # Update or create profile
            if name not in self.player_profiles:
                self.player_profiles[name] = PlayerProfile(
                    name=name,
                    first_seen=current_time,
                    last_seen=current_time
                )
            else:
                self.player_profiles[name].last_seen = current_time
                self.player_profiles[name].total_observation_time += self.config.update_frequency
            
            # Classify current activity
            activity = self._classify_activity(name, snapshot)
            self.player_profiles[name].update_activity(activity)
            
            results["players_detected"].append(name)
            results["activity_updates"][name] = activity.value
            
            # Track tool usage
            if snapshot.held_item:
                profile = self.player_profiles[name]
                profile.preferred_tools[snapshot.held_item] = \
                    profile.preferred_tools.get(snapshot.held_item, 0) + 1
    
    def _classify_activity(self, player_name: str, current_snapshot: PlayerSnapshot) -> PlayerActivity:
        """Classify what activity a player is currently doing."""
        snapshots = self.tracked_players.get(player_name, [])
        
        if len(snapshots) < 2:
            return PlayerActivity.IDLE
        
        recent_snapshots = snapshots[-5:]  # Last 5 snapshots
        
        # Calculate movement
        total_distance = 0
        for i in range(1, len(recent_snapshots)):
            prev = recent_snapshots[i-1]
            curr = recent_snapshots[i]
            distance = ((curr.position[0] - prev.position[0])**2 + 
                       (curr.position[1] - prev.position[1])**2 + 
                       (curr.position[2] - prev.position[2])**2)**0.5
            total_distance += distance
        
        avg_speed = total_distance / len(recent_snapshots) if recent_snapshots else 0
        
        # Check for sprinting/running
        if current_snapshot.is_sprinting or avg_speed > 5.0:
            return PlayerActivity.RUNNING
        
        # Check held item for activity hints
        held = current_snapshot.held_item or ""
        
        if "pickaxe" in held.lower():
            if avg_speed < 0.5:  # Stationary with pickaxe = mining
                return PlayerActivity.MINING
        elif "sword" in held.lower() or "bow" in held.lower():
            return PlayerActivity.FIGHTING
        elif "hoe" in held.lower():
            return PlayerActivity.FARMING
        elif "axe" in held.lower() and "pickaxe" not in held.lower():
            return PlayerActivity.GATHERING
        
        # Check for building (placing blocks)
        if any(block in held.lower() for block in ["_planks", "_log", "cobblestone", "stone", "brick"]):
            if avg_speed < 1.0:
                return PlayerActivity.BUILDING
        
        # Check Y-axis changes (swimming/mining vertical)
        y_changes = [recent_snapshots[i].position[1] - recent_snapshots[i-1].position[1] 
                     for i in range(1, len(recent_snapshots))]
        avg_y_change = sum(y_changes) / len(y_changes) if y_changes else 0
        
        if avg_y_change < -0.5:  # Going down
            return PlayerActivity.MINING
        
        # Check for general movement
        if avg_speed > 0.5:
            if total_distance > 10:
                return PlayerActivity.EXPLORING
            return PlayerActivity.WALKING
        
        return PlayerActivity.IDLE
    
    def _analyze_activity_buffers(self, current_time: float, results: Dict):
        """Analyze buffered activities to detect completed behavior patterns."""
        window = self.config.activity_detection_window
        
        for player_name, snapshots in list(self.tracked_players.items()):
            if len(snapshots) < 5:
                continue
            
            # Check if we have a consistent activity over the window
            recent = [s for s in snapshots if s.timestamp > current_time - window]
            if len(recent) < 3:
                continue
            
            activities = [self._classify_activity(player_name, s) for s in recent]
            
            # Find dominant activity
            activity_counts = defaultdict(int)
            for a in activities:
                activity_counts[a] += 1
            
            if not activity_counts:
                continue
                
            dominant_activity, count = max(activity_counts.items(), key=lambda x: x[1])
            confidence = count / len(activities)
            
            # Record if confidence is high enough and not idle
            if (confidence >= self.config.min_observation_confidence and 
                dominant_activity != PlayerActivity.IDLE):
                
                behavior = ObservedBehavior(
                    player_name=player_name,
                    activity=dominant_activity,
                    start_time=recent[0].timestamp,
                    end_time=recent[-1].timestamp,
                    success=True,  # Assume success if completed
                    context={
                        "biome": "unknown",
                        "time_of_day": "unknown",
                        "player_health": recent[-1].health
                    },
                    tools_used=[s.held_item for s in recent if s.held_item],
                    blocks_involved=[b for s in recent for b in s.nearby_blocks],
                    items_gained=[],
                    items_lost=[],
                    sequence=recent,
                    confidence=confidence
                )
                
                # Avoid duplicates
                if not self._is_duplicate_behavior(behavior):
                    self.observed_behaviors.append(behavior)
                    results["new_behaviors"].append(behavior.to_dict())
                    
                    # Trim memory if needed
                    if len(self.observed_behaviors) > self.config.max_observation_memory:
                        self.observed_behaviors = self.observed_behaviors[-self.config.max_observation_memory:]
    
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
        
        return self._classify_activity(player_name, snapshots[-1])
    
    def get_player_profile(self, player_name: str) -> Optional[PlayerProfile]:
        """Get the profile of an observed player."""
        return self.player_profiles.get(player_name)
    
    def get_behaviors_by_activity(self, activity: PlayerActivity) -> List[ObservedBehavior]:
        """Get all observed behaviors of a specific activity type."""
        return [b for b in self.observed_behaviors if b.activity == activity]
    
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
        
        # Sort by frequency
        sorted_strategies = sorted(
            tool_strategies.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:top_k]
        
        return [
            {
                "tools": list(tools),
                "count": len(behaviors),
                "avg_confidence": sum(b.confidence for b in behaviors) / len(behaviors),
                "avg_duration": sum(b.duration() for b in behaviors) / len(behaviors)
            }
            for tools, behaviors in sorted_strategies
        ]
    
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
            "player_profiles": {
                name: {
                    "primary_activity": p.get_primary_activity(),
                    "total_time": p.total_observation_time,
                    "success_rate": p.success_rate
                }
                for name, p in self.player_profiles.items()
            }
        }
