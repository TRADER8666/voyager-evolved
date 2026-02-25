"""Human-like Behavior System for Voyager Evolved.

This module adds natural, human-like behavior patterns to the agent's actions,
including movement variations, pauses, mistakes, and natural camera movements.
"""

import random
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MovementModifier:
    """Modifier for a single movement action."""
    deviation_x: float = 0.0
    deviation_z: float = 0.0
    speed_multiplier: float = 1.0
    pause_before: float = 0.0
    pause_after: float = 0.0
    look_around: bool = False


class HumanBehaviorSystem:
    """Adds human-like behavior patterns to agent actions.
    
    Features:
    - Path deviation (not perfectly optimal)
    - Occasional pauses for "thinking"
    - Minor mistakes and corrections
    - Natural camera/head movements
    - Varied action pacing
    """
    
    def __init__(self, config, personality_engine=None):
        self.config = config.human_behavior
        self.personality = personality_engine
        
        # State tracking
        self.last_action_time = time.time()
        self.actions_since_break = 0
        self.current_focus_point: Optional[Tuple[float, float, float]] = None
        self.interesting_things: List[Dict] = []
        self.mistake_history: List[float] = []
        
        # Movement state
        self.movement_momentum = 0.0
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        
    def process_movement(self, target_position: Tuple[float, float, float],
                        current_position: Tuple[float, float, float]) -> MovementModifier:
        """Process a movement action to add human-like variations.
        
        Args:
            target_position: Where the agent wants to go
            current_position: Current agent position
            
        Returns:
            MovementModifier with adjustments to apply
        """
        modifier = MovementModifier()
        
        # Maybe deviate from optimal path
        if random.random() < self.config.path_deviation_chance:
            deviation_amount = self.config.path_deviation_amount
            modifier.deviation_x = random.uniform(-deviation_amount, deviation_amount)
            modifier.deviation_z = random.uniform(-deviation_amount, deviation_amount)
        
        # Speed variation based on personality/energy
        base_speed = 1.0
        if self.personality:
            base_speed = 1.0 / self.personality.get_action_timing_modifier()
        
        # Add random speed variation
        speed_var = self.config.action_speed_variation
        modifier.speed_multiplier = base_speed * (1.0 + random.uniform(-speed_var, speed_var))
        
        # Maybe pause before moving (thinking)
        if random.random() < self.config.thinking_pause_chance:
            min_pause, max_pause = self.config.thinking_pause_duration
            modifier.pause_before = random.uniform(min_pause, max_pause)
        
        # Maybe look around during movement
        if random.random() < self.config.look_around_frequency:
            modifier.look_around = True
        
        return modifier
    
    def should_make_mistake(self) -> Tuple[bool, str]:
        """Determine if the agent should make a minor mistake.
        
        Returns:
            (should_mistake, mistake_type)
        """
        if random.random() >= self.config.mistake_chance:
            return False, ""
        
        # Don't make too many mistakes in a row
        current_time = time.time()
        recent_mistakes = [t for t in self.mistake_history if current_time - t < 60]
        if len(recent_mistakes) >= 3:
            return False, ""
        
        mistake_types = [
            "wrong_direction",  # Move wrong way briefly
            "misclick",  # Click wrong spot
            "overshoot",  # Go past target
            "wrong_tool",  # Use wrong tool momentarily
            "hesitation",  # Start and stop action
        ]
        
        mistake_type = random.choice(mistake_types)
        self.mistake_history.append(current_time)
        
        return True, mistake_type
    
    def should_recover_from_mistake(self) -> bool:
        """Determine if agent should correct a mistake."""
        return random.random() < self.config.recovery_from_mistake
    
    def get_look_around_movement(self) -> Dict[str, float]:
        """Generate natural looking-around head movement."""
        # Smooth random movement
        target_yaw = self.head_yaw + random.gauss(0, 30)
        target_pitch = random.gauss(0, 15)  # Mostly look horizontally
        
        # Smooth interpolation
        smoothness = self.config.head_turn_smoothness
        self.head_yaw += (target_yaw - self.head_yaw) * smoothness
        self.head_pitch += (target_pitch - self.head_pitch) * smoothness
        
        # Clamp values
        self.head_pitch = max(-90, min(90, self.head_pitch))
        
        return {
            "yaw": self.head_yaw,
            "pitch": self.head_pitch
        }
    
    def get_focus_on_entity_movement(self, entity_position: Tuple[float, float, float],
                                     agent_position: Tuple[float, float, float]) -> Dict[str, float]:
        """Generate head movement to look at an entity/point of interest."""
        dx = entity_position[0] - agent_position[0]
        dy = entity_position[1] - agent_position[1]
        dz = entity_position[2] - agent_position[2]
        
        # Calculate target yaw and pitch
        target_yaw = math.degrees(math.atan2(-dx, dz))
        horizontal_dist = math.sqrt(dx*dx + dz*dz)
        target_pitch = math.degrees(math.atan2(dy, horizontal_dist))
        
        # Smooth interpolation
        smoothness = self.config.head_turn_smoothness
        self.head_yaw += (target_yaw - self.head_yaw) * smoothness
        self.head_pitch += (target_pitch - self.head_pitch) * smoothness
        
        return {
            "yaw": self.head_yaw,
            "pitch": self.head_pitch
        }
    
    def should_take_break(self) -> Tuple[bool, float]:
        """Determine if agent should take a break between tasks.
        
        Returns:
            (should_break, break_duration)
        """
        if not self.config.break_between_tasks:
            return False, 0.0
        
        self.actions_since_break += 1
        
        # Exponentially increasing chance of break
        break_chance = min(0.5, self.actions_since_break * 0.05)
        
        if random.random() < break_chance:
            self.actions_since_break = 0
            min_break, max_break = self.config.break_duration
            duration = random.uniform(min_break, max_break)
            return True, duration
        
        return False, 0.0
    
    def update_interesting_things(self, events: List) -> List[Dict]:
        """Update list of interesting things to look at."""
        if not self.config.look_at_interesting_things:
            return []
        
        interesting = []
        
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                # Nearby entities are interesting
                entities = event[1].get("nearbyEntities", [])
                for entity in entities:
                    if isinstance(entity, dict) and "position" in entity:
                        interesting.append({
                            "type": "entity",
                            "name": entity.get("name", "unknown"),
                            "position": tuple(entity["position"].values()) if isinstance(entity["position"], dict) else entity["position"],
                            "interest_level": self._calculate_interest_level(entity)
                        })
                
                # Nearby players are very interesting
                players = event[1].get("nearbyPlayers", [])
                for player in players:
                    if isinstance(player, dict) and "position" in player:
                        interesting.append({
                            "type": "player",
                            "name": player.get("name", "unknown"),
                            "position": tuple(player["position"].values()) if isinstance(player["position"], dict) else player["position"],
                            "interest_level": 1.0  # Players are always interesting
                        })
        
        # Sort by interest level
        interesting.sort(key=lambda x: x["interest_level"], reverse=True)
        self.interesting_things = interesting[:5]  # Keep top 5
        
        return self.interesting_things
    
    def _calculate_interest_level(self, entity: Dict) -> float:
        """Calculate how interesting an entity is."""
        name = entity.get("name", "").lower()
        
        # Hostile mobs are interesting (threat)
        if any(mob in name for mob in ["zombie", "skeleton", "creeper", "spider", "enderman"]):
            return 0.9
        
        # Animals are moderately interesting
        if any(animal in name for animal in ["cow", "pig", "sheep", "chicken", "horse"]):
            return 0.5
        
        # Rare mobs are very interesting
        if any(rare in name for rare in ["villager", "iron_golem", "wolf", "cat"]):
            return 0.8
        
        return 0.3
    
    def should_look_at_interesting_thing(self) -> Optional[Dict]:
        """Decide if agent should look at something interesting."""
        if not self.interesting_things:
            return None
        
        if random.random() < self.config.look_around_frequency:
            # Weighted selection by interest level
            weights = [t["interest_level"] for t in self.interesting_things]
            selected = random.choices(self.interesting_things, weights=weights, k=1)[0]
            return selected
        
        return None
    
    def generate_natural_action_code(self, base_code: str, action_type: str) -> str:
        """Wrap action code with human-like behaviors.
        
        Args:
            base_code: The original action code
            action_type: Type of action (move, mine, build, etc.)
            
        Returns:
            Modified code with human-like behaviors
        """
        modifications = []
        
        # Add thinking pause
        if random.random() < self.config.thinking_pause_chance:
            min_pause, max_pause = self.config.thinking_pause_duration
            pause = random.uniform(min_pause, max_pause)
            modifications.append(f"await bot.waitForTicks({int(pause * 20)});  // Thinking...")
        
        # Add hesitation for some actions
        if random.random() < self.config.decision_hesitation:
            modifications.append("await bot.waitForTicks(10);  // Slight hesitation")
        
        # Add occasional looking around
        if random.random() < self.config.look_around_frequency:
            look_code = self._generate_look_around_code()
            modifications.append(look_code)
        
        # Combine modifications with base code
        if modifications:
            pre_code = "\n".join(modifications)
            return f"{pre_code}\n{base_code}"
        
        return base_code
    
    def _generate_look_around_code(self) -> str:
        """Generate JavaScript code for looking around naturally."""
        yaw_change = random.uniform(-45, 45)
        pitch_change = random.uniform(-20, 20)
        
        return f"""
// Natural look around
await bot.look(bot.entity.yaw + {yaw_change:.2f} * Math.PI / 180, bot.entity.pitch + {pitch_change:.2f} * Math.PI / 180, false);
await bot.waitForTicks(5);
"""
    
    def add_movement_variation(self, movement_code: str) -> str:
        """Add subtle variations to movement code."""
        # Add path deviation
        if random.random() < self.config.path_deviation_chance:
            deviation = self.config.path_deviation_amount
            dx = random.uniform(-deviation, deviation)
            dz = random.uniform(-deviation, deviation)
            
            # Try to inject deviation into movement
            deviation_code = f"""
// Slight path variation
const deviation = {{ x: {dx:.2f}, z: {dz:.2f} }};
"""
            movement_code = deviation_code + movement_code
        
        return movement_code
    
    def simulate_suboptimal_choice(self, options: List[str], optimal_index: int) -> int:
        """Potentially choose a suboptimal but valid option.
        
        Args:
            options: List of valid options
            optimal_index: Index of the optimal choice
            
        Returns:
            Chosen index (may be different from optimal)
        """
        if len(options) <= 1:
            return 0
        
        if random.random() < self.config.suboptimal_choice_chance:
            # Choose a different valid option
            other_indices = [i for i in range(len(options)) if i != optimal_index]
            return random.choice(other_indices)
        
        return optimal_index
    
    def get_behavior_summary(self) -> Dict:
        """Get summary of recent human-like behaviors."""
        return {
            "actions_since_break": self.actions_since_break,
            "recent_mistakes": len([t for t in self.mistake_history if time.time() - t < 300]),
            "interesting_things_nearby": len(self.interesting_things),
            "current_head_position": {"yaw": self.head_yaw, "pitch": self.head_pitch}
        }
