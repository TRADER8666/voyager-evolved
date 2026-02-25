"""Enhanced Human-like Behavior System for Voyager Evolved (Linux Optimized).

This module adds natural, human-like behavior patterns including:
- Fatigue system (slower actions when "tired")
- Attention span (focus on one thing, then switch)
- Emotional responses to events
- Better idle behaviors (looking around, small movements)
- Learning curve (improve at tasks over time)
"""

import random
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict


class EmotionalState(Enum):
    """Emotional states that affect behavior."""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CAUTIOUS = "cautious"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    SATISFIED = "satisfied"
    ANXIOUS = "anxious"
    BORED = "bored"


@dataclass
class FatigueState:
    """Tracks fatigue level and its effects."""
    physical_fatigue: float = 0.0  # 0 (rested) to 1 (exhausted)
    mental_fatigue: float = 0.0
    last_rest_time: float = field(default_factory=time.time)
    active_time: float = 0.0
    
    def update(self, activity_level: float, time_delta: float):
        """Update fatigue based on activity."""
        # Increase fatigue with activity
        physical_increase = activity_level * 0.01 * (time_delta / 60)
        mental_increase = 0.005 * (time_delta / 60)  # Mental fatigue is constant
        
        self.physical_fatigue = min(1.0, self.physical_fatigue + physical_increase)
        self.mental_fatigue = min(1.0, self.mental_fatigue + mental_increase)
        self.active_time += time_delta
    
    def rest(self, duration: float):
        """Rest and reduce fatigue."""
        rest_effectiveness = 0.1 * (duration / 60)
        self.physical_fatigue = max(0.0, self.physical_fatigue - rest_effectiveness)
        self.mental_fatigue = max(0.0, self.mental_fatigue - rest_effectiveness * 0.5)
        self.last_rest_time = time.time()
    
    def overall_fatigue(self) -> float:
        """Get combined fatigue level."""
        return (self.physical_fatigue * 0.6 + self.mental_fatigue * 0.4)
    
    def needs_rest(self) -> bool:
        """Check if rest is needed."""
        return self.overall_fatigue() > 0.7


@dataclass
class AttentionState:
    """Tracks attention span and focus."""
    current_focus: Optional[str] = None
    focus_start_time: float = 0.0
    focus_duration: float = 0.0
    attention_span: float = 120.0  # Base attention span in seconds
    distractions: List[str] = field(default_factory=list)
    focus_switches: int = 0
    
    def start_focus(self, target: str):
        """Start focusing on a new target."""
        if self.current_focus and self.current_focus != target:
            self.focus_switches += 1
        self.current_focus = target
        self.focus_start_time = time.time()
        self.focus_duration = 0.0
    
    def update(self, time_delta: float):
        """Update focus duration."""
        if self.current_focus:
            self.focus_duration += time_delta
    
    def should_switch_focus(self, personality_impulsivity: float = 0.3) -> bool:
        """Check if focus should naturally shift."""
        if not self.current_focus:
            return False
        
        # Base probability increases with time
        base_prob = self.focus_duration / self.attention_span
        
        # Impulsive personalities switch more often
        impulsivity_mod = 1.0 + personality_impulsivity * 0.5
        
        # Distractions increase probability
        distraction_mod = 1.0 + len(self.distractions) * 0.1
        
        switch_prob = base_prob * impulsivity_mod * distraction_mod
        return random.random() < switch_prob * 0.1  # Per update check
    
    def add_distraction(self, distraction: str):
        """Add a potential distraction."""
        if len(self.distractions) < 5:
            self.distractions.append(distraction)
    
    def clear_distractions(self):
        """Clear all distractions."""
        self.distractions = []


@dataclass
class EmotionalResponse:
    """A recorded emotional response to an event."""
    trigger: str
    emotion: EmotionalState
    intensity: float  # 0 to 1
    timestamp: float
    duration: float = 30.0  # How long the emotion lasts
    
    def is_active(self) -> bool:
        return time.time() - self.timestamp < self.duration
    
    def current_intensity(self) -> float:
        """Get current intensity with decay."""
        elapsed = time.time() - self.timestamp
        if elapsed >= self.duration:
            return 0.0
        decay = 1.0 - (elapsed / self.duration)
        return self.intensity * decay


@dataclass
class LearningProgress:
    """Tracks improvement at specific tasks over time."""
    task_type: str
    attempts: int = 0
    successes: int = 0
    total_time: float = 0.0
    improvement_rate: float = 0.0
    skill_level: float = 0.1  # 0 to 1
    
    def record_attempt(self, success: bool, duration: float):
        """Record a task attempt."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.total_time += duration
        
        # Update skill level with learning curve
        if success:
            # Diminishing returns as skill increases
            improvement = 0.05 * (1 - self.skill_level)
            self.skill_level = min(1.0, self.skill_level + improvement)
        else:
            # Small decrease on failure
            self.skill_level = max(0.05, self.skill_level - 0.01)
        
        # Calculate improvement rate
        if self.attempts > 1:
            recent_success_rate = self.successes / self.attempts
            self.improvement_rate = (recent_success_rate - 0.5) * 2  # -1 to 1


@dataclass
class MovementModifier:
    """Modifier for a single movement action."""
    deviation_x: float = 0.0
    deviation_z: float = 0.0
    speed_multiplier: float = 1.0
    pause_before: float = 0.0
    pause_after: float = 0.0
    look_around: bool = False
    emotional_modifier: Optional[str] = None


class HumanBehaviorSystem:
    """Enhanced human-like behavior system with fatigue, attention, and emotions.
    
    Features:
    - Fatigue system: Actions slow down and become less precise when tired
    - Attention span: Natural focus shifts and distractions
    - Emotional responses: React to events with appropriate emotions
    - Idle behaviors: Natural looking around and small movements
    - Learning curve: Improve at tasks over time
    """
    
    def __init__(self, config, personality_engine=None):
        self.config = config.human_behavior
        self.personality = personality_engine
        
        # Fatigue tracking
        self.fatigue = FatigueState()
        
        # Attention tracking
        self.attention = AttentionState()
        
        # Emotional state
        self.emotional_responses: List[EmotionalResponse] = []
        self.current_emotion = EmotionalState.NEUTRAL
        
        # Learning progress
        self.learning_progress: Dict[str, LearningProgress] = {}
        
        # State tracking
        self.last_action_time = time.time()
        self.last_update_time = time.time()
        self.actions_since_break = 0
        self.current_focus_point: Optional[Tuple[float, float, float]] = None
        self.interesting_things: List[Dict] = []
        self.mistake_history: List[float] = []
        
        # Movement state
        self.movement_momentum = 0.0
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        
        # Idle behavior state
        self.idle_time = 0.0
        self.last_idle_action = 0.0
        
        # Emotional triggers
        self._setup_emotional_triggers()
    
    def _setup_emotional_triggers(self):
        """Setup emotional response triggers."""
        self.emotional_triggers = {
            # Positive triggers
            "task_success": (EmotionalState.SATISFIED, 0.6),
            "found_diamonds": (EmotionalState.EXCITED, 0.9),
            "level_up": (EmotionalState.EXCITED, 0.7),
            "new_discovery": (EmotionalState.CURIOUS, 0.6),
            
            # Negative triggers
            "task_failure": (EmotionalState.FRUSTRATED, 0.5),
            "death": (EmotionalState.FRUSTRATED, 0.9),
            "lost_items": (EmotionalState.FRUSTRATED, 0.7),
            "stuck": (EmotionalState.FRUSTRATED, 0.6),
            
            # Danger triggers
            "hostile_nearby": (EmotionalState.CAUTIOUS, 0.7),
            "low_health": (EmotionalState.ANXIOUS, 0.8),
            "night_time": (EmotionalState.CAUTIOUS, 0.4),
            
            # Neutral triggers
            "repeated_task": (EmotionalState.BORED, 0.4),
            "nothing_happening": (EmotionalState.BORED, 0.3),
        }
    
    def update(self, time_delta: float = None):
        """Update the behavior system state."""
        current_time = time.time()
        if time_delta is None:
            time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update fatigue
        activity_level = 0.5  # Default medium activity
        self.fatigue.update(activity_level, time_delta)
        
        # Update attention
        self.attention.update(time_delta)
        
        # Clean up expired emotional responses
        self.emotional_responses = [r for r in self.emotional_responses if r.is_active()]
        
        # Update current emotion based on active responses
        self._update_current_emotion()
        
        # Track idle time
        if current_time - self.last_action_time > 5.0:
            self.idle_time += time_delta
        else:
            self.idle_time = 0.0
    
    def _update_current_emotion(self):
        """Update current emotion based on active responses."""
        if not self.emotional_responses:
            self.current_emotion = EmotionalState.NEUTRAL
            return
        
        # Get strongest active emotion
        active = [(r.emotion, r.current_intensity()) for r in self.emotional_responses]
        if active:
            strongest = max(active, key=lambda x: x[1])
            if strongest[1] > 0.2:
                self.current_emotion = strongest[0]
            else:
                self.current_emotion = EmotionalState.NEUTRAL
    
    def trigger_emotion(self, trigger: str, intensity_override: float = None):
        """Trigger an emotional response."""
        if trigger not in self.emotional_triggers:
            return
        
        emotion, base_intensity = self.emotional_triggers[trigger]
        intensity = intensity_override if intensity_override else base_intensity
        
        # Personality affects emotional intensity
        if self.personality:
            impulsivity = self.personality.traits.get("impulsivity", 0.3)
            intensity *= (0.7 + impulsivity * 0.6)
        
        response = EmotionalResponse(
            trigger=trigger,
            emotion=emotion,
            intensity=min(1.0, intensity),
            timestamp=time.time(),
            duration=30.0 + random.uniform(-5, 10)
        )
        self.emotional_responses.append(response)
        
        # Limit active responses
        if len(self.emotional_responses) > 5:
            self.emotional_responses = sorted(
                self.emotional_responses, 
                key=lambda r: r.current_intensity(),
                reverse=True
            )[:5]
    
    def get_fatigue_modifier(self) -> float:
        """Get action speed modifier based on fatigue."""
        fatigue = self.fatigue.overall_fatigue()
        
        # Fatigue slows actions exponentially
        modifier = 1.0 + fatigue ** 2 * 0.5
        
        return modifier
    
    def get_precision_modifier(self) -> float:
        """Get precision modifier based on fatigue and emotion."""
        fatigue = self.fatigue.overall_fatigue()
        
        # Base precision decreases with fatigue
        precision = 1.0 - fatigue * 0.3
        
        # Emotions affect precision
        if self.current_emotion == EmotionalState.ANXIOUS:
            precision *= 0.8
        elif self.current_emotion == EmotionalState.FRUSTRATED:
            precision *= 0.85
        elif self.current_emotion == EmotionalState.CAUTIOUS:
            precision *= 1.1  # More careful = more precise
        elif self.current_emotion == EmotionalState.EXCITED:
            precision *= 0.9  # Excitement can make you sloppy
        
        return max(0.5, min(1.2, precision))
    
    def process_movement(self, target_position: Tuple[float, float, float],
                        current_position: Tuple[float, float, float]) -> MovementModifier:
        """Process a movement action with human-like variations."""
        modifier = MovementModifier()
        
        # Fatigue affects speed
        fatigue_mod = self.get_fatigue_modifier()
        modifier.speed_multiplier = 1.0 / fatigue_mod
        
        # Precision affects deviation
        precision = self.get_precision_modifier()
        if random.random() < self.config.path_deviation_chance * (2 - precision):
            deviation_amount = self.config.path_deviation_amount / precision
            modifier.deviation_x = random.uniform(-deviation_amount, deviation_amount)
            modifier.deviation_z = random.uniform(-deviation_amount, deviation_amount)
        
        # Emotional effects on movement
        if self.current_emotion == EmotionalState.ANXIOUS:
            modifier.speed_multiplier *= 1.2  # Move faster when anxious
            modifier.look_around = True
        elif self.current_emotion == EmotionalState.CAUTIOUS:
            modifier.speed_multiplier *= 0.8  # Move slower when cautious
            modifier.look_around = True
        elif self.current_emotion == EmotionalState.EXCITED:
            modifier.speed_multiplier *= 1.15
        elif self.current_emotion == EmotionalState.BORED:
            modifier.speed_multiplier *= 0.9
            # Might pause to look around
            if random.random() < 0.2:
                modifier.pause_before = random.uniform(0.5, 2.0)
        
        modifier.emotional_modifier = self.current_emotion.value
        
        # Thinking pauses
        mental_fatigue = self.fatigue.mental_fatigue
        if random.random() < self.config.thinking_pause_chance * (1 + mental_fatigue):
            min_pause, max_pause = self.config.thinking_pause_duration
            modifier.pause_before = random.uniform(min_pause, max_pause * (1 + mental_fatigue))
        
        # Random look around
        if random.random() < self.config.look_around_frequency:
            modifier.look_around = True
        
        self.last_action_time = time.time()
        return modifier
    
    def should_make_mistake(self) -> Tuple[bool, str]:
        """Determine if the agent should make a minor mistake."""
        # Base chance affected by fatigue and emotion
        base_chance = self.config.mistake_chance
        
        fatigue_mod = 1.0 + self.fatigue.overall_fatigue()
        emotion_mod = 1.0
        if self.current_emotion == EmotionalState.FRUSTRATED:
            emotion_mod = 1.5
        elif self.current_emotion == EmotionalState.ANXIOUS:
            emotion_mod = 1.3
        elif self.current_emotion == EmotionalState.CAUTIOUS:
            emotion_mod = 0.7
        
        adjusted_chance = base_chance * fatigue_mod * emotion_mod
        
        if random.random() >= adjusted_chance:
            return False, ""
        
        # Don't make too many mistakes in a row
        current_time = time.time()
        recent_mistakes = [t for t in self.mistake_history if current_time - t < 60]
        if len(recent_mistakes) >= 3:
            return False, ""
        
        mistake_types = [
            "wrong_direction",
            "misclick",
            "overshoot",
            "wrong_tool",
            "hesitation",
            "double_click",
            "forgot_item"
        ]
        
        # Weight mistakes by emotional state
        weights = [1.0] * len(mistake_types)
        if self.current_emotion == EmotionalState.FRUSTRATED:
            weights[mistake_types.index("misclick")] = 2.0
            weights[mistake_types.index("wrong_tool")] = 1.5
        elif self.current_emotion == EmotionalState.ANXIOUS:
            weights[mistake_types.index("hesitation")] = 2.0
            weights[mistake_types.index("double_click")] = 1.5
        
        mistake_type = random.choices(mistake_types, weights=weights, k=1)[0]
        self.mistake_history.append(current_time)
        
        return True, mistake_type
    
    def should_take_break(self) -> Tuple[bool, float, str]:
        """Determine if agent should take a break.
        
        Returns:
            (should_break, duration, reason)
        """
        if not self.config.break_between_tasks:
            return False, 0.0, ""
        
        self.actions_since_break += 1
        
        # Fatigue-based break need
        if self.fatigue.needs_rest():
            self.actions_since_break = 0
            duration = random.uniform(3.0, 8.0)
            self.fatigue.rest(duration)
            return True, duration, "fatigue"
        
        # Attention-based break
        if self.attention.should_switch_focus(
            self.personality.traits.get("impulsivity", 0.3) if self.personality else 0.3
        ):
            self.actions_since_break = 0
            duration = random.uniform(1.0, 3.0)
            return True, duration, "attention_shift"
        
        # Random break probability increases with actions
        break_chance = min(0.5, self.actions_since_break * 0.05)
        
        # Emotional state affects break taking
        if self.current_emotion == EmotionalState.FRUSTRATED:
            break_chance *= 1.5
        elif self.current_emotion == EmotionalState.BORED:
            break_chance *= 1.3
        elif self.current_emotion == EmotionalState.EXCITED:
            break_chance *= 0.5
        
        if random.random() < break_chance:
            self.actions_since_break = 0
            min_break, max_break = self.config.break_duration
            duration = random.uniform(min_break, max_break)
            return True, duration, "natural_pause"
        
        return False, 0.0, ""
    
    def record_task_attempt(self, task_type: str, success: bool, duration: float):
        """Record a task attempt for learning curve."""
        if task_type not in self.learning_progress:
            self.learning_progress[task_type] = LearningProgress(task_type=task_type)
        
        self.learning_progress[task_type].record_attempt(success, duration)
        
        # Trigger emotional response
        if success:
            self.trigger_emotion("task_success")
        else:
            self.trigger_emotion("task_failure")
    
    def get_skill_level(self, task_type: str) -> float:
        """Get current skill level for a task type."""
        if task_type not in self.learning_progress:
            return 0.1  # Base skill
        return self.learning_progress[task_type].skill_level
    
    def get_task_speed_modifier(self, task_type: str) -> float:
        """Get speed modifier based on skill level and fatigue."""
        skill = self.get_skill_level(task_type)
        fatigue_mod = self.get_fatigue_modifier()
        
        # Higher skill = faster execution
        skill_speed = 0.7 + skill * 0.6  # 0.7x to 1.3x
        
        return skill_speed / fatigue_mod
    
    def get_idle_behavior(self) -> Optional[Dict[str, Any]]:
        """Get an idle behavior to perform."""
        if self.idle_time < 2.0:
            return None
        
        current_time = time.time()
        if current_time - self.last_idle_action < 3.0:
            return None
        
        self.last_idle_action = current_time
        
        behaviors = [
            {"type": "look_around", "weight": 3.0},
            {"type": "small_movement", "weight": 1.5},
            {"type": "check_inventory", "weight": 1.0},
            {"type": "look_at_sky", "weight": 0.5},
            {"type": "crouch_uncrouch", "weight": 0.3},
        ]
        
        # Emotional state affects idle behaviors
        if self.current_emotion == EmotionalState.ANXIOUS:
            behaviors[0]["weight"] *= 2.0  # Look around more
        elif self.current_emotion == EmotionalState.BORED:
            behaviors[1]["weight"] *= 2.0  # Move around more
        elif self.current_emotion == EmotionalState.CURIOUS:
            behaviors[3]["weight"] *= 3.0  # Look at things
        
        weights = [b["weight"] for b in behaviors]
        selected = random.choices(behaviors, weights=weights, k=1)[0]
        
        if selected["type"] == "look_around":
            yaw_change = random.gauss(0, 45)
            pitch_change = random.gauss(0, 20)
            return {
                "type": "look_around",
                "yaw_delta": yaw_change,
                "pitch_delta": pitch_change,
                "duration": random.uniform(0.5, 1.5)
            }
        elif selected["type"] == "small_movement":
            return {
                "type": "small_movement",
                "direction": random.choice(["forward", "backward", "left", "right"]),
                "distance": random.uniform(0.5, 2.0),
                "duration": random.uniform(0.3, 1.0)
            }
        elif selected["type"] == "check_inventory":
            return {
                "type": "check_inventory",
                "duration": random.uniform(1.0, 3.0)
            }
        elif selected["type"] == "look_at_sky":
            return {
                "type": "look_at_sky",
                "pitch": random.uniform(-80, -60),
                "duration": random.uniform(1.0, 3.0)
            }
        elif selected["type"] == "crouch_uncrouch":
            return {
                "type": "crouch_uncrouch",
                "duration": random.uniform(0.5, 1.0)
            }
        
        return None
    
    def get_look_around_movement(self) -> Dict[str, float]:
        """Generate natural looking-around head movement."""
        # Emotional state affects head movement
        if self.current_emotion == EmotionalState.ANXIOUS:
            # Quick, nervous movements
            target_yaw = self.head_yaw + random.gauss(0, 60)
            target_pitch = random.gauss(0, 25)
            smoothness = self.config.head_turn_smoothness * 1.5
        elif self.current_emotion == EmotionalState.CURIOUS:
            # Slower, more deliberate looking
            target_yaw = self.head_yaw + random.gauss(0, 40)
            target_pitch = random.gauss(-10, 20)
            smoothness = self.config.head_turn_smoothness * 0.7
        else:
            target_yaw = self.head_yaw + random.gauss(0, 30)
            target_pitch = random.gauss(0, 15)
            smoothness = self.config.head_turn_smoothness
        
        self.head_yaw += (target_yaw - self.head_yaw) * smoothness
        self.head_pitch += (target_pitch - self.head_pitch) * smoothness
        self.head_pitch = max(-90, min(90, self.head_pitch))
        
        return {
            "yaw": self.head_yaw,
            "pitch": self.head_pitch
        }
    
    def update_interesting_things(self, events: List) -> List[Dict]:
        """Update list of interesting things to look at."""
        if not self.config.look_at_interesting_things:
            return []
        
        interesting = []
        
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                # Nearby entities
                entities = event[1].get("nearbyEntities", [])
                for entity in entities:
                    if isinstance(entity, dict) and "position" in entity:
                        interest_level = self._calculate_interest_level(entity)
                        # Emotional state affects interest
                        if self.current_emotion == EmotionalState.CURIOUS:
                            interest_level *= 1.3
                        elif self.current_emotion == EmotionalState.CAUTIOUS:
                            # More interested in potential threats
                            if self._is_threat(entity):
                                interest_level *= 1.5
                        
                        interesting.append({
                            "type": "entity",
                            "name": entity.get("name", "unknown"),
                            "position": self._extract_position(entity["position"]),
                            "interest_level": interest_level
                        })
                
                # Nearby players
                players = event[1].get("nearbyPlayers", [])
                for player in players:
                    if isinstance(player, dict) and "position" in player:
                        interest_level = 1.0
                        if self.current_emotion == EmotionalState.CURIOUS:
                            interest_level = 1.3
                        
                        # Add as potential distraction
                        self.attention.add_distraction(f"player:{player.get('name', 'unknown')}")
                        
                        interesting.append({
                            "type": "player",
                            "name": player.get("name", "unknown"),
                            "position": self._extract_position(player["position"]),
                            "interest_level": interest_level
                        })
        
        # Sort and limit
        interesting.sort(key=lambda x: x["interest_level"], reverse=True)
        self.interesting_things = interesting[:5]
        
        return self.interesting_things
    
    def _extract_position(self, pos) -> Tuple[float, float, float]:
        """Extract position tuple from various formats."""
        if isinstance(pos, dict):
            return (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
        elif isinstance(pos, (list, tuple)):
            return tuple(pos[:3])
        return (0, 0, 0)
    
    def _calculate_interest_level(self, entity: Dict) -> float:
        """Calculate how interesting an entity is."""
        name = entity.get("name", "").lower()
        
        # High interest: threats
        hostile = ["zombie", "skeleton", "creeper", "spider", "enderman", "witch"]
        if any(mob in name for mob in hostile):
            return 0.9
        
        # High interest: rare
        rare = ["villager", "iron_golem", "wolf", "cat", "parrot", "panda"]
        if any(r in name for r in rare):
            return 0.8
        
        # Medium interest: animals
        animals = ["cow", "pig", "sheep", "chicken", "horse", "rabbit"]
        if any(a in name for a in animals):
            return 0.5
        
        return 0.3
    
    def _is_threat(self, entity: Dict) -> bool:
        """Check if entity is a threat."""
        name = entity.get("name", "").lower()
        hostile = ["zombie", "skeleton", "creeper", "spider", "enderman", 
                  "witch", "pillager", "vindicator", "phantom"]
        return any(mob in name for mob in hostile)
    
    def generate_natural_action_code(self, base_code: str, action_type: str) -> str:
        """Wrap action code with human-like behaviors."""
        modifications = []
        
        # Add thinking pause based on mental fatigue
        if random.random() < self.config.thinking_pause_chance * (1 + self.fatigue.mental_fatigue):
            min_pause, max_pause = self.config.thinking_pause_duration
            pause = random.uniform(min_pause, max_pause)
            modifications.append(f"await bot.waitForTicks({int(pause * 20)});  // Thinking...")
        
        # Emotional modifications
        if self.current_emotion == EmotionalState.ANXIOUS:
            modifications.append("// Nervous - checking surroundings")
            modifications.append("await bot.look(bot.entity.yaw + Math.random() * 0.5, bot.entity.pitch, false);")
        elif self.current_emotion == EmotionalState.CAUTIOUS:
            modifications.append("await bot.waitForTicks(10);  // Being careful...")
        elif self.current_emotion == EmotionalState.EXCITED:
            # Might skip some safety checks
            pass
        
        # Fatigue modifications
        if self.fatigue.overall_fatigue() > 0.5:
            modifications.append("// Tired - moving slower")
        
        # Hesitation
        if random.random() < self.config.decision_hesitation * (1 + self.fatigue.mental_fatigue):
            modifications.append("await bot.waitForTicks(10);  // Slight hesitation")
        
        # Looking around
        if random.random() < self.config.look_around_frequency:
            look_code = self._generate_look_around_code()
            modifications.append(look_code)
        
        if modifications:
            pre_code = "\n".join(modifications)
            return f"{pre_code}\n{base_code}"
        
        return base_code
    
    def _generate_look_around_code(self) -> str:
        """Generate JavaScript code for looking around naturally."""
        yaw_change = random.uniform(-45, 45)
        pitch_change = random.uniform(-20, 20)
        
        return f"""// Natural look around
await bot.look(bot.entity.yaw + {yaw_change:.2f} * Math.PI / 180, bot.entity.pitch + {pitch_change:.2f} * Math.PI / 180, false);
await bot.waitForTicks(5);"""
    
    def get_behavior_summary(self) -> Dict:
        """Get summary of current behavior state."""
        return {
            "fatigue": {
                "physical": self.fatigue.physical_fatigue,
                "mental": self.fatigue.mental_fatigue,
                "overall": self.fatigue.overall_fatigue(),
                "needs_rest": self.fatigue.needs_rest()
            },
            "attention": {
                "current_focus": self.attention.current_focus,
                "focus_duration": self.attention.focus_duration,
                "focus_switches": self.attention.focus_switches,
                "distractions": len(self.attention.distractions)
            },
            "emotion": {
                "current": self.current_emotion.value,
                "active_responses": len(self.emotional_responses)
            },
            "learning": {
                task: {
                    "skill_level": prog.skill_level,
                    "attempts": prog.attempts,
                    "success_rate": prog.successes / max(prog.attempts, 1)
                }
                for task, prog in self.learning_progress.items()
            },
            "state": {
                "actions_since_break": self.actions_since_break,
                "idle_time": self.idle_time,
                "interesting_things": len(self.interesting_things)
            }
        }
