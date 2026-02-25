"""Personality Engine for Voyager Evolved.

This module manages the agent's personality traits, mood, and how they
influence decision-making and behavior over time.
"""

import random
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import voyager.utils as U


@dataclass
class MoodState:
    """Current mood/emotional state of the agent."""
    happiness: float = 0.5  # -1 to 1
    frustration: float = 0.0  # 0 to 1
    excitement: float = 0.5  # 0 to 1
    confidence: float = 0.5  # 0 to 1
    boredom: float = 0.0  # 0 to 1
    
    def overall_mood(self) -> float:
        """Calculate overall mood score."""
        return (self.happiness + self.confidence + self.excitement - 
                self.frustration - self.boredom) / 3
    
    def decay(self, rate: float = 0.1):
        """Decay mood towards neutral over time."""
        self.happiness = self.happiness * (1 - rate) + 0.5 * rate
        self.frustration = self.frustration * (1 - rate)
        self.excitement = self.excitement * (1 - rate) + 0.5 * rate
        self.confidence = self.confidence * (1 - rate) + 0.5 * rate
        self.boredom = self.boredom * (1 - rate)


class PersonalityEngine:
    """Manages agent personality and its influence on behavior.
    
    Personality traits influence:
    - Goal selection and prioritization
    - Risk assessment
    - Learning rate from observations
    - Response to success/failure
    - Human-likeness of actions
    """
    
    def __init__(self, config, ckpt_dir: str = "ckpt", resume: bool = False):
        self.config = config.personality
        self.ckpt_dir = ckpt_dir
        
        # Core traits (mostly stable)
        self.traits = {
            "curiosity": self.config.curiosity,
            "caution": self.config.caution,
            "sociability": self.config.sociability,
            "persistence": self.config.persistence,
            "creativity": self.config.creativity,
            "impulsivity": self.config.impulsivity,
            "imitation_tendency": self.config.imitation_tendency,
            "self_reliance": self.config.self_reliance
        }
        
        # Dynamic state
        self.mood = MoodState()
        self.energy = self.config.energy_level
        
        # Experience tracking
        self.recent_successes: List[float] = []
        self.recent_failures: List[float] = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Memory of preferences developed through experience
        self.learned_preferences: Dict[str, float] = {}
        self.avoided_situations: Dict[str, int] = {}
        
        # Setup persistence
        U.f_mkdir(f"{ckpt_dir}/personality")
        
        if resume:
            self._load_state()
    
    def _load_state(self):
        """Load saved personality state."""
        try:
            state_path = f"{self.ckpt_dir}/personality/state.json"
            if U.f_exists(state_path):
                data = U.load_json(state_path)
                
                # Load evolved traits
                self.traits.update(data.get("traits", {}))
                
                # Load mood
                mood_data = data.get("mood", {})
                self.mood = MoodState(**mood_data)
                
                # Load learned preferences
                self.learned_preferences = data.get("learned_preferences", {})
                self.avoided_situations = data.get("avoided_situations", {})
                
                print(f"\033[36mLoaded personality state\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load personality state: {e}\033[0m")
    
    def save_state(self):
        """Save personality state to disk."""
        state = {
            "traits": self.traits,
            "mood": asdict(self.mood),
            "energy": self.energy,
            "learned_preferences": self.learned_preferences,
            "avoided_situations": self.avoided_situations
        }
        U.dump_json(state, f"{self.ckpt_dir}/personality/state.json")
    
    def record_success(self, task: str, difficulty: float = 0.5):
        """Record a successful task completion."""
        current_time = time.time()
        self.recent_successes.append(current_time)
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        
        # Mood updates
        self.mood.happiness = min(1.0, self.mood.happiness + 0.1 * difficulty)
        self.mood.confidence = min(1.0, self.mood.confidence + 0.05)
        self.mood.frustration = max(0.0, self.mood.frustration - 0.1)
        self.mood.boredom = max(0.0, self.mood.boredom - 0.1)
        
        # Excitement from streak
        if self.consecutive_successes > 3:
            self.mood.excitement = min(1.0, self.mood.excitement + 0.15)
        
        # Trait evolution (very slow)
        if random.random() < 0.1:
            self.traits["confidence"] = min(1.0, self.traits.get("confidence", 0.5) + 0.01)
        
        # Learn preference for successful approaches
        task_category = self._extract_task_category(task)
        self.learned_preferences[task_category] = \
            self.learned_preferences.get(task_category, 0.5) + 0.05
        
        # Trim old entries
        cutoff = current_time - 3600  # Last hour
        self.recent_successes = [t for t in self.recent_successes if t > cutoff]
    
    def record_failure(self, task: str, reason: str = ""):
        """Record a failed task attempt."""
        current_time = time.time()
        self.recent_failures.append(current_time)
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        # Mood updates
        self.mood.frustration = min(1.0, self.mood.frustration + 0.15)
        self.mood.happiness = max(-1.0, self.mood.happiness - 0.1)
        self.mood.confidence = max(0.0, self.mood.confidence - 0.05)
        
        # Boredom from repeated failures
        if self.consecutive_failures > 3:
            self.mood.boredom = min(1.0, self.mood.boredom + 0.1)
            self.mood.excitement = max(0.0, self.mood.excitement - 0.1)
        
        # Learn to avoid problematic situations
        task_category = self._extract_task_category(task)
        self.avoided_situations[task_category] = \
            self.avoided_situations.get(task_category, 0) + 1
        
        # Decrease preference
        self.learned_preferences[task_category] = \
            max(0.0, self.learned_preferences.get(task_category, 0.5) - 0.1)
        
        # Trim old entries
        cutoff = current_time - 3600
        self.recent_failures = [t for t in self.recent_failures if t > cutoff]
    
    def _extract_task_category(self, task: str) -> str:
        """Extract category from task description."""
        task_lower = task.lower()
        
        categories = [
            ("mine", "mining"),
            ("craft", "crafting"),
            ("build", "building"),
            ("farm", "farming"),
            ("cook", "cooking"),
            ("explore", "exploration"),
            ("fight", "combat"),
            ("kill", "combat"),
            ("collect", "gathering"),
            ("obtain", "gathering"),
            ("find", "exploration"),
            ("smelt", "crafting")
        ]
        
        for keyword, category in categories:
            if keyword in task_lower:
                return category
        
        return "general"
    
    def get_goal_preference_modifier(self, goal_type: str) -> float:
        """Get how much personality affects preference for a goal type.
        
        Returns a multiplier (0.5 to 1.5) for goal priority.
        """
        modifier = 1.0
        
        # Curiosity affects exploration
        if goal_type in ["exploration", "discovery"]:
            modifier += (self.traits["curiosity"] - 0.5) * 0.6
        
        # Caution affects risky goals
        if goal_type in ["combat", "exploration", "cave_mining"]:
            modifier -= (self.traits["caution"] - 0.5) * 0.4
        
        # Sociability affects learning from others
        if goal_type in ["social_learning", "observation"]:
            modifier += (self.traits["sociability"] - 0.5) * 0.6
        
        # Persistence affects difficult goals
        if goal_type in ["difficult", "multi_step"]:
            modifier += (self.traits["persistence"] - 0.5) * 0.4
        
        # Creativity affects novel approaches
        if goal_type in ["creative", "building", "experimental"]:
            modifier += (self.traits["creativity"] - 0.5) * 0.5
        
        # Learned preferences
        if goal_type in self.learned_preferences:
            modifier += (self.learned_preferences[goal_type] - 0.5) * 0.3
        
        # Mood affects everything
        overall_mood = self.mood.overall_mood()
        modifier += overall_mood * 0.15
        
        # Energy level affects ambitious goals
        if goal_type in ["difficult", "exploration", "combat"]:
            modifier *= (0.5 + self.energy * 0.5)
        
        return max(0.5, min(1.5, modifier))
    
    def should_take_risk(self, risk_level: float) -> Tuple[bool, str]:
        """Decide whether to take a risky action.
        
        Args:
            risk_level: 0.0 (safe) to 1.0 (very risky)
            
        Returns:
            (decision, reasoning)
        """
        # Base willingness from traits
        risk_tolerance = 1.0 - self.traits["caution"]
        risk_tolerance += self.traits["impulsivity"] * 0.3
        
        # Mood affects risk taking
        if self.mood.excitement > 0.7:
            risk_tolerance += 0.2
        if self.mood.frustration > 0.5:
            risk_tolerance -= 0.15  # Frustrated = more cautious
        if self.mood.confidence > 0.7:
            risk_tolerance += 0.15
        
        # Recent experiences
        if self.consecutive_failures > 2:
            risk_tolerance -= 0.2  # Become more cautious
        if self.consecutive_successes > 3:
            risk_tolerance += 0.1  # Feel lucky
        
        # Make decision with some randomness
        threshold = risk_level + random.gauss(0, 0.1)
        decision = risk_tolerance > threshold
        
        # Generate reasoning
        if decision:
            if self.mood.confidence > 0.7:
                reason = "Feeling confident, let's try it!"
            elif self.traits["impulsivity"] > 0.6:
                reason = "Acting on impulse."
            else:
                reason = "The risk seems acceptable."
        else:
            if self.mood.frustration > 0.5:
                reason = "Too frustrated to take more risks."
            elif self.traits["caution"] > 0.7:
                reason = "Better to play it safe."
            else:
                reason = "The risk is too high right now."
        
        return decision, reason
    
    def should_imitate_player(self, player_success_rate: float) -> Tuple[bool, float]:
        """Decide whether to imitate an observed player behavior.
        
        Returns:
            (should_imitate, imitation_fidelity)
        """
        base_imitation = self.traits["imitation_tendency"]
        self_reliance = self.traits["self_reliance"]
        
        # More likely to imitate successful players
        success_boost = (player_success_rate - 0.5) * 0.4
        
        # Less likely if very self-reliant
        self_reliance_penalty = (self_reliance - 0.5) * 0.3
        
        # More likely if frustrated (looking for alternatives)
        frustration_boost = self.mood.frustration * 0.2
        
        # Less likely if very confident
        confidence_penalty = (self.mood.confidence - 0.5) * 0.15
        
        imitation_score = base_imitation + success_boost - self_reliance_penalty + \
                         frustration_boost - confidence_penalty
        
        should_imitate = random.random() < imitation_score
        
        # Fidelity: how closely to copy (higher = more exact copy)
        fidelity = 0.5 + (self.traits["imitation_tendency"] * 0.3)
        if self.mood.confidence < 0.3:
            fidelity += 0.2  # Less confident = copy more exactly
        
        return should_imitate, min(1.0, fidelity)
    
    def get_action_timing_modifier(self) -> float:
        """Get a modifier for action timing (for human-like pacing).
        
        Returns multiplier for action delays.
        """
        # Base from energy
        base = 1.0 - (self.energy - 0.5) * 0.4  # Higher energy = faster
        
        # Mood effects
        if self.mood.excitement > 0.7:
            base *= 0.8  # Excited = faster
        if self.mood.boredom > 0.5:
            base *= 1.2  # Bored = slower
        if self.mood.frustration > 0.7:
            base *= 0.9  # Frustrated = slightly faster (agitated)
        
        # Add randomness
        base *= 1.0 + random.gauss(0, 0.1)
        
        return max(0.5, min(2.0, base))
    
    def update_energy(self, activity_level: float, time_elapsed: float):
        """Update energy level based on activity.
        
        Args:
            activity_level: 0 (resting) to 1 (very active)
            time_elapsed: Time in seconds
        """
        # Drain energy with activity
        drain_rate = activity_level * 0.01 * (time_elapsed / 60)
        self.energy = max(0.0, self.energy - drain_rate)
        
        # Recover energy slightly when idle
        if activity_level < 0.2:
            recovery = 0.005 * (time_elapsed / 60)
            self.energy = min(1.0, self.energy + recovery)
        
        # Mood decay over time
        self.mood.decay(rate=0.01 * (time_elapsed / 60))
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of current personality state."""
        return {
            "traits": self.traits,
            "mood": asdict(self.mood),
            "overall_mood": self.mood.overall_mood(),
            "energy": self.energy,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "top_preferences": sorted(
                self.learned_preferences.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "avoided_situations": sorted(
                self.avoided_situations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
