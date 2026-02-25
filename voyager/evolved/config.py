"""Configuration system for Voyager Evolved."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class PersonalityConfig:
    """Configuration for personality traits that influence decision-making."""
    # Core personality dimensions (0.0 to 1.0)
    curiosity: float = 0.7  # Drive to explore and try new things
    caution: float = 0.5  # Tendency to avoid risks
    sociability: float = 0.6  # Interest in observing/interacting with players
    persistence: float = 0.7  # Determination to complete tasks
    creativity: float = 0.5  # Willingness to try unconventional approaches
    impulsivity: float = 0.3  # Tendency to act without full planning
    
    # Learning traits
    imitation_tendency: float = 0.6  # How much to copy other players
    self_reliance: float = 0.5  # Trust in own abilities vs learned behaviors
    
    # Mood modifiers (evolve over time)
    current_mood: float = 0.5  # -1.0 (frustrated) to 1.0 (enthusiastic)
    energy_level: float = 0.8  # Affects action speed and complexity


@dataclass 
class ObservationConfig:
    """Configuration for player observation system."""
    # Detection settings
    observation_radius: int = 32  # Blocks radius to detect players
    update_frequency: float = 1.0  # Seconds between observation updates
    max_tracked_players: int = 10  # Maximum players to track simultaneously
    
    # Behavior classification
    activity_detection_window: float = 5.0  # Seconds to analyze for activity type
    min_observation_confidence: float = 0.6  # Minimum confidence to record behavior
    
    # Memory settings
    max_observation_memory: int = 1000  # Maximum stored observations
    observation_decay_rate: float = 0.1  # How fast old observations lose weight
    
    # Player profiling
    build_player_profiles: bool = True  # Create detailed player profiles
    track_player_skills: bool = True  # Track what skills players demonstrate


@dataclass
class EvolutionaryGoalConfig:
    """Configuration for the evolutionary goal generation system."""
    # Goal weights (relative importance)
    survival_weight: float = 0.3  # Food, health, shelter
    exploration_weight: float = 0.25  # Discovering new areas/items
    social_learning_weight: float = 0.2  # Learning from other players
    skill_development_weight: float = 0.15  # Improving existing abilities
    creativity_weight: float = 0.1  # Trying new approaches
    
    # Survival instincts
    hunger_threshold: float = 8.0  # Hunger level that triggers food-seeking
    health_threshold: float = 10.0  # Health level that triggers safety-seeking
    shelter_priority_at_night: bool = True  # Prioritize shelter at night
    
    # Exploration settings
    novelty_seeking_radius: int = 100  # Blocks to consider for novelty
    revisit_cooldown: float = 300.0  # Seconds before revisiting explored areas
    biome_diversity_bonus: float = 0.2  # Bonus for exploring new biomes
    
    # Evolution parameters
    goal_mutation_rate: float = 0.1  # Rate of random goal modifications
    fitness_memory_size: int = 100  # Number of past goals to track fitness
    adaptation_rate: float = 0.05  # How fast to adapt goal preferences
    
    # Social learning
    observe_before_imitate: int = 3  # Times to observe before copying
    success_imitation_boost: float = 0.3  # Boost for observed successful actions


@dataclass
class HumanBehaviorConfig:
    """Configuration for human-like behavior patterns."""
    # Movement naturalness
    path_deviation_chance: float = 0.15  # Chance to deviate from optimal path
    path_deviation_amount: float = 2.0  # Max blocks to deviate
    look_around_frequency: float = 0.2  # Chance to look around per action
    
    # Pauses and hesitation
    thinking_pause_chance: float = 0.1  # Chance to pause before actions
    thinking_pause_duration: tuple = (0.5, 2.0)  # Min/max pause duration
    decision_hesitation: float = 0.05  # Chance to reconsider decisions
    
    # Mistakes and suboptimal choices
    mistake_chance: float = 0.05  # Chance of making minor mistakes
    suboptimal_choice_chance: float = 0.1  # Chance of non-optimal but valid choice
    recovery_from_mistake: float = 0.8  # Chance to correct mistakes
    
    # Camera/head movement
    natural_head_movement: bool = True  # Enable natural looking behavior
    look_at_interesting_things: bool = True  # Look at mobs, players, etc.
    head_turn_smoothness: float = 0.3  # Lower = smoother head turns
    
    # Pacing variation
    action_speed_variation: float = 0.2  # Variation in action timing
    break_between_tasks: bool = True  # Take breaks between major tasks
    break_duration: tuple = (1.0, 5.0)  # Min/max break duration


@dataclass
class EvolvedConfig:
    """Main configuration for Voyager Evolved."""
    # Sub-configurations
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    evolutionary_goals: EvolutionaryGoalConfig = field(default_factory=EvolutionaryGoalConfig)
    human_behavior: HumanBehaviorConfig = field(default_factory=HumanBehaviorConfig)
    
    # General settings
    enable_player_observation: bool = True
    enable_evolutionary_goals: bool = True
    enable_human_behavior: bool = True
    enable_observational_learning: bool = True
    
    # Logging and debugging
    log_observations: bool = True
    log_goal_evolution: bool = True
    log_behavior_decisions: bool = False
    verbose_mode: bool = False
    
    # Persistence
    save_personality_state: bool = True
    save_learned_behaviors: bool = True
    checkpoint_interval: int = 50  # Save state every N iterations
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EvolvedConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.personality = PersonalityConfig(**data.get('personality', {}))
        config.observation = ObservationConfig(**data.get('observation', {}))
        config.evolutionary_goals = EvolutionaryGoalConfig(**data.get('evolutionary_goals', {}))
        config.human_behavior = HumanBehaviorConfig(**data.get('human_behavior', {}))
        
        # Load general settings
        for key in ['enable_player_observation', 'enable_evolutionary_goals', 
                    'enable_human_behavior', 'enable_observational_learning',
                    'log_observations', 'log_goal_evolution', 'log_behavior_decisions',
                    'verbose_mode', 'save_personality_state', 'save_learned_behaviors',
                    'checkpoint_interval']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @classmethod
    def create_curious_explorer(cls) -> 'EvolvedConfig':
        """Create a configuration for a curious, exploration-focused agent."""
        config = cls()
        config.personality.curiosity = 0.9
        config.personality.caution = 0.3
        config.personality.impulsivity = 0.5
        config.evolutionary_goals.exploration_weight = 0.4
        config.evolutionary_goals.survival_weight = 0.2
        return config
    
    @classmethod
    def create_social_learner(cls) -> 'EvolvedConfig':
        """Create a configuration for an agent focused on learning from others."""
        config = cls()
        config.personality.sociability = 0.9
        config.personality.imitation_tendency = 0.8
        config.observation.observation_radius = 48
        config.evolutionary_goals.social_learning_weight = 0.4
        return config
    
    @classmethod
    def create_survivor(cls) -> 'EvolvedConfig':
        """Create a configuration for a survival-focused agent."""
        config = cls()
        config.personality.caution = 0.8
        config.personality.persistence = 0.9
        config.evolutionary_goals.survival_weight = 0.5
        config.human_behavior.mistake_chance = 0.02
        return config
    
    @classmethod
    def create_natural_player(cls) -> 'EvolvedConfig':
        """Create a configuration that maximizes human-like behavior."""
        config = cls()
        config.human_behavior.path_deviation_chance = 0.25
        config.human_behavior.thinking_pause_chance = 0.15
        config.human_behavior.look_around_frequency = 0.3
        config.human_behavior.action_speed_variation = 0.3
        return config
