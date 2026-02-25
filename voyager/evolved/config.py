"""Enhanced Configuration System for Voyager Evolved (Linux Optimized).

Features:
- Performance profiles (fast/balanced/quality)
- Linux-optimized default settings
- Debug mode with detailed logging
- Configuration validation
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class PerformanceProfile(Enum):
    """Performance profiles for different use cases."""
    FAST = "fast"  # Prioritize speed, lower quality
    BALANCED = "balanced"  # Default balance
    QUALITY = "quality"  # Prioritize quality, slower


@dataclass
class PersonalityConfig:
    """Configuration for personality traits that influence decision-making."""
    # Core personality dimensions (0.0 to 1.0)
    curiosity: float = 0.7
    caution: float = 0.5
    sociability: float = 0.6
    persistence: float = 0.7
    creativity: float = 0.5
    impulsivity: float = 0.3
    
    # Learning traits
    imitation_tendency: float = 0.6
    self_reliance: float = 0.5
    
    # Mood modifiers
    current_mood: float = 0.5
    energy_level: float = 0.8
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration values."""
        errors = []
        for field_name in ['curiosity', 'caution', 'sociability', 'persistence', 
                          'creativity', 'impulsivity', 'imitation_tendency', 
                          'self_reliance', 'current_mood', 'energy_level']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{field_name} must be between 0.0 and 1.0 (got {value})")
        return len(errors) == 0, errors


@dataclass 
class ObservationConfig:
    """Configuration for player observation system."""
    observation_radius: int = 32
    update_frequency: float = 1.0
    max_tracked_players: int = 10
    activity_detection_window: float = 5.0
    min_observation_confidence: float = 0.6
    max_observation_memory: int = 1000
    observation_decay_rate: float = 0.1
    build_player_profiles: bool = True
    track_player_skills: bool = True
    
    # Pattern recognition
    enable_pattern_recognition: bool = True
    pattern_clustering_eps: float = 0.5
    min_pattern_samples: int = 3
    
    # Batch processing
    batch_size: int = 10
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.observation_radius < 1 or self.observation_radius > 128:
            errors.append("observation_radius must be between 1 and 128")
        if self.update_frequency < 0.1:
            errors.append("update_frequency must be >= 0.1")
        if self.max_tracked_players < 1:
            errors.append("max_tracked_players must be >= 1")
        if self.min_observation_confidence < 0 or self.min_observation_confidence > 1:
            errors.append("min_observation_confidence must be between 0 and 1")
        return len(errors) == 0, errors


@dataclass
class EvolutionaryGoalConfig:
    """Configuration for the evolutionary goal generation system."""
    # Goal weights
    survival_weight: float = 0.3
    exploration_weight: float = 0.25
    social_learning_weight: float = 0.2
    skill_development_weight: float = 0.15
    creativity_weight: float = 0.1
    
    # Survival thresholds
    hunger_threshold: float = 8.0
    health_threshold: float = 10.0
    shelter_priority_at_night: bool = True
    
    # Exploration settings
    novelty_seeking_radius: int = 100
    revisit_cooldown: float = 300.0
    biome_diversity_bonus: float = 0.2
    
    # Evolution parameters
    goal_mutation_rate: float = 0.1
    fitness_memory_size: int = 100
    adaptation_rate: float = 0.05
    
    # Social learning
    observe_before_imitate: int = 3
    success_imitation_boost: float = 0.3
    
    # Goal chaining
    enable_goal_chaining: bool = True
    max_chain_length: int = 5
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        weights = [self.survival_weight, self.exploration_weight, 
                  self.social_learning_weight, self.skill_development_weight,
                  self.creativity_weight]
        for i, w in enumerate(weights):
            if not 0 <= w <= 1:
                errors.append(f"Weight {i} must be between 0 and 1")
        if self.hunger_threshold < 0 or self.hunger_threshold > 20:
            errors.append("hunger_threshold must be between 0 and 20")
        if self.health_threshold < 0 or self.health_threshold > 20:
            errors.append("health_threshold must be between 0 and 20")
        return len(errors) == 0, errors


@dataclass
class HumanBehaviorConfig:
    """Configuration for human-like behavior patterns."""
    # Movement
    path_deviation_chance: float = 0.15
    path_deviation_amount: float = 2.0
    look_around_frequency: float = 0.2
    
    # Pauses
    thinking_pause_chance: float = 0.1
    thinking_pause_duration: tuple = (0.5, 2.0)
    decision_hesitation: float = 0.05
    
    # Mistakes
    mistake_chance: float = 0.05
    suboptimal_choice_chance: float = 0.1
    recovery_from_mistake: float = 0.8
    
    # Camera movement
    natural_head_movement: bool = True
    look_at_interesting_things: bool = True
    head_turn_smoothness: float = 0.3
    
    # Pacing
    action_speed_variation: float = 0.2
    break_between_tasks: bool = True
    break_duration: tuple = (1.0, 5.0)
    
    # Fatigue system
    enable_fatigue: bool = True
    fatigue_recovery_rate: float = 0.1
    max_continuous_activity: float = 600.0  # 10 minutes
    
    # Attention system
    enable_attention_system: bool = True
    base_attention_span: float = 120.0  # 2 minutes
    
    # Emotional responses
    enable_emotions: bool = True
    emotion_decay_rate: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        for field_name in ['path_deviation_chance', 'look_around_frequency',
                          'thinking_pause_chance', 'decision_hesitation',
                          'mistake_chance', 'suboptimal_choice_chance']:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                errors.append(f"{field_name} must be between 0 and 1")
        return len(errors) == 0, errors


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    # Caching
    enable_llm_cache: bool = True
    cache_max_entries: int = 500
    cache_max_size_mb: float = 50.0
    
    # Batch processing
    batch_processing: bool = True
    batch_size: int = 10
    batch_interval: float = 1.0
    
    # Async processing
    enable_async: bool = True
    max_async_workers: int = 4
    
    # Memory management
    memory_limit_percent: float = 80.0
    cleanup_interval: float = 60.0
    
    # Linux optimizations
    use_linux_optimizations: bool = True
    malloc_arena_max: int = 2
    gc_threshold: Tuple[int, int, int] = (700, 10, 10)
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.cache_max_entries < 10:
            errors.append("cache_max_entries should be >= 10")
        if self.memory_limit_percent < 50 or self.memory_limit_percent > 95:
            errors.append("memory_limit_percent should be between 50 and 95")
        if self.max_async_workers < 1 or self.max_async_workers > 16:
            errors.append("max_async_workers should be between 1 and 16")
        return len(errors) == 0, errors


@dataclass
class SkillConfig:
    """Skill system configuration."""
    # Retrieval
    retrieval_top_k: int = 5
    
    # Difficulty
    enable_difficulty_tracking: bool = True
    auto_adjust_difficulty: bool = True
    
    # Prerequisites
    enable_prerequisites: bool = True
    enforce_prerequisites: bool = False  # Soft vs hard enforcement
    
    # Versioning
    enable_versioning: bool = True
    max_versions_kept: int = 5
    
    # Composition
    enable_composition: bool = True
    max_composition_depth: int = 3
    
    # Success tracking
    track_success_rate: bool = True
    success_history_length: int = 100
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if self.retrieval_top_k < 1 or self.retrieval_top_k > 20:
            errors.append("retrieval_top_k should be between 1 and 20")
        return len(errors) == 0, errors


@dataclass
class DebugConfig:
    """Debug and logging configuration."""
    debug_mode: bool = False
    verbose_logging: bool = False
    log_llm_calls: bool = False
    log_observations: bool = True
    log_goal_evolution: bool = True
    log_behavior_decisions: bool = False
    log_skill_usage: bool = False
    log_performance_metrics: bool = True
    
    # Log file settings
    log_file: Optional[str] = None
    log_max_size_mb: float = 10.0
    log_backup_count: int = 5
    
    # Performance monitoring
    enable_profiling: bool = False
    profile_interval: float = 60.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        return True, []


@dataclass
class EvolvedConfig:
    """Main configuration for Voyager Evolved."""
    # Sub-configurations
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    evolutionary_goals: EvolutionaryGoalConfig = field(default_factory=EvolutionaryGoalConfig)
    human_behavior: HumanBehaviorConfig = field(default_factory=HumanBehaviorConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    # Feature toggles
    enable_player_observation: bool = True
    enable_evolutionary_goals: bool = True
    enable_human_behavior: bool = True
    enable_observational_learning: bool = True
    
    # Persistence
    save_personality_state: bool = True
    save_learned_behaviors: bool = True
    checkpoint_interval: int = 50
    
    # Performance profile
    profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all configuration."""
        all_errors = []
        
        configs = [
            ("personality", self.personality),
            ("observation", self.observation),
            ("evolutionary_goals", self.evolutionary_goals),
            ("human_behavior", self.human_behavior),
            ("performance", self.performance),
            ("skill", self.skill),
            ("debug", self.debug)
        ]
        
        for name, config in configs:
            valid, errors = config.validate()
            for error in errors:
                all_errors.append(f"{name}: {error}")
        
        return len(all_errors) == 0, all_errors
    
    def apply_profile(self, profile: PerformanceProfile = None):
        """Apply a performance profile."""
        if profile:
            self.profile = profile
        
        if self.profile == PerformanceProfile.FAST:
            # Optimize for speed
            self.observation.update_frequency = 2.0
            self.observation.max_tracked_players = 5
            self.observation.enable_pattern_recognition = False
            self.evolutionary_goals.fitness_memory_size = 50
            self.human_behavior.enable_fatigue = False
            self.human_behavior.enable_attention_system = False
            self.performance.cache_max_entries = 1000
            self.performance.batch_size = 20
            self.skill.retrieval_top_k = 3
            
        elif self.profile == PerformanceProfile.QUALITY:
            # Optimize for quality
            self.observation.update_frequency = 0.5
            self.observation.max_tracked_players = 20
            self.observation.max_observation_memory = 2000
            self.evolutionary_goals.fitness_memory_size = 200
            self.human_behavior.thinking_pause_chance = 0.15
            self.human_behavior.look_around_frequency = 0.3
            self.performance.cache_max_entries = 250
            self.skill.retrieval_top_k = 10
            
        # BALANCED uses default values
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        data = self._to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _to_dict(self) -> Dict:
        """Convert config to dictionary."""
        data = asdict(self)
        data['profile'] = self.profile.value
        # Convert tuples to lists for JSON
        data['human_behavior']['thinking_pause_duration'] = list(self.human_behavior.thinking_pause_duration)
        data['human_behavior']['break_duration'] = list(self.human_behavior.break_duration)
        data['performance']['gc_threshold'] = list(self.performance.gc_threshold)
        return data
    
    @classmethod
    def load(cls, path: str) -> 'EvolvedConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load sub-configs
        if 'personality' in data:
            config.personality = PersonalityConfig(**data['personality'])
        if 'observation' in data:
            config.observation = ObservationConfig(**data['observation'])
        if 'evolutionary_goals' in data:
            config.evolutionary_goals = EvolutionaryGoalConfig(**data['evolutionary_goals'])
        if 'human_behavior' in data:
            hb = data['human_behavior']
            # Convert lists back to tuples
            if 'thinking_pause_duration' in hb:
                hb['thinking_pause_duration'] = tuple(hb['thinking_pause_duration'])
            if 'break_duration' in hb:
                hb['break_duration'] = tuple(hb['break_duration'])
            config.human_behavior = HumanBehaviorConfig(**hb)
        if 'performance' in data:
            perf = data['performance']
            if 'gc_threshold' in perf:
                perf['gc_threshold'] = tuple(perf['gc_threshold'])
            config.performance = PerformanceConfig(**perf)
        if 'skill' in data:
            config.skill = SkillConfig(**data['skill'])
        if 'debug' in data:
            config.debug = DebugConfig(**data['debug'])
        
        # Load feature toggles
        for key in ['enable_player_observation', 'enable_evolutionary_goals', 
                    'enable_human_behavior', 'enable_observational_learning',
                    'save_personality_state', 'save_learned_behaviors',
                    'checkpoint_interval']:
            if key in data:
                setattr(config, key, data[key])
        
        # Load profile
        if 'profile' in data:
            config.profile = PerformanceProfile(data['profile'])
        
        return config
    
    @classmethod
    def create_fast_profile(cls) -> 'EvolvedConfig':
        """Create a fast performance profile configuration."""
        config = cls()
        config.apply_profile(PerformanceProfile.FAST)
        return config
    
    @classmethod
    def create_quality_profile(cls) -> 'EvolvedConfig':
        """Create a quality-focused profile configuration."""
        config = cls()
        config.apply_profile(PerformanceProfile.QUALITY)
        return config
    
    @classmethod
    def create_debug_profile(cls) -> 'EvolvedConfig':
        """Create a debug-enabled profile configuration."""
        config = cls()
        config.debug.debug_mode = True
        config.debug.verbose_logging = True
        config.debug.log_llm_calls = True
        config.debug.log_observations = True
        config.debug.log_goal_evolution = True
        config.debug.log_behavior_decisions = True
        config.debug.log_skill_usage = True
        config.debug.enable_profiling = True
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
        config.human_behavior.enable_fatigue = True
        config.human_behavior.enable_attention_system = True
        config.human_behavior.enable_emotions = True
        return config
    
    @classmethod
    def create_linux_optimized(cls) -> 'EvolvedConfig':
        """Create a Linux-optimized configuration."""
        config = cls()
        config.performance.use_linux_optimizations = True
        config.performance.enable_async = True
        config.performance.max_async_workers = 4
        config.performance.memory_limit_percent = 75.0
        config.performance.enable_llm_cache = True
        config.performance.batch_processing = True
        return config
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*50)
        print("Voyager Evolved Configuration Summary")
        print("="*50)
        print(f"\nPerformance Profile: {self.profile.value}")
        print(f"\nFeatures Enabled:")
        print(f"  - Player Observation: {self.enable_player_observation}")
        print(f"  - Evolutionary Goals: {self.enable_evolutionary_goals}")
        print(f"  - Human Behavior: {self.enable_human_behavior}")
        print(f"  - Observational Learning: {self.enable_observational_learning}")
        
        print(f"\nPersonality:")
        print(f"  - Curiosity: {self.personality.curiosity}")
        print(f"  - Caution: {self.personality.caution}")
        print(f"  - Creativity: {self.personality.creativity}")
        
        print(f"\nPerformance:")
        print(f"  - LLM Cache: {self.performance.enable_llm_cache}")
        print(f"  - Batch Processing: {self.performance.batch_processing}")
        print(f"  - Async Workers: {self.performance.max_async_workers}")
        
        print(f"\nDebug:")
        print(f"  - Debug Mode: {self.debug.debug_mode}")
        print(f"  - Verbose Logging: {self.debug.verbose_logging}")
        
        # Validate
        valid, errors = self.validate()
        if not valid:
            print(f"\n⚠️  Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"\n✓ Configuration is valid")
        print("="*50 + "\n")
