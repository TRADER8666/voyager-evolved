"""Voyager Evolved - Enhanced Minecraft AI with Human-like Behavior.

Linux-optimized, Ollama-powered autonomous Minecraft agent.

Modules:
- config: Configuration system with profiles and validation
- personality: Dynamic personality engine
- player_observer: Player tracking and pattern recognition
- evolutionary_goals: Priority-based goal generation
- human_behavior: Fatigue, attention, and emotions
- observational_learning: Learning from observed players
- performance: Caching, async processing, optimization
"""

from .config import EvolvedConfig, PerformanceProfile
from .personality import PersonalityEngine
from .player_observer import PlayerObserver, PlayerActivity
from .evolutionary_goals import EvolutionaryGoalSystem, GoalCategory
from .human_behavior import HumanBehaviorSystem, EmotionalState
from .observational_learning import ObservationalLearning
from .voyager_evolved import VoyagerEvolved
from .performance import (
    get_llm_cache,
    get_memory_manager,
    get_perf_monitor,
    get_task_manager,
    get_performance_report,
    optimize_for_linux,
    timed,
    cache_llm_response
)

__all__ = [
    # Main class
    "VoyagerEvolved",
    
    # Configuration
    "EvolvedConfig",
    "PerformanceProfile",
    
    # Core systems
    "PersonalityEngine",
    "PlayerObserver",
    "PlayerActivity",
    "EvolutionaryGoalSystem",
    "GoalCategory",
    "HumanBehaviorSystem",
    "EmotionalState",
    "ObservationalLearning",
    
    # Performance
    "get_llm_cache",
    "get_memory_manager",
    "get_perf_monitor",
    "get_task_manager",
    "get_performance_report",
    "optimize_for_linux",
    "timed",
    "cache_llm_response",
]

__version__ = "2.0.0"
__author__ = "Voyager Evolved Team"
__license__ = "MIT"
