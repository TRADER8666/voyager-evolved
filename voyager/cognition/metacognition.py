"""
Meta-Cognition System for Voyager Evolved

Implements higher-order cognitive processes:
- "Thinking about thinking"
- Self-monitoring and self-regulation

Based on Flavell's metacognition theory and Nelson & Narens' framework:
- Metacognitive knowledge (knowing what you know)
- Metacognitive regulation (controlling cognitive processes)
- Metacognitive experiences (feelings of knowing)

Features:
1. Self-Awareness of Knowledge Gaps
2. Confidence Estimation
3. Strategy Selection & Switching
4. Learning from Mistakes
5. Cognitive Load Monitoring
6. Uncertainty Quantification
"""

import time
import threading
import json
import os
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Callable, Set
from collections import defaultdict, deque


# ============================================================================
# COGNITIVE STATES AND STRATEGIES
# ============================================================================

class CognitiveStrategy(Enum):
    """Available cognitive/problem-solving strategies."""
    
    # Exploration strategies
    EXPLORE_RANDOM = auto()     # Random exploration
    EXPLORE_SYSTEMATIC = auto() # Systematic coverage
    EXPLORE_CURIOSITY = auto()  # Novelty-driven
    
    # Problem-solving strategies
    TRIAL_ERROR = auto()        # Try things and learn
    PLANNING = auto()           # Think ahead, plan steps
    ANALOGY = auto()            # Use similar past experiences
    DECOMPOSITION = auto()      # Break into subproblems
    
    # Learning strategies
    OBSERVE_LEARN = auto()      # Watch and learn
    ASK_HELP = auto()           # Seek assistance
    PRACTICE = auto()           # Repetition for mastery
    
    # Efficiency strategies
    SATISFICE = auto()          # Good enough solution
    OPTIMIZE = auto()           # Best possible solution
    
    # Social strategies
    COOPERATE = auto()          # Work with others
    COMPETE = auto()            # Outperform others
    IMITATE = auto()            # Copy successful behaviors


class ConfidenceLevel(Enum):
    """Discrete confidence levels."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class KnowledgeGap:
    """Represents a gap in knowledge or skill."""
    id: str
    domain: str  # e.g., "crafting", "combat", "navigation"
    description: str
    importance: float = 0.5  # How important to fill this gap
    identified_at: float = field(default_factory=time.time)
    attempts_to_fill: int = 0
    related_failures: List[str] = field(default_factory=list)
    
    @property
    def priority(self) -> float:
        """Calculate priority for addressing this gap."""
        age_hours = (time.time() - self.identified_at) / 3600
        urgency = min(1.0, age_hours / 24)  # More urgent over time
        return self.importance * 0.6 + urgency * 0.4


@dataclass
class StrategyRecord:
    """Record of strategy performance."""
    strategy: CognitiveStrategy
    context: str
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0
    last_used: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    @property
    def avg_time_ms(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0
        return self.total_time_ms / total


@dataclass
class Mistake:
    """Record of a mistake for learning."""
    id: str
    task: str
    description: str
    cause: str
    context: Dict[str, Any]
    occurred_at: float = field(default_factory=time.time)
    lesson_learned: Optional[str] = None
    prevention_strategy: Optional[str] = None
    recurrence_count: int = 1


@dataclass
class CognitiveState:
    """Current cognitive state of the agent."""
    mental_load: float = 0.0  # 0-1, cognitive load
    fatigue: float = 0.0  # 0-1, mental fatigue
    confidence: float = 0.5  # Overall confidence
    uncertainty: float = 0.5  # Overall uncertainty
    focus_level: float = 1.0  # 0-1, attention focus
    
    active_strategy: Optional[CognitiveStrategy] = None
    strategy_start_time: float = field(default_factory=time.time)
    
    def update_load(self, task_complexity: float, working_memory_usage: float):
        """Update cognitive load estimate."""
        self.mental_load = min(1.0, task_complexity * 0.5 + working_memory_usage * 0.5)
    
    def apply_fatigue(self, elapsed_seconds: float, activity_intensity: float = 0.5):
        """Accumulate mental fatigue."""
        fatigue_rate = activity_intensity * 0.0001  # Per second
        self.fatigue = min(1.0, self.fatigue + elapsed_seconds * fatigue_rate)
    
    def rest(self, duration_seconds: float):
        """Recover from fatigue."""
        recovery_rate = 0.0002  # Per second
        self.fatigue = max(0.0, self.fatigue - duration_seconds * recovery_rate)


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrates confidence estimates based on actual outcomes.
    
    Implements:
    - Tracking predicted vs actual success
    - Adjusting confidence based on calibration
    - Over/under-confidence detection
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # Predictions and outcomes
        self.predictions: deque = deque(maxlen=history_size)
        
        # Calibration by confidence bucket
        self.buckets: Dict[str, Dict[str, int]] = {
            "0.0-0.2": {"predicted": 0, "actual": 0, "count": 0},
            "0.2-0.4": {"predicted": 0, "actual": 0, "count": 0},
            "0.4-0.6": {"predicted": 0, "actual": 0, "count": 0},
            "0.6-0.8": {"predicted": 0, "actual": 0, "count": 0},
            "0.8-1.0": {"predicted": 0, "actual": 0, "count": 0},
        }
        
        # Running calibration error
        self.calibration_error: float = 0.0
    
    def _get_bucket(self, confidence: float) -> str:
        """Get bucket key for a confidence value."""
        if confidence < 0.2:
            return "0.0-0.2"
        elif confidence < 0.4:
            return "0.2-0.4"
        elif confidence < 0.6:
            return "0.4-0.6"
        elif confidence < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"
    
    def record_prediction(self, task: str, confidence: float) -> str:
        """Record a confidence prediction."""
        pred_id = f"pred_{len(self.predictions)}_{int(time.time()*1000)}"
        self.predictions.append({
            "id": pred_id,
            "task": task,
            "confidence": confidence,
            "outcome": None,
            "time": time.time()
        })
        return pred_id
    
    def record_outcome(self, pred_id: str, success: bool):
        """Record the actual outcome of a prediction."""
        for pred in self.predictions:
            if pred["id"] == pred_id:
                pred["outcome"] = success
                
                # Update calibration
                bucket = self._get_bucket(pred["confidence"])
                self.buckets[bucket]["count"] += 1
                self.buckets[bucket]["predicted"] += pred["confidence"]
                if success:
                    self.buckets[bucket]["actual"] += 1
                
                self._update_calibration_error()
                break
    
    def _update_calibration_error(self):
        """Calculate overall calibration error."""
        total_error = 0.0
        total_weight = 0
        
        for bucket, data in self.buckets.items():
            if data["count"] > 0:
                expected = data["predicted"] / data["count"]
                actual = data["actual"] / data["count"]
                error = abs(expected - actual)
                total_error += error * data["count"]
                total_weight += data["count"]
        
        if total_weight > 0:
            self.calibration_error = total_error / total_weight
    
    def adjust_confidence(self, raw_confidence: float) -> float:
        """Adjust confidence based on calibration history."""
        # If well-calibrated, return as-is
        if self.calibration_error < 0.1:
            return raw_confidence
        
        # Check if over or under confident in this bucket
        bucket = self._get_bucket(raw_confidence)
        data = self.buckets[bucket]
        
        if data["count"] < 5:
            return raw_confidence
        
        expected = data["predicted"] / data["count"]
        actual = data["actual"] / data["count"]
        
        if expected > actual:
            # Over-confident: reduce confidence
            adjustment = (expected - actual) * 0.5
            return max(0.0, raw_confidence - adjustment)
        else:
            # Under-confident: increase confidence
            adjustment = (actual - expected) * 0.5
            return min(1.0, raw_confidence + adjustment)
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return {
            "overall_error": f"{self.calibration_error:.2%}",
            "buckets": {
                bucket: {
                    "count": data["count"],
                    "avg_predicted": data["predicted"] / max(1, data["count"]),
                    "actual_success_rate": data["actual"] / max(1, data["count"])
                }
                for bucket, data in self.buckets.items()
            }
        }


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyQuantifier:
    """
    Quantifies different types of uncertainty.
    
    Types:
    - Aleatoric: Inherent randomness (can't be reduced)
    - Epistemic: Due to lack of knowledge (can be reduced)
    - Model: Due to model limitations
    """
    
    def __init__(self):
        # Track uncertainty sources
        self.knowledge_uncertainty: Dict[str, float] = {}  # domain -> uncertainty
        self.environment_uncertainty: float = 0.5
        self.action_uncertainty: Dict[str, float] = {}  # action -> uncertainty
    
    def estimate_task_uncertainty(
        self,
        task: str,
        domain: str,
        familiarity: float,  # 0-1, how familiar with this task
        environment_stability: float,  # 0-1, how stable/predictable
        past_success_rate: float  # Historical success rate
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate uncertainty for a task.
        
        Returns:
            (total_uncertainty, breakdown_by_type)
        """
        # Epistemic uncertainty (from lack of knowledge)
        knowledge_unc = self.knowledge_uncertainty.get(domain, 0.5)
        epistemic = (1 - familiarity) * 0.5 + knowledge_unc * 0.5
        
        # Aleatoric uncertainty (inherent randomness)
        aleatoric = (1 - environment_stability) * 0.3 + (1 - past_success_rate) * 0.3
        
        # Model uncertainty (from LLM/prediction limitations)
        model_unc = 0.2  # Base model uncertainty
        
        # Total uncertainty
        total = epistemic * 0.4 + aleatoric * 0.35 + model_unc * 0.25
        
        return total, {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "model": model_unc
        }
    
    def update_knowledge_uncertainty(self, domain: str, success: bool):
        """Update uncertainty for a domain based on outcome."""
        current = self.knowledge_uncertainty.get(domain, 0.5)
        if success:
            # Reduce uncertainty
            self.knowledge_uncertainty[domain] = current * 0.95
        else:
            # Increase uncertainty
            self.knowledge_uncertainty[domain] = min(1.0, current * 1.05)
    
    def get_most_uncertain_domains(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get domains with highest uncertainty."""
        sorted_domains = sorted(
            self.knowledge_uncertainty.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_domains[:n]


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """
    Selects appropriate cognitive strategies based on context.
    
    Uses:
    - Task characteristics
    - Past strategy performance
    - Current cognitive state
    - Time constraints
    """
    
    def __init__(self):
        # Strategy performance by context
        self.strategy_records: Dict[str, Dict[CognitiveStrategy, StrategyRecord]] = defaultdict(dict)
        
        # Default strategy preferences by task type
        self.default_strategies: Dict[str, List[CognitiveStrategy]] = {
            "exploration": [
                CognitiveStrategy.EXPLORE_CURIOSITY,
                CognitiveStrategy.EXPLORE_SYSTEMATIC
            ],
            "crafting": [
                CognitiveStrategy.PLANNING,
                CognitiveStrategy.ANALOGY
            ],
            "combat": [
                CognitiveStrategy.TRIAL_ERROR,
                CognitiveStrategy.PRACTICE
            ],
            "building": [
                CognitiveStrategy.PLANNING,
                CognitiveStrategy.DECOMPOSITION
            ],
            "social": [
                CognitiveStrategy.OBSERVE_LEARN,
                CognitiveStrategy.IMITATE,
                CognitiveStrategy.COOPERATE
            ],
            "unknown": [
                CognitiveStrategy.TRIAL_ERROR,
                CognitiveStrategy.OBSERVE_LEARN
            ]
        }
    
    def select_strategy(
        self,
        task_type: str,
        complexity: float,  # 0-1
        time_pressure: float,  # 0-1
        confidence: float,  # Current confidence
        cognitive_load: float  # Current load
    ) -> CognitiveStrategy:
        """Select the best strategy for the situation."""
        
        # Get candidate strategies
        candidates = self.default_strategies.get(
            task_type, 
            self.default_strategies["unknown"]
        )
        
        # Score each candidate
        scored = []
        for strategy in candidates:
            score = self._score_strategy(
                strategy, task_type, complexity, 
                time_pressure, confidence, cognitive_load
            )
            scored.append((score, strategy))
        
        # Select best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    def _score_strategy(
        self,
        strategy: CognitiveStrategy,
        task_type: str,
        complexity: float,
        time_pressure: float,
        confidence: float,
        cognitive_load: float
    ) -> float:
        """Score a strategy for the current situation."""
        score = 0.5  # Base score
        
        # Historical performance
        if task_type in self.strategy_records:
            record = self.strategy_records[task_type].get(strategy)
            if record:
                score += record.success_rate * 0.3
                
                # Time efficiency if under pressure
                if time_pressure > 0.5 and record.avg_time_ms > 0:
                    time_score = 1.0 / (1 + record.avg_time_ms / 10000)
                    score += time_score * time_pressure * 0.2
        
        # Strategy-specific adjustments
        if strategy == CognitiveStrategy.PLANNING:
            # Good for complex, low-pressure situations
            score += complexity * 0.2
            score -= time_pressure * 0.2
            
        elif strategy == CognitiveStrategy.TRIAL_ERROR:
            # Good for low confidence, simple tasks
            score += (1 - confidence) * 0.2
            score += (1 - complexity) * 0.1
            
        elif strategy == CognitiveStrategy.SATISFICE:
            # Good under time pressure
            score += time_pressure * 0.3
            
        elif strategy == CognitiveStrategy.DECOMPOSITION:
            # Good for complex tasks
            score += complexity * 0.3
            
        elif strategy == CognitiveStrategy.OBSERVE_LEARN:
            # Good when uncertain
            score += (1 - confidence) * 0.3
        
        # Reduce score if cognitive load is high and strategy is demanding
        if cognitive_load > 0.7:
            if strategy in [CognitiveStrategy.PLANNING, CognitiveStrategy.OPTIMIZE]:
                score -= 0.2
        
        return score
    
    def record_strategy_outcome(
        self,
        strategy: CognitiveStrategy,
        task_type: str,
        success: bool,
        duration_ms: float
    ):
        """Record the outcome of using a strategy."""
        if task_type not in self.strategy_records:
            self.strategy_records[task_type] = {}
        
        if strategy not in self.strategy_records[task_type]:
            self.strategy_records[task_type][strategy] = StrategyRecord(
                strategy=strategy,
                context=task_type
            )
        
        record = self.strategy_records[task_type][strategy]
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1
        record.total_time_ms += duration_ms
        record.last_used = time.time()
    
    def should_switch_strategy(
        self,
        current_strategy: CognitiveStrategy,
        elapsed_ms: float,
        progress: float,  # 0-1, progress toward goal
        expected_total_ms: float
    ) -> bool:
        """Determine if we should switch strategies."""
        # Check if making adequate progress
        expected_progress = elapsed_ms / max(1, expected_total_ms)
        progress_deficit = expected_progress - progress
        
        # Switch if significantly behind
        if progress_deficit > 0.3:
            return True
        
        # Switch if taking too long
        if elapsed_ms > expected_total_ms * 2:
            return True
        
        return False


# ============================================================================
# METACOGNITION SYSTEM - Main Class
# ============================================================================

class MetaCognition:
    """
    Main metacognition system that monitors and regulates cognitive processes.
    
    Implements:
    - Self-awareness of knowledge and abilities
    - Confidence calibration
    - Strategy selection
    - Learning from mistakes
    - Cognitive load monitoring
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_mistakes: int = 500,
        max_knowledge_gaps: int = 100
    ):
        self.persist_path = persist_path
        self.max_mistakes = max_mistakes
        self.max_knowledge_gaps = max_knowledge_gaps
        
        # Core components
        self.state = CognitiveState()
        self.calibrator = ConfidenceCalibrator()
        self.uncertainty = UncertaintyQuantifier()
        self.strategy_selector = StrategySelector()
        
        # Knowledge gaps
        self.knowledge_gaps: Dict[str, KnowledgeGap] = {}
        self._gap_count = 0
        
        # Mistakes and lessons
        self.mistakes: Dict[str, Mistake] = {}
        self._mistake_count = 0
        self.lessons: List[str] = []
        
        # Self-model
        self.known_abilities: Dict[str, float] = {}  # ability -> proficiency
        self.known_limitations: List[str] = []
        self.strengths: List[str] = []
        self.weaknesses: List[str] = []
        
        # Monitoring
        self.task_history: deque = deque(maxlen=100)
        self._monitoring_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        if self.persist_path and os.path.exists(self.persist_path):
            self._load()
    
    # ========================================================================
    # SELF-AWARENESS
    # ========================================================================
    
    def identify_knowledge_gap(
        self,
        domain: str,
        description: str,
        importance: float = 0.5,
        related_failure: Optional[str] = None
    ) -> str:
        """Identify a gap in knowledge or ability."""
        with self._lock:
            # Check if gap already exists
            for gap_id, gap in self.knowledge_gaps.items():
                if gap.domain == domain and gap.description == description:
                    gap.importance = max(gap.importance, importance)
                    if related_failure:
                        gap.related_failures.append(related_failure)
                    return gap_id
            
            # Create new gap
            self._gap_count += 1
            gap_id = f"gap_{self._gap_count}"
            
            self.knowledge_gaps[gap_id] = KnowledgeGap(
                id=gap_id,
                domain=domain,
                description=description,
                importance=importance,
                related_failures=[related_failure] if related_failure else []
            )
            
            # Update uncertainty
            self.uncertainty.knowledge_uncertainty[domain] = min(
                1.0,
                self.uncertainty.knowledge_uncertainty.get(domain, 0.5) + 0.1
            )
            
            # Cleanup if over limit
            if len(self.knowledge_gaps) > self.max_knowledge_gaps:
                self._cleanup_gaps()
            
            return gap_id
    
    def fill_knowledge_gap(self, gap_id: str, success: bool):
        """Record attempt to fill a knowledge gap."""
        with self._lock:
            if gap_id not in self.knowledge_gaps:
                return
            
            gap = self.knowledge_gaps[gap_id]
            gap.attempts_to_fill += 1
            
            if success:
                # Remove gap
                del self.knowledge_gaps[gap_id]
                
                # Update uncertainty
                self.uncertainty.update_knowledge_uncertainty(gap.domain, True)
    
    def get_priority_gaps(self, n: int = 5) -> List[KnowledgeGap]:
        """Get highest priority knowledge gaps."""
        with self._lock:
            gaps = list(self.knowledge_gaps.values())
            gaps.sort(key=lambda g: g.priority, reverse=True)
            return gaps[:n]
    
    def update_ability(self, ability: str, outcome: bool, difficulty: float = 0.5):
        """Update self-model of abilities based on outcome."""
        with self._lock:
            current = self.known_abilities.get(ability, 0.5)
            
            if outcome:
                # Successful: increase proficiency, more for difficult tasks
                increase = 0.05 + difficulty * 0.05
                self.known_abilities[ability] = min(1.0, current + increase)
            else:
                # Failed: decrease proficiency
                decrease = 0.03 + (1 - difficulty) * 0.02
                self.known_abilities[ability] = max(0.0, current - decrease)
            
            # Update strengths/weaknesses
            self._update_strengths_weaknesses()
    
    def _update_strengths_weaknesses(self):
        """Update lists of strengths and weaknesses."""
        if not self.known_abilities:
            return
        
        sorted_abilities = sorted(
            self.known_abilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top 3 are strengths
        self.strengths = [a[0] for a in sorted_abilities[:3] if a[1] >= 0.6]
        
        # Bottom 3 are weaknesses
        self.weaknesses = [a[0] for a in sorted_abilities[-3:] if a[1] <= 0.4]
    
    def get_ability_confidence(self, ability: str) -> float:
        """Get confidence in a specific ability."""
        with self._lock:
            base_confidence = self.known_abilities.get(ability, 0.5)
            return self.calibrator.adjust_confidence(base_confidence)
    
    # ========================================================================
    # CONFIDENCE ESTIMATION
    # ========================================================================
    
    def estimate_confidence(
        self,
        task: str,
        task_type: str,
        related_abilities: List[str],
        complexity: float = 0.5
    ) -> Tuple[float, str]:
        """
        Estimate confidence for a task.
        
        Returns:
            (confidence, explanation)
        """
        with self._lock:
            # Base confidence from related abilities
            if related_abilities:
                ability_scores = [
                    self.known_abilities.get(a, 0.5) 
                    for a in related_abilities
                ]
                base_confidence = sum(ability_scores) / len(ability_scores)
            else:
                base_confidence = 0.5
            
            # Adjust for complexity
            complexity_adjustment = (1 - complexity) * 0.2 - 0.1
            
            # Adjust for cognitive load
            load_adjustment = -self.state.mental_load * 0.1
            
            # Adjust for fatigue
            fatigue_adjustment = -self.state.fatigue * 0.15
            
            # Calculate raw confidence
            raw_confidence = base_confidence + complexity_adjustment + load_adjustment + fatigue_adjustment
            raw_confidence = max(0.0, min(1.0, raw_confidence))
            
            # Calibrate
            calibrated = self.calibrator.adjust_confidence(raw_confidence)
            
            # Generate explanation
            explanation = self._explain_confidence(
                calibrated, base_confidence, complexity,
                self.state.mental_load, self.state.fatigue
            )
            
            # Record prediction
            self.calibrator.record_prediction(task, calibrated)
            
            return calibrated, explanation
    
    def _explain_confidence(
        self,
        confidence: float,
        ability_base: float,
        complexity: float,
        load: float,
        fatigue: float
    ) -> str:
        """Generate explanation for confidence level."""
        parts = []
        
        if confidence >= 0.8:
            parts.append("I'm quite confident about this")
        elif confidence >= 0.6:
            parts.append("I think I can do this")
        elif confidence >= 0.4:
            parts.append("I'm uncertain about this")
        else:
            parts.append("This is challenging for me")
        
        if ability_base < 0.4:
            parts.append("because I lack experience with it")
        
        if complexity > 0.7:
            parts.append("and it's quite complex")
        
        if load > 0.7:
            parts.append("and I'm already processing a lot")
        
        if fatigue > 0.5:
            parts.append("and I'm feeling mentally tired")
        
        return ", ".join(parts) + "."
    
    def record_task_outcome(
        self,
        task: str,
        success: bool,
        pred_id: Optional[str] = None
    ):
        """Record task outcome for calibration."""
        with self._lock:
            if pred_id:
                self.calibrator.record_outcome(pred_id, success)
            
            self.task_history.append({
                "task": task,
                "success": success,
                "time": time.time()
            })
    
    # ========================================================================
    # STRATEGY SELECTION
    # ========================================================================
    
    def select_strategy(
        self,
        task_type: str,
        complexity: float = 0.5,
        time_pressure: float = 0.0
    ) -> CognitiveStrategy:
        """Select cognitive strategy for a task."""
        with self._lock:
            strategy = self.strategy_selector.select_strategy(
                task_type=task_type,
                complexity=complexity,
                time_pressure=time_pressure,
                confidence=self.state.confidence,
                cognitive_load=self.state.mental_load
            )
            
            self.state.active_strategy = strategy
            self.state.strategy_start_time = time.time()
            
            return strategy
    
    def should_switch_strategy(
        self,
        progress: float,
        expected_duration_ms: float
    ) -> Tuple[bool, Optional[CognitiveStrategy]]:
        """Check if strategy should be switched."""
        with self._lock:
            if not self.state.active_strategy:
                return False, None
            
            elapsed = (time.time() - self.state.strategy_start_time) * 1000
            
            should_switch = self.strategy_selector.should_switch_strategy(
                self.state.active_strategy,
                elapsed,
                progress,
                expected_duration_ms
            )
            
            if should_switch:
                # Get alternative strategy
                # (Could be more sophisticated)
                return True, CognitiveStrategy.TRIAL_ERROR
            
            return False, None
    
    def record_strategy_outcome(
        self,
        task_type: str,
        success: bool,
        duration_ms: float
    ):
        """Record outcome of strategy use."""
        with self._lock:
            if self.state.active_strategy:
                self.strategy_selector.record_strategy_outcome(
                    self.state.active_strategy,
                    task_type,
                    success,
                    duration_ms
                )
    
    # ========================================================================
    # LEARNING FROM MISTAKES
    # ========================================================================
    
    def record_mistake(
        self,
        task: str,
        description: str,
        cause: str,
        context: Dict[str, Any]
    ) -> str:
        """Record a mistake for future learning."""
        with self._lock:
            # Check for similar past mistakes
            for mistake in self.mistakes.values():
                if mistake.task == task and mistake.cause == cause:
                    mistake.recurrence_count += 1
                    return mistake.id
            
            self._mistake_count += 1
            mistake_id = f"mistake_{self._mistake_count}"
            
            self.mistakes[mistake_id] = Mistake(
                id=mistake_id,
                task=task,
                description=description,
                cause=cause,
                context=context
            )
            
            # Identify knowledge gap
            self.identify_knowledge_gap(
                domain=task.split()[0] if task else "general",
                description=f"Failed: {description}",
                importance=0.6,
                related_failure=mistake_id
            )
            
            # Cleanup if over limit
            if len(self.mistakes) > self.max_mistakes:
                self._cleanup_mistakes()
            
            return mistake_id
    
    def learn_from_mistake(
        self,
        mistake_id: str,
        lesson: str,
        prevention_strategy: Optional[str] = None
    ):
        """Record what was learned from a mistake."""
        with self._lock:
            if mistake_id not in self.mistakes:
                return
            
            mistake = self.mistakes[mistake_id]
            mistake.lesson_learned = lesson
            mistake.prevention_strategy = prevention_strategy
            
            self.lessons.append(lesson)
    
    def check_for_past_mistakes(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Check for relevant past mistakes and return warnings."""
        with self._lock:
            warnings = []
            
            for mistake in self.mistakes.values():
                if mistake.task == task or task in mistake.description:
                    if mistake.lesson_learned:
                        warnings.append(f"Remember: {mistake.lesson_learned}")
                    elif mistake.prevention_strategy:
                        warnings.append(f"Consider: {mistake.prevention_strategy}")
            
            return warnings
    
    def get_recurring_mistakes(self, min_count: int = 2) -> List[Mistake]:
        """Get mistakes that keep recurring."""
        with self._lock:
            return [
                m for m in self.mistakes.values()
                if m.recurrence_count >= min_count
            ]
    
    # ========================================================================
    # COGNITIVE LOAD MONITORING
    # ========================================================================
    
    def update_cognitive_state(
        self,
        working_memory_items: int,
        working_memory_capacity: int,
        task_complexity: float,
        elapsed_seconds: float
    ):
        """Update current cognitive state."""
        with self._lock:
            # Update load
            wm_usage = working_memory_items / working_memory_capacity
            self.state.update_load(task_complexity, wm_usage)
            
            # Update fatigue
            activity_intensity = (task_complexity + wm_usage) / 2
            self.state.apply_fatigue(elapsed_seconds, activity_intensity)
            
            # Update overall confidence
            recent_success = self._calculate_recent_success_rate()
            self.state.confidence = (
                recent_success * 0.4 + 
                (1 - self.state.mental_load) * 0.3 +
                (1 - self.state.fatigue) * 0.3
            )
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate from recent tasks."""
        if not self.task_history:
            return 0.5
        
        recent = list(self.task_history)[-20:]
        successes = sum(1 for t in recent if t["success"])
        return successes / len(recent)
    
    def needs_rest(self) -> bool:
        """Check if agent needs rest."""
        return self.state.fatigue > 0.8 or self.state.mental_load > 0.9
    
    def rest(self, duration_seconds: float):
        """Rest to recover from fatigue."""
        with self._lock:
            self.state.rest(duration_seconds)
    
    # ========================================================================
    # INTROSPECTION
    # ========================================================================
    
    def get_self_assessment(self) -> str:
        """Generate a self-assessment narrative."""
        with self._lock:
            parts = []
            
            # Overall state
            if self.state.confidence > 0.7:
                parts.append("I'm feeling confident right now.")
            elif self.state.confidence < 0.4:
                parts.append("I'm feeling uncertain about my abilities.")
            
            if self.state.fatigue > 0.6:
                parts.append("I'm getting tired and may make mistakes.")
            
            if self.state.mental_load > 0.7:
                parts.append("I'm processing a lot of information.")
            
            # Strengths and weaknesses
            if self.strengths:
                parts.append(f"My strengths are: {', '.join(self.strengths)}.")
            
            if self.weaknesses:
                parts.append(f"I struggle with: {', '.join(self.weaknesses)}.")
            
            # Knowledge gaps
            gaps = self.get_priority_gaps(3)
            if gaps:
                gap_descs = [g.description for g in gaps]
                parts.append(f"I need to learn: {'; '.join(gap_descs)}.")
            
            # Recent performance
            success_rate = self._calculate_recent_success_rate()
            if success_rate > 0.7:
                parts.append("I've been doing well recently.")
            elif success_rate < 0.4:
                parts.append("I've been struggling lately.")
            
            return " ".join(parts) if parts else "I'm in a neutral state."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metacognition statistics."""
        with self._lock:
            return {
                "cognitive_state": {
                    "mental_load": f"{self.state.mental_load:.2f}",
                    "fatigue": f"{self.state.fatigue:.2f}",
                    "confidence": f"{self.state.confidence:.2f}",
                    "focus": f"{self.state.focus_level:.2f}"
                },
                "knowledge_gaps": len(self.knowledge_gaps),
                "mistakes_recorded": len(self.mistakes),
                "lessons_learned": len(self.lessons),
                "known_abilities": len(self.known_abilities),
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
                "calibration": self.calibrator.get_calibration_report(),
                "recent_success_rate": f"{self._calculate_recent_success_rate():.2%}"
            }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _cleanup_gaps(self):
        """Remove lowest priority gaps."""
        gaps = sorted(
            self.knowledge_gaps.items(),
            key=lambda x: x[1].priority
        )
        to_remove = len(gaps) - int(self.max_knowledge_gaps * 0.8)
        for i in range(to_remove):
            del self.knowledge_gaps[gaps[i][0]]
    
    def _cleanup_mistakes(self):
        """Remove old, non-recurring mistakes."""
        mistakes = sorted(
            self.mistakes.items(),
            key=lambda x: (x[1].recurrence_count, -x[1].occurred_at)
        )
        to_remove = len(mistakes) - int(self.max_mistakes * 0.8)
        for i in range(to_remove):
            del self.mistakes[mistakes[i][0]]
    
    def save(self):
        """Save metacognition data."""
        if not self.persist_path:
            return
        
        with self._lock:
            try:
                data = {
                    "knowledge_gaps": {
                        gid: {
                            "id": g.id,
                            "domain": g.domain,
                            "description": g.description,
                            "importance": g.importance,
                            "attempts": g.attempts_to_fill
                        }
                        for gid, g in self.knowledge_gaps.items()
                    },
                    "mistakes": {
                        mid: {
                            "id": m.id,
                            "task": m.task,
                            "description": m.description,
                            "cause": m.cause,
                            "lesson": m.lesson_learned,
                            "prevention": m.prevention_strategy,
                            "recurrence": m.recurrence_count
                        }
                        for mid, m in self.mistakes.items()
                    },
                    "known_abilities": self.known_abilities,
                    "lessons": self.lessons,
                    "gap_count": self._gap_count,
                    "mistake_count": self._mistake_count
                }
                
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save metacognition: {e}")
    
    def _load(self):
        """Load metacognition data."""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self._gap_count = data.get("gap_count", 0)
            self._mistake_count = data.get("mistake_count", 0)
            self.known_abilities = data.get("known_abilities", {})
            self.lessons = data.get("lessons", [])
            
            for gid, gdata in data.get("knowledge_gaps", {}).items():
                self.knowledge_gaps[gid] = KnowledgeGap(
                    id=gdata["id"],
                    domain=gdata["domain"],
                    description=gdata["description"],
                    importance=gdata.get("importance", 0.5),
                    attempts_to_fill=gdata.get("attempts", 0)
                )
            
            for mid, mdata in data.get("mistakes", {}).items():
                self.mistakes[mid] = Mistake(
                    id=mdata["id"],
                    task=mdata["task"],
                    description=mdata["description"],
                    cause=mdata["cause"],
                    context={},
                    lesson_learned=mdata.get("lesson"),
                    prevention_strategy=mdata.get("prevention"),
                    recurrence_count=mdata.get("recurrence", 1)
                )
            
            self._update_strengths_weaknesses()
            
        except Exception as e:
            print(f"Failed to load metacognition: {e}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_metacog_instance: Optional[MetaCognition] = None
_metacog_lock = threading.Lock()


def get_metacognition(
    persist_path: Optional[str] = None,
    **kwargs
) -> MetaCognition:
    """Get or create the global metacognition instance."""
    global _metacog_instance
    
    with _metacog_lock:
        if _metacog_instance is None:
            _metacog_instance = MetaCognition(
                persist_path=persist_path or os.path.expanduser(
                    "~/.voyager_evolved/metacognition.json"
                ),
                **kwargs
            )
        return _metacog_instance


def reset_metacognition():
    """Reset the global metacognition instance."""
    global _metacog_instance
    
    with _metacog_lock:
        if _metacog_instance:
            _metacog_instance.save()
        _metacog_instance = None
