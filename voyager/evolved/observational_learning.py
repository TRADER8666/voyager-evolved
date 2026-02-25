"""Observational Learning Integration for Voyager Evolved.

This module connects the player observation system to the skill library,
enabling the agent to learn from observed player behaviors.
"""

import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import voyager.utils as U

from .player_observer import PlayerObserver, ObservedBehavior, PlayerActivity


@dataclass
class LearnedStrategy:
    """A strategy learned from observing players."""
    id: str
    activity: str
    source_player: str
    description: str
    tools_used: List[str]
    blocks_involved: List[str]
    observation_count: int
    success_count: int
    avg_confidence: float
    avg_duration: float
    last_observed: float
    times_used: int = 0
    times_succeeded: int = 0
    adapted: bool = False  # Whether it's been adapted from original
    
    def observation_success_rate(self) -> float:
        if self.observation_count == 0:
            return 0.5
        return self.success_count / self.observation_count
    
    def own_success_rate(self) -> float:
        if self.times_used == 0:
            return self.observation_success_rate()  # Use observed rate as prior
        return self.times_succeeded / self.times_used
    
    def priority_score(self) -> float:
        """Calculate priority for using this strategy."""
        # Combine observation frequency, success rate, and recency
        frequency_score = min(1.0, self.observation_count / 10)
        success_score = self.own_success_rate()
        recency_score = 1.0 - min(1.0, (time.time() - self.last_observed) / 3600)
        
        return frequency_score * 0.3 + success_score * 0.5 + recency_score * 0.2
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedStrategy':
        return cls(**data)


@dataclass
class ContextualAdaptation:
    """An adaptation of a learned strategy to a specific context."""
    original_strategy_id: str
    context: Dict[str, Any]  # Biome, time, resources available, etc.
    modifications: Dict[str, Any]  # How to modify the strategy
    success_rate: float = 0.5
    attempts: int = 0


class ObservationalLearningIntegration:
    """Integrates player observations into the skill/behavior library.
    
    Features:
    - Extract strategies from observed behaviors
    - Prioritize frequently-used strategies
    - Adapt observed behaviors to current context
    - Learn from both successes and failures
    """
    
    def __init__(self, observer: PlayerObserver, skill_manager, config, 
                 ckpt_dir: str = "ckpt", resume: bool = False):
        self.observer = observer
        self.skill_manager = skill_manager
        self.config = config
        self.ckpt_dir = ckpt_dir
        
        # Learned strategies
        self.strategies: Dict[str, LearnedStrategy] = {}
        self.strategy_counter = 0
        
        # Context adaptations
        self.adaptations: List[ContextualAdaptation] = []
        
        # Activity to skill mapping
        self.activity_skill_map: Dict[str, List[str]] = defaultdict(list)
        
        # Setup persistence
        U.f_mkdir(f"{ckpt_dir}/observational_learning")
        
        if resume:
            self._load_state()
    
    def _load_state(self):
        """Load saved learning state."""
        try:
            strategies_path = f"{self.ckpt_dir}/observational_learning/strategies.json"
            if U.f_exists(strategies_path):
                data = U.load_json(strategies_path)
                self.strategies = {
                    k: LearnedStrategy.from_dict(v) for k, v in data.items()
                }
                print(f"\033[36mLoaded {len(self.strategies)} learned strategies\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load learning state: {e}\033[0m")
    
    def save_state(self):
        """Save learning state to disk."""
        strategies_data = {k: v.to_dict() for k, v in self.strategies.items()}
        U.dump_json(strategies_data, f"{self.ckpt_dir}/observational_learning/strategies.json")
    
    def process_new_observations(self) -> List[LearnedStrategy]:
        """Process new observations and extract learnable strategies.
        
        Returns:
            List of newly learned or updated strategies
        """
        if not self.observer:
            return []
        
        new_strategies = []
        
        # Get recent observations that haven't been processed
        observations = self.observer.observed_behaviors
        
        # Group by activity
        activity_groups = defaultdict(list)
        for obs in observations[-100:]:  # Process recent observations
            activity_groups[obs.activity].append(obs)
        
        # Extract strategies for each activity
        for activity, obs_list in activity_groups.items():
            strategy = self._extract_strategy(activity, obs_list)
            if strategy:
                existing = self._find_similar_strategy(strategy)
                if existing:
                    self._merge_strategies(existing, strategy)
                else:
                    self.strategies[strategy.id] = strategy
                    new_strategies.append(strategy)
        
        return new_strategies
    
    def _extract_strategy(self, activity: PlayerActivity, 
                         observations: List[ObservedBehavior]) -> Optional[LearnedStrategy]:
        """Extract a strategy from a group of observations."""
        if not observations:
            return None
        
        # Need minimum observations
        min_obs = self.config.evolutionary_goals.observe_before_imitate
        if len(observations) < min_obs:
            return None
        
        # Find common patterns
        tool_counts = defaultdict(int)
        block_counts = defaultdict(int)
        durations = []
        confidences = []
        success_count = 0
        
        source_players = set()
        
        for obs in observations:
            for tool in obs.tools_used:
                if tool:
                    tool_counts[tool] += 1
            for block in obs.blocks_involved:
                if block:
                    block_counts[block] += 1
            durations.append(obs.duration())
            confidences.append(obs.confidence)
            if obs.success:
                success_count += 1
            source_players.add(obs.player_name)
        
        # Get most common tools and blocks
        common_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        common_blocks = sorted(block_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate description
        description = self._generate_strategy_description(activity, common_tools, common_blocks)
        
        self.strategy_counter += 1
        return LearnedStrategy(
            id=f"observed_strategy_{self.strategy_counter}",
            activity=activity.value,
            source_player=random.choice(list(source_players)) if source_players else "unknown",
            description=description,
            tools_used=[t[0] for t in common_tools],
            blocks_involved=[b[0] for b in common_blocks],
            observation_count=len(observations),
            success_count=success_count,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.5,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            last_observed=max(obs.end_time for obs in observations)
        )
    
    def _generate_strategy_description(self, activity: PlayerActivity, 
                                       tools: List[Tuple[str, int]], 
                                       blocks: List[Tuple[str, int]]) -> str:
        """Generate a human-readable description of the strategy."""
        descriptions = {
            PlayerActivity.MINING: "Mine resources",
            PlayerActivity.BUILDING: "Build structures",
            PlayerActivity.CRAFTING: "Craft items",
            PlayerActivity.FARMING: "Farm crops",
            PlayerActivity.FIGHTING: "Combat enemies",
            PlayerActivity.EXPLORING: "Explore the world",
            PlayerActivity.GATHERING: "Gather materials",
        }
        
        base = descriptions.get(activity, activity.value.replace("_", " ").title())
        
        if tools:
            tool_str = tools[0][0].replace("_", " ")
            base += f" using {tool_str}"
        
        if blocks:
            block_str = blocks[0][0].replace("_", " ")
            base += f" with {block_str}"
        
        return base + " (learned from observation)"
    
    def _find_similar_strategy(self, new_strategy: LearnedStrategy) -> Optional[LearnedStrategy]:
        """Find an existing strategy similar to the new one."""
        for existing in self.strategies.values():
            if existing.activity != new_strategy.activity:
                continue
            
            # Check tool overlap
            tool_overlap = len(set(existing.tools_used) & set(new_strategy.tools_used))
            if tool_overlap >= len(existing.tools_used) * 0.5:
                return existing
        
        return None
    
    def _merge_strategies(self, existing: LearnedStrategy, new: LearnedStrategy):
        """Merge a new strategy into an existing one."""
        # Update counts
        existing.observation_count += new.observation_count
        existing.success_count += new.success_count
        
        # Update averages
        total_obs = existing.observation_count + new.observation_count
        existing.avg_confidence = (
            existing.avg_confidence * existing.observation_count + 
            new.avg_confidence * new.observation_count
        ) / total_obs
        
        # Update tools (add new ones)
        for tool in new.tools_used:
            if tool not in existing.tools_used:
                existing.tools_used.append(tool)
        
        # Update last observed
        existing.last_observed = max(existing.last_observed, new.last_observed)
    
    def get_relevant_strategies(self, context: Dict[str, Any], 
                                task_type: Optional[str] = None,
                                top_k: int = 5) -> List[LearnedStrategy]:
        """Get strategies relevant to the current context and task.
        
        Args:
            context: Current game context (biome, time, resources, etc.)
            task_type: Optional specific task type to filter for
            top_k: Number of strategies to return
            
        Returns:
            List of relevant strategies, sorted by priority
        """
        relevant = []
        
        for strategy in self.strategies.values():
            # Filter by task type if specified
            if task_type:
                if not self._strategy_matches_task(strategy, task_type):
                    continue
            
            # Calculate relevance score
            relevance = strategy.priority_score()
            
            # Boost for context match
            context_boost = self._calculate_context_boost(strategy, context)
            relevance *= (1 + context_boost)
            
            relevant.append((relevance, strategy))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[0], reverse=True)
        
        return [s for _, s in relevant[:top_k]]
    
    def _strategy_matches_task(self, strategy: LearnedStrategy, task_type: str) -> bool:
        """Check if a strategy is relevant for a task type."""
        task_lower = task_type.lower()
        activity = strategy.activity.lower()
        
        # Direct match
        if activity in task_lower or task_lower in activity:
            return True
        
        # Common mappings
        mappings = {
            "mining": ["mine", "dig", "ore", "coal", "iron", "diamond"],
            "building": ["build", "place", "construct", "house", "shelter"],
            "crafting": ["craft", "make", "create"],
            "farming": ["farm", "plant", "harvest", "crop", "wheat"],
            "gathering": ["collect", "gather", "get", "obtain"],
            "exploring": ["explore", "find", "search", "discover"]
        }
        
        if activity in mappings:
            return any(keyword in task_lower for keyword in mappings[activity])
        
        return False
    
    def _calculate_context_boost(self, strategy: LearnedStrategy, context: Dict) -> float:
        """Calculate boost based on context match."""
        boost = 0.0
        
        # Check tool availability
        if "inventory" in context:
            inventory_items = [item.lower() for item in context["inventory"].keys()]
            for tool in strategy.tools_used:
                if any(tool.lower() in item for item in inventory_items):
                    boost += 0.1
        
        # Check if relevant blocks are nearby
        if "nearby_blocks" in context:
            nearby = [b.lower() for b in context["nearby_blocks"]]
            for block in strategy.blocks_involved:
                if any(block.lower() in b for b in nearby):
                    boost += 0.05
        
        return min(boost, 0.5)  # Cap boost
    
    def adapt_strategy_to_context(self, strategy: LearnedStrategy, 
                                  context: Dict[str, Any]) -> LearnedStrategy:
        """Adapt a learned strategy to the current context.
        
        Args:
            strategy: Strategy to adapt
            context: Current game context
            
        Returns:
            Adapted strategy
        """
        adapted = LearnedStrategy(
            id=f"{strategy.id}_adapted",
            activity=strategy.activity,
            source_player=strategy.source_player,
            description=strategy.description + " (adapted)",
            tools_used=list(strategy.tools_used),
            blocks_involved=list(strategy.blocks_involved),
            observation_count=strategy.observation_count,
            success_count=strategy.success_count,
            avg_confidence=strategy.avg_confidence,
            avg_duration=strategy.avg_duration,
            last_observed=strategy.last_observed,
            times_used=0,
            times_succeeded=0,
            adapted=True
        )
        
        # Adapt tools based on what's available
        if "inventory" in context:
            inventory_items = list(context["inventory"].keys())
            adapted.tools_used = self._substitute_tools(adapted.tools_used, inventory_items)
        
        return adapted
    
    def _substitute_tools(self, needed_tools: List[str], available_items: List[str]) -> List[str]:
        """Substitute unavailable tools with alternatives."""
        result = []
        
        # Tool tier mapping
        tiers = ["wooden", "stone", "iron", "golden", "diamond"]
        tool_types = ["pickaxe", "axe", "sword", "shovel", "hoe"]
        
        for tool in needed_tools:
            tool_lower = tool.lower()
            
            # Find tool type
            tool_type = None
            for tt in tool_types:
                if tt in tool_lower:
                    tool_type = tt
                    break
            
            if tool_type:
                # Try to find an available tool of the same type
                found = False
                for item in available_items:
                    if tool_type in item.lower():
                        result.append(item)
                        found = True
                        break
                
                if not found:
                    result.append(tool)  # Keep original as fallback
            else:
                result.append(tool)
        
        return result
    
    def record_strategy_usage(self, strategy_id: str, success: bool):
        """Record the result of using a learned strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            strategy.times_used += 1
            if success:
                strategy.times_succeeded += 1
    
    def add_to_skill_library(self, strategy: LearnedStrategy, skill_code: str):
        """Add a learned strategy to the skill library.
        
        Args:
            strategy: Strategy to add
            skill_code: Generated code for the strategy
        """
        if not self.skill_manager:
            return
        
        # Create skill info
        info = {
            "task": strategy.description,
            "program_name": f"observed_{strategy.activity}_{strategy.id}",
            "program_code": skill_code,
            "conversations": [],
            "success": True
        }
        
        # Add to skill manager
        self.skill_manager.add_new_skill(info)
        
        # Map activity to skill
        self.activity_skill_map[strategy.activity].append(info["program_name"])
    
    def get_skills_for_activity(self, activity: str) -> List[str]:
        """Get skill names learned for a specific activity."""
        return self.activity_skill_map.get(activity, [])
    
    def get_learning_summary(self) -> Dict:
        """Get summary of observational learning progress."""
        activity_counts = defaultdict(int)
        total_observations = 0
        total_own_uses = 0
        
        for strategy in self.strategies.values():
            activity_counts[strategy.activity] += 1
            total_observations += strategy.observation_count
            total_own_uses += strategy.times_used
        
        return {
            "total_strategies_learned": len(self.strategies),
            "total_observations_processed": total_observations,
            "total_strategy_uses": total_own_uses,
            "strategies_by_activity": dict(activity_counts),
            "top_strategies": [
                {
                    "id": s.id,
                    "activity": s.activity,
                    "description": s.description,
                    "own_success_rate": s.own_success_rate(),
                    "times_used": s.times_used
                }
                for s in sorted(self.strategies.values(), 
                               key=lambda x: x.priority_score(), 
                               reverse=True)[:5]
            ]
        }
