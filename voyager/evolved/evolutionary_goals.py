"""Evolutionary Goal System for Voyager Evolved.

This module replaces/enhances the automatic curriculum with an emergent
goal generation system based on survival instincts, curiosity, social
learning, and evolutionary fitness.
"""

import random
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import voyager.utils as U


class GoalCategory(Enum):
    """Categories of emergent goals."""
    SURVIVAL = "survival"  # Food, health, shelter
    EXPLORATION = "exploration"  # Discovering new areas/items
    SOCIAL_LEARNING = "social_learning"  # Learning from others
    SKILL_DEVELOPMENT = "skill_development"  # Improving abilities
    CREATIVITY = "creativity"  # Novel approaches and building
    RESOURCE_GATHERING = "resource_gathering"  # Collecting materials
    TOOL_PROGRESSION = "tool_progression"  # Better equipment


@dataclass
class Goal:
    """A single emergent goal."""
    id: str
    category: GoalCategory
    description: str
    priority: float  # 0 to 1
    fitness: float  # Historical success rate
    attempts: int
    successes: int
    created_at: float
    last_attempted: float
    context: Dict[str, Any]
    prerequisites: List[str]  # Goal IDs that should be completed first
    derived_from: Optional[str]  # Source (observation, instinct, mutation)
    
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.5  # Neutral prior
        return self.successes / self.attempts
    
    def age(self) -> float:
        return time.time() - self.created_at
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['category'] = self.category.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        data['category'] = GoalCategory(data['category'])
        return cls(**data)


@dataclass
class SurvivalState:
    """Current survival-related state of the agent."""
    health: float = 20.0
    max_health: float = 20.0
    hunger: float = 20.0
    has_shelter: bool = False
    shelter_location: Optional[Tuple[float, float, float]] = None
    time_of_day: str = "day"
    is_safe: bool = True
    nearby_threats: List[str] = field(default_factory=list)
    
    def health_urgency(self) -> float:
        return 1.0 - (self.health / self.max_health)
    
    def hunger_urgency(self) -> float:
        return 1.0 - (self.hunger / 20.0)
    
    def safety_urgency(self) -> float:
        if self.is_safe:
            return 0.0
        return 0.5 + len(self.nearby_threats) * 0.1


@dataclass
class ExplorationMemory:
    """Memory of explored areas and discoveries."""
    visited_chunks: Set[Tuple[int, int]] = field(default_factory=set)
    discovered_biomes: Set[str] = field(default_factory=set)
    found_structures: Set[str] = field(default_factory=set)
    interesting_locations: List[Dict] = field(default_factory=list)
    last_visited: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def novelty_score(self, chunk: Tuple[int, int]) -> float:
        if chunk not in self.visited_chunks:
            return 1.0
        
        last_visit = self.last_visited.get(chunk, 0)
        time_since = time.time() - last_visit
        return min(1.0, time_since / 3600)  # Full novelty after 1 hour


class EvolutionaryGoalSystem:
    """Generates and manages emergent goals based on evolutionary principles.
    
    Goals are generated from:
    1. Survival instincts (food, shelter, safety)
    2. Curiosity-driven exploration
    3. Social learning (observed player behaviors)
    4. Evolutionary fitness (what has worked before)
    5. Personality traits
    """
    
    def __init__(self, config, personality_engine, player_observer, 
                 ckpt_dir: str = "ckpt", resume: bool = False):
        self.config = config.evolutionary_goals
        self.main_config = config
        self.personality = personality_engine
        self.observer = player_observer
        self.ckpt_dir = ckpt_dir
        
        # Goal management
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.failed_goals: List[Goal] = []
        self.goal_history: Dict[str, Goal] = {}
        
        # State tracking
        self.survival_state = SurvivalState()
        self.exploration_memory = ExplorationMemory()
        
        # Fitness tracking for goal types
        self.category_fitness: Dict[str, List[float]] = defaultdict(list)
        self.category_weights: Dict[str, float] = {
            GoalCategory.SURVIVAL.value: self.config.survival_weight,
            GoalCategory.EXPLORATION.value: self.config.exploration_weight,
            GoalCategory.SOCIAL_LEARNING.value: self.config.social_learning_weight,
            GoalCategory.SKILL_DEVELOPMENT.value: self.config.skill_development_weight,
            GoalCategory.CREATIVITY.value: self.config.creativity_weight,
            GoalCategory.RESOURCE_GATHERING.value: 0.15,
            GoalCategory.TOOL_PROGRESSION.value: 0.15
        }
        
        # Goal ID counter
        self.goal_counter = 0
        
        # Setup persistence
        U.f_mkdir(f"{ckpt_dir}/goals")
        
        if resume:
            self._load_state()
    
    def _load_state(self):
        """Load saved goal state."""
        try:
            state_path = f"{self.ckpt_dir}/goals/state.json"
            if U.f_exists(state_path):
                data = U.load_json(state_path)
                
                self.goal_history = {
                    k: Goal.from_dict(v) for k, v in data.get("goal_history", {}).items()
                }
                self.category_weights = data.get("category_weights", self.category_weights)
                self.category_fitness = defaultdict(list, data.get("category_fitness", {}))
                self.goal_counter = data.get("goal_counter", 0)
                
                print(f"\033[36mLoaded {len(self.goal_history)} goals from history\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load goal state: {e}\033[0m")
    
    def save_state(self):
        """Save goal state to disk."""
        state = {
            "goal_history": {k: v.to_dict() for k, v in self.goal_history.items()},
            "category_weights": self.category_weights,
            "category_fitness": dict(self.category_fitness),
            "goal_counter": self.goal_counter
        }
        U.dump_json(state, f"{self.ckpt_dir}/goals/state.json")
    
    def update_survival_state(self, events: List):
        """Update survival state from game events."""
        for event in events:
            if len(event) >= 2 and isinstance(event[1], dict):
                status = event[1].get("status", {})
                
                if "health" in status:
                    self.survival_state.health = status["health"]
                if "food" in status:
                    self.survival_state.hunger = status["food"]
                if "timeOfDay" in status:
                    time_str = status["timeOfDay"]
                    self.survival_state.time_of_day = "night" if "night" in time_str.lower() else "day"
                
                # Check for threats
                entities = event[1].get("nearbyEntities", [])
                threats = [e.get("name", "") for e in entities 
                          if any(mob in e.get("name", "").lower() 
                                for mob in ["zombie", "skeleton", "creeper", "spider", "enderman"])]
                self.survival_state.nearby_threats = threats
                self.survival_state.is_safe = len(threats) == 0
                
                # Update exploration memory
                if "position" in status:
                    pos = status["position"]
                    chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
                    self.exploration_memory.visited_chunks.add(chunk)
                    self.exploration_memory.last_visited[chunk] = time.time()
                
                if "biome" in event[1]:
                    self.exploration_memory.discovered_biomes.add(event[1]["biome"])
    
    def generate_next_goal(self, events: List, completed_tasks: List[str], 
                           inventory: Dict) -> Tuple[str, str, Dict]:
        """Generate the next goal based on emergent behavior.
        
        Returns:
            (task_description, context, goal_metadata)
        """
        self.update_survival_state(events)
        
        # Generate candidate goals from different sources
        candidates = []
        
        # 1. Survival-based goals (highest priority when needed)
        survival_goals = self._generate_survival_goals(inventory)
        candidates.extend(survival_goals)
        
        # 2. Exploration-based goals
        exploration_goals = self._generate_exploration_goals(events)
        candidates.extend(exploration_goals)
        
        # 3. Social learning goals (from observed players)
        social_goals = self._generate_social_learning_goals()
        candidates.extend(social_goals)
        
        # 4. Skill development goals
        skill_goals = self._generate_skill_goals(completed_tasks, inventory)
        candidates.extend(skill_goals)
        
        # 5. Tool progression goals
        tool_goals = self._generate_tool_progression_goals(inventory)
        candidates.extend(tool_goals)
        
        # 6. Creativity/novelty goals (mutations of successful goals)
        creative_goals = self._generate_creative_goals(completed_tasks)
        candidates.extend(creative_goals)
        
        if not candidates:
            # Fallback: basic exploration
            return self._fallback_goal()
        
        # Select goal based on weighted fitness and personality
        selected = self._select_goal(candidates)
        
        # Store goal
        self.active_goals.append(selected)
        self.goal_history[selected.id] = selected
        
        return selected.description, self._generate_context(selected), {
            "goal_id": selected.id,
            "category": selected.category.value,
            "priority": selected.priority,
            "derived_from": selected.derived_from
        }
    
    def _generate_survival_goals(self, inventory: Dict) -> List[Goal]:
        """Generate goals based on survival instincts."""
        goals = []
        current_time = time.time()
        
        # Health-based goals
        if self.survival_state.health < self.config.health_threshold:
            urgency = self.survival_state.health_urgency()
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.SURVIVAL,
                description="Find food and heal: health is low",
                priority=0.5 + urgency * 0.5,
                fitness=0.7,
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"urgency": "health", "current_health": self.survival_state.health},
                prerequisites=[],
                derived_from="instinct:health"
            ))
        
        # Hunger-based goals
        if self.survival_state.hunger < self.config.hunger_threshold:
            urgency = self.survival_state.hunger_urgency()
            
            # Check if we have food in inventory
            food_items = ["bread", "cooked", "apple", "steak", "porkchop", "chicken", "carrot"]
            has_food = any(food in item.lower() for item in inventory.keys() for food in food_items)
            
            if has_food:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SURVIVAL,
                    description="Eat food from inventory to restore hunger",
                    priority=0.4 + urgency * 0.4,
                    fitness=0.9,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"urgency": "hunger", "has_food": True},
                    prerequisites=[],
                    derived_from="instinct:hunger"
                ))
            else:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SURVIVAL,
                    description="Find and collect food: hunt animals or find crops",
                    priority=0.5 + urgency * 0.5,
                    fitness=0.6,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"urgency": "hunger", "has_food": False},
                    prerequisites=[],
                    derived_from="instinct:hunger"
                ))
        
        # Shelter-based goals (night time)
        if self.config.shelter_priority_at_night and self.survival_state.time_of_day == "night":
            if not self.survival_state.has_shelter:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SURVIVAL,
                    description="Build or find shelter: it's night time and dangerous",
                    priority=0.8,
                    fitness=0.6,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"urgency": "shelter", "time": "night"},
                    prerequisites=[],
                    derived_from="instinct:shelter"
                ))
        
        # Safety-based goals (threats nearby)
        if not self.survival_state.is_safe:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.SURVIVAL,
                description=f"Deal with nearby threats: {', '.join(self.survival_state.nearby_threats[:3])}",
                priority=0.9,
                fitness=0.5,
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"urgency": "safety", "threats": self.survival_state.nearby_threats},
                prerequisites=[],
                derived_from="instinct:safety"
            ))
        
        return goals
    
    def _generate_exploration_goals(self, events: List) -> List[Goal]:
        """Generate curiosity-driven exploration goals."""
        goals = []
        current_time = time.time()
        
        # Personality affects exploration drive
        curiosity = self.personality.traits.get("curiosity", 0.5)
        
        # Find unexplored directions
        unexplored_score = 1.0 - len(self.exploration_memory.visited_chunks) / 100
        unexplored_score = max(0.3, unexplored_score)
        
        if curiosity > 0.4 or unexplored_score > 0.7:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description="Explore in a new direction to discover new areas",
                priority=0.3 + curiosity * 0.3,
                fitness=self._get_category_fitness(GoalCategory.EXPLORATION),
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "curiosity", "unexplored_score": unexplored_score},
                prerequisites=[],
                derived_from="curiosity:exploration"
            ))
        
        # Biome diversity
        num_biomes = len(self.exploration_memory.discovered_biomes)
        if num_biomes < 5 and curiosity > 0.3:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description="Travel to find a new biome type",
                priority=0.25 + self.config.biome_diversity_bonus,
                fitness=0.5,
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "biome_diversity", "known_biomes": list(self.exploration_memory.discovered_biomes)},
                prerequisites=[],
                derived_from="curiosity:biome"
            ))
        
        # Cave exploration (if appropriate equipment)
        if curiosity > 0.6:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description="Find and explore a cave system",
                priority=0.3 + curiosity * 0.2,
                fitness=0.4,
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "adventure"},
                prerequisites=[],
                derived_from="curiosity:caves"
            ))
        
        return goals
    
    def _generate_social_learning_goals(self) -> List[Goal]:
        """Generate goals based on observing other players."""
        goals = []
        current_time = time.time()
        
        if not self.observer:
            return goals
        
        # Get observed behaviors
        from .player_observer import PlayerActivity
        
        # Check what activities other players do frequently
        activity_counts = defaultdict(int)
        for behavior in self.observer.observed_behaviors[-50:]:
            activity_counts[behavior.activity] += 1
        
        # Generate goals to imitate observed behaviors
        sociability = self.personality.traits.get("sociability", 0.5)
        imitation = self.personality.traits.get("imitation_tendency", 0.5)
        
        for activity, count in activity_counts.items():
            if count < self.config.observe_before_imitate:
                continue
            
            if activity == PlayerActivity.MINING:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SOCIAL_LEARNING,
                    description="Try mining like the observed players do",
                    priority=0.3 + imitation * 0.2 + self.config.success_imitation_boost,
                    fitness=0.6,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"learned_from": "player_observation", "activity": "mining"},
                    prerequisites=[],
                    derived_from=f"social:{activity.value}"
                ))
            
            elif activity == PlayerActivity.BUILDING:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SOCIAL_LEARNING,
                    description="Try building a structure like observed players",
                    priority=0.25 + imitation * 0.2,
                    fitness=0.5,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"learned_from": "player_observation", "activity": "building"},
                    prerequisites=[],
                    derived_from=f"social:{activity.value}"
                ))
            
            elif activity == PlayerActivity.FARMING:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SOCIAL_LEARNING,
                    description="Start a farm like the observed players",
                    priority=0.2 + imitation * 0.2,
                    fitness=0.55,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"learned_from": "player_observation", "activity": "farming"},
                    prerequisites=[],
                    derived_from=f"social:{activity.value}"
                ))
        
        return goals
    
    def _generate_skill_goals(self, completed_tasks: List[str], inventory: Dict) -> List[Goal]:
        """Generate goals for skill development."""
        goals = []
        current_time = time.time()
        
        # Analyze what skills have been developed
        skill_categories = defaultdict(int)
        for task in completed_tasks:
            task_lower = task.lower()
            if "mine" in task_lower:
                skill_categories["mining"] += 1
            if "craft" in task_lower:
                skill_categories["crafting"] += 1
            if "build" in task_lower:
                skill_categories["building"] += 1
            if "farm" in task_lower:
                skill_categories["farming"] += 1
            if "smelt" in task_lower:
                skill_categories["smelting"] += 1
        
        # Find underdeveloped skills
        all_skills = ["mining", "crafting", "building", "farming", "smelting"]
        for skill in all_skills:
            if skill_categories[skill] < 3:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SKILL_DEVELOPMENT,
                    description=f"Practice {skill} to improve this skill",
                    priority=0.25 + self.personality.traits.get("persistence", 0.5) * 0.15,
                    fitness=self._get_category_fitness(GoalCategory.SKILL_DEVELOPMENT),
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"skill": skill, "current_level": skill_categories[skill]},
                    prerequisites=[],
                    derived_from="skill_gap"
                ))
        
        return goals
    
    def _generate_tool_progression_goals(self, inventory: Dict) -> List[Goal]:
        """Generate goals for tool progression."""
        goals = []
        current_time = time.time()
        
        # Tool tiers
        tiers = ["wooden", "stone", "iron", "diamond"]
        tools = ["pickaxe", "sword", "axe"]
        
        # Find current best tools
        best_tier = {tool: -1 for tool in tools}
        for item in inventory.keys():
            item_lower = item.lower()
            for i, tier in enumerate(tiers):
                for tool in tools:
                    if tier in item_lower and tool in item_lower:
                        best_tier[tool] = max(best_tier[tool], i)
        
        # Generate progression goals
        for tool, current in best_tier.items():
            if current < len(tiers) - 1:
                next_tier = tiers[current + 1]
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.TOOL_PROGRESSION,
                    description=f"Craft a {next_tier} {tool}",
                    priority=0.3 + (current + 1) * 0.1,
                    fitness=0.6,
                    attempts=0,
                    successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"tool": tool, "current_tier": tiers[current] if current >= 0 else "none", "target_tier": next_tier},
                    prerequisites=[],
                    derived_from="progression"
                ))
        
        return goals
    
    def _generate_creative_goals(self, completed_tasks: List[str]) -> List[Goal]:
        """Generate creative/mutated goals based on past successes."""
        goals = []
        current_time = time.time()
        
        creativity = self.personality.traits.get("creativity", 0.5)
        
        if creativity < 0.3 or random.random() > self.config.goal_mutation_rate:
            return goals
        
        # Mutate a successful past goal
        successful_goals = [g for g in self.goal_history.values() 
                           if g.success_rate() > 0.6 and g.attempts > 0]
        
        if successful_goals:
            parent = random.choice(successful_goals)
            
            # Create mutation
            mutations = [
                f"Try a different approach to: {parent.description}",
                f"Do {parent.description} but bigger/more",
                f"Combine {parent.description} with exploration",
            ]
            
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.CREATIVITY,
                description=random.choice(mutations),
                priority=0.2 + creativity * 0.2,
                fitness=parent.fitness * 0.8,  # Slightly lower expected fitness
                attempts=0,
                successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"parent_goal": parent.id, "mutation_type": "variation"},
                prerequisites=[],
                derived_from=f"mutation:{parent.id}"
            ))
        
        return goals
    
    def _select_goal(self, candidates: List[Goal]) -> Goal:
        """Select the best goal based on weighted fitness and personality."""
        if not candidates:
            raise ValueError("No candidates to select from")
        
        # Score each candidate
        scores = []
        for goal in candidates:
            score = goal.priority
            
            # Apply category weight
            category_weight = self.category_weights.get(goal.category.value, 0.1)
            score *= category_weight
            
            # Apply fitness (historical success)
            score *= (0.5 + goal.fitness * 0.5)
            
            # Apply personality modifier
            personality_mod = self.personality.get_goal_preference_modifier(goal.category.value)
            score *= personality_mod
            
            # Freshness bonus (prefer goals not recently attempted)
            if goal.id in self.goal_history:
                time_since = time.time() - goal.last_attempted
                freshness = min(1.0, time_since / 300)  # Full freshness after 5 min
                score *= (0.7 + freshness * 0.3)
            
            scores.append(score)
        
        # Softmax selection with temperature based on impulsivity
        temperature = 0.5 + self.personality.traits.get("impulsivity", 0.3) * 0.5
        exp_scores = [math.exp(s / temperature) for s in scores]
        total = sum(exp_scores)
        probabilities = [s / total for s in exp_scores]
        
        # Select
        selected_idx = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        return candidates[selected_idx]
    
    def _generate_context(self, goal: Goal) -> str:
        """Generate context string for a goal."""
        context_parts = []
        
        if goal.derived_from:
            if "instinct" in goal.derived_from:
                context_parts.append("This is an urgent survival need.")
            elif "social" in goal.derived_from:
                context_parts.append("Learned this from watching other players.")
            elif "curiosity" in goal.derived_from:
                context_parts.append("Driven by curiosity and exploration.")
            elif "mutation" in goal.derived_from:
                context_parts.append("Trying a creative variation.")
        
        if goal.context:
            if "urgency" in goal.context:
                context_parts.append(f"Urgency level: {goal.context['urgency']}")
        
        return " ".join(context_parts)
    
    def record_goal_result(self, goal_id: str, success: bool):
        """Record the result of attempting a goal."""
        if goal_id in self.goal_history:
            goal = self.goal_history[goal_id]
            goal.attempts += 1
            if success:
                goal.successes += 1
                self.completed_goals.append(goal)
            else:
                self.failed_goals.append(goal)
            goal.last_attempted = time.time()
            
            # Update category fitness
            category = goal.category.value
            self.category_fitness[category].append(1.0 if success else 0.0)
            if len(self.category_fitness[category]) > self.config.fitness_memory_size:
                self.category_fitness[category] = self.category_fitness[category][-self.config.fitness_memory_size:]
            
            # Adapt category weights based on success
            if success:
                self.category_weights[category] *= (1 + self.config.adaptation_rate)
            else:
                self.category_weights[category] *= (1 - self.config.adaptation_rate * 0.5)
            
            # Normalize weights
            total = sum(self.category_weights.values())
            self.category_weights = {k: v/total for k, v in self.category_weights.items()}
            
            # Remove from active goals
            self.active_goals = [g for g in self.active_goals if g.id != goal_id]
    
    def _get_category_fitness(self, category: GoalCategory) -> float:
        """Get the fitness score for a goal category."""
        history = self.category_fitness.get(category.value, [])
        if not history:
            return 0.5
        return sum(history) / len(history)
    
    def _next_goal_id(self) -> str:
        """Generate next goal ID."""
        self.goal_counter += 1
        return f"goal_{self.goal_counter}"
    
    def _fallback_goal(self) -> Tuple[str, str, Dict]:
        """Generate a fallback goal when no candidates exist."""
        return (
            "Explore the surroundings and gather basic resources",
            "General exploration and resource gathering.",
            {"goal_id": self._next_goal_id(), "category": "fallback", "priority": 0.5}
        )
    
    def get_goal_statistics(self) -> Dict:
        """Get statistics about goal generation and success."""
        return {
            "total_goals_generated": len(self.goal_history),
            "completed": len(self.completed_goals),
            "failed": len(self.failed_goals),
            "active": len(self.active_goals),
            "category_weights": self.category_weights,
            "category_fitness": {k: sum(v)/len(v) if v else 0.5 
                                 for k, v in self.category_fitness.items()},
            "exploration_stats": {
                "chunks_visited": len(self.exploration_memory.visited_chunks),
                "biomes_discovered": list(self.exploration_memory.discovered_biomes)
            }
        }
