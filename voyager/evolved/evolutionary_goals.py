"""Enhanced Evolutionary Goal System for Voyager Evolved (Linux Optimized).

This module provides advanced emergent goal generation based on survival instincts,
curiosity, social learning, and evolutionary fitness.

Key Improvements:
- Goal priority system (survival > curiosity > social)
- Goal chaining (complete A to unlock B)
- Success/failure tracking for goal adaptation
- Better fitness calculation based on outcomes
- Goal mutation (variations on successful goals)
"""

import random
import time
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import voyager.utils as U


class GoalCategory(Enum):
    """Categories of emergent goals with priority ordering."""
    SURVIVAL = "survival"  # Priority 1: Food, health, shelter
    SAFETY = "safety"  # Priority 2: Avoiding danger
    TOOL_PROGRESSION = "tool_progression"  # Priority 3: Better equipment
    RESOURCE_GATHERING = "resource_gathering"  # Priority 4: Collecting materials
    SKILL_DEVELOPMENT = "skill_development"  # Priority 5: Improving abilities
    EXPLORATION = "exploration"  # Priority 6: Discovering new areas
    SOCIAL_LEARNING = "social_learning"  # Priority 7: Learning from others
    CREATIVITY = "creativity"  # Priority 8: Novel approaches

    @property
    def base_priority(self) -> float:
        """Get base priority for this category (higher = more important)."""
        priorities = {
            GoalCategory.SURVIVAL: 1.0,
            GoalCategory.SAFETY: 0.95,
            GoalCategory.TOOL_PROGRESSION: 0.7,
            GoalCategory.RESOURCE_GATHERING: 0.6,
            GoalCategory.SKILL_DEVELOPMENT: 0.5,
            GoalCategory.EXPLORATION: 0.4,
            GoalCategory.SOCIAL_LEARNING: 0.35,
            GoalCategory.CREATIVITY: 0.3
        }
        return priorities.get(self, 0.3)


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting for prerequisites
    ABANDONED = "abandoned"


@dataclass
class GoalOutcome:
    """Record of a goal attempt outcome."""
    timestamp: float
    success: bool
    duration: float
    attempts: int
    context: Dict[str, Any]
    failure_reason: Optional[str] = None


@dataclass
class Goal:
    """An enhanced emergent goal with chaining and tracking."""
    id: str
    category: GoalCategory
    description: str
    priority: float  # 0 to 1, dynamically adjusted
    fitness: float  # Historical success rate
    attempts: int
    successes: int
    created_at: float
    last_attempted: float
    context: Dict[str, Any]
    prerequisites: List[str]  # Goal IDs that should be completed first
    unlocks: List[str]  # Goal IDs unlocked by completing this
    derived_from: Optional[str]  # Source (observation, instinct, mutation)
    status: GoalStatus = GoalStatus.PENDING
    difficulty: float = 0.5  # Estimated difficulty 0-1
    expected_duration: float = 60.0  # Expected time in seconds
    outcomes: List[GoalOutcome] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)  # IDs of mutated versions
    parent_goal: Optional[str] = None  # If this is a mutation
    
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.5  # Neutral prior
        return self.successes / self.attempts
    
    def age(self) -> float:
        return time.time() - self.created_at
    
    def is_unlocked(self, completed_goals: Set[str]) -> bool:
        """Check if all prerequisites are met."""
        return all(prereq in completed_goals for prereq in self.prerequisites)
    
    def effective_priority(self) -> float:
        """Calculate effective priority considering category base and fitness."""
        base = self.category.base_priority
        # Adjust by fitness and attempts
        fitness_mod = (self.fitness - 0.5) * 0.2
        # Fresh goals get a bonus
        freshness = max(0, 1.0 - self.age() / 3600) * 0.1
        return min(1.0, max(0.0, self.priority * base + fitness_mod + freshness))
    
    def record_outcome(self, success: bool, duration: float, 
                       failure_reason: Optional[str] = None):
        """Record the outcome of an attempt."""
        self.attempts += 1
        if success:
            self.successes += 1
        
        self.outcomes.append(GoalOutcome(
            timestamp=time.time(),
            success=success,
            duration=duration,
            attempts=self.attempts,
            context=self.context.copy(),
            failure_reason=failure_reason
        ))
        
        # Update fitness with exponential moving average
        outcome_value = 1.0 if success else 0.0
        alpha = 0.3  # Learning rate
        self.fitness = self.fitness * (1 - alpha) + outcome_value * alpha
        
        # Update difficulty estimate
        if success:
            self.difficulty = self.difficulty * 0.9  # Easier than thought
        else:
            self.difficulty = min(1.0, self.difficulty * 1.1)  # Harder than thought
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['category'] = self.category.value
        data['status'] = self.status.value
        data['outcomes'] = [asdict(o) for o in self.outcomes]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        data['category'] = GoalCategory(data['category'])
        data['status'] = GoalStatus(data['status'])
        data['outcomes'] = [GoalOutcome(**o) for o in data.get('outcomes', [])]
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
    light_level: int = 15
    biome: str = "plains"
    
    def health_urgency(self) -> float:
        return 1.0 - (self.health / self.max_health)
    
    def hunger_urgency(self) -> float:
        return 1.0 - (self.hunger / 20.0)
    
    def safety_urgency(self) -> float:
        if self.is_safe:
            return 0.0
        return 0.5 + min(0.5, len(self.nearby_threats) * 0.15)
    
    def overall_urgency(self) -> float:
        """Get overall survival urgency."""
        return max(self.health_urgency(), self.hunger_urgency(), self.safety_urgency())


@dataclass
class ExplorationMemory:
    """Memory of explored areas and discoveries."""
    visited_chunks: Set[Tuple[int, int]] = field(default_factory=set)
    discovered_biomes: Set[str] = field(default_factory=set)
    found_structures: Set[str] = field(default_factory=set)
    interesting_locations: List[Dict] = field(default_factory=list)
    last_visited: Dict[Tuple[int, int], float] = field(default_factory=dict)
    death_locations: List[Tuple[float, float, float]] = field(default_factory=list)
    
    def novelty_score(self, chunk: Tuple[int, int]) -> float:
        if chunk not in self.visited_chunks:
            return 1.0
        
        last_visit = self.last_visited.get(chunk, 0)
        time_since = time.time() - last_visit
        return min(1.0, time_since / 3600)  # Full novelty after 1 hour
    
    def is_dangerous_area(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is near a death location."""
        for death_loc in self.death_locations[-10:]:
            dist = sum((a - b) ** 2 for a, b in zip(position, death_loc)) ** 0.5
            if dist < 30:
                return True
        return False


class GoalChainManager:
    """Manages goal chains and prerequisites."""
    
    def __init__(self):
        self.chains: Dict[str, List[str]] = {}  # Chain ID -> ordered goal IDs
        self.goal_to_chain: Dict[str, str] = {}  # Goal ID -> Chain ID
        
        # Define standard progression chains
        self._define_default_chains()
    
    def _define_default_chains(self):
        """Define standard goal chains."""
        # Tool progression chain
        self.chains["tool_progression"] = [
            "get_wood",
            "craft_crafting_table",
            "craft_wooden_pickaxe",
            "mine_stone",
            "craft_stone_pickaxe",
            "mine_iron",
            "smelt_iron",
            "craft_iron_pickaxe",
            "mine_diamonds",
            "craft_diamond_pickaxe"
        ]
        
        # Shelter chain
        self.chains["shelter"] = [
            "gather_wood",
            "craft_planks",
            "build_basic_shelter",
            "craft_door",
            "craft_bed"
        ]
        
        # Farming chain
        self.chains["farming"] = [
            "find_seeds",
            "craft_hoe",
            "till_soil",
            "plant_crops",
            "harvest_crops"
        ]
    
    def get_chain_position(self, goal_id: str) -> Tuple[Optional[str], int]:
        """Get chain and position for a goal."""
        for chain_id, goals in self.chains.items():
            if goal_id in goals:
                return chain_id, goals.index(goal_id)
        return None, -1
    
    def get_next_in_chain(self, goal_id: str) -> Optional[str]:
        """Get next goal in the chain."""
        chain_id, pos = self.get_chain_position(goal_id)
        if chain_id and pos >= 0 and pos < len(self.chains[chain_id]) - 1:
            return self.chains[chain_id][pos + 1]
        return None
    
    def get_prerequisites(self, goal_id: str) -> List[str]:
        """Get all prerequisites for a goal."""
        chain_id, pos = self.get_chain_position(goal_id)
        if chain_id and pos > 0:
            return self.chains[chain_id][:pos]
        return []


class GoalMutator:
    """Creates variations of successful goals."""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutation_history: List[Tuple[str, str]] = []  # (parent_id, child_id)
    
    def should_mutate(self) -> bool:
        return random.random() < self.mutation_rate
    
    def mutate_goal(self, parent: Goal, goal_counter: int) -> Optional[Goal]:
        """Create a mutation of a successful goal."""
        if parent.success_rate() < 0.5:
            return None
        
        mutation_types = [
            self._scale_mutation,
            self._context_mutation,
            self._combination_mutation,
            self._efficiency_mutation
        ]
        
        mutator = random.choice(mutation_types)
        return mutator(parent, goal_counter)
    
    def _scale_mutation(self, parent: Goal, goal_counter: int) -> Goal:
        """Scale up or down the goal."""
        scales = ["smaller", "larger", "faster", "more thorough"]
        scale = random.choice(scales)
        
        return Goal(
            id=f"goal_{goal_counter}",
            category=parent.category,
            description=f"{scale.capitalize()} version: {parent.description}",
            priority=parent.priority * 0.9,
            fitness=parent.fitness * 0.8,
            attempts=0,
            successes=0,
            created_at=time.time(),
            last_attempted=0,
            context={**parent.context, "mutation_type": "scale", "scale": scale},
            prerequisites=parent.prerequisites.copy(),
            unlocks=[],
            derived_from=f"mutation:scale:{parent.id}",
            parent_goal=parent.id,
            difficulty=parent.difficulty * (1.1 if scale in ["larger", "more thorough"] else 0.9)
        )
    
    def _context_mutation(self, parent: Goal, goal_counter: int) -> Goal:
        """Try the goal in a different context."""
        contexts = ["at night", "in a cave", "underwater", "at high altitude", "during rain"]
        context = random.choice(contexts)
        
        return Goal(
            id=f"goal_{goal_counter}",
            category=parent.category,
            description=f"Try {parent.description} {context}",
            priority=parent.priority * 0.85,
            fitness=parent.fitness * 0.7,
            attempts=0,
            successes=0,
            created_at=time.time(),
            last_attempted=0,
            context={**parent.context, "mutation_type": "context", "new_context": context},
            prerequisites=parent.prerequisites.copy(),
            unlocks=[],
            derived_from=f"mutation:context:{parent.id}",
            parent_goal=parent.id,
            difficulty=parent.difficulty * 1.2
        )
    
    def _combination_mutation(self, parent: Goal, goal_counter: int) -> Goal:
        """Combine with another activity."""
        combinations = ["while exploring", "while gathering resources", 
                       "after building shelter", "before nightfall"]
        combo = random.choice(combinations)
        
        return Goal(
            id=f"goal_{goal_counter}",
            category=parent.category,
            description=f"{parent.description} {combo}",
            priority=parent.priority * 0.85,
            fitness=parent.fitness * 0.75,
            attempts=0,
            successes=0,
            created_at=time.time(),
            last_attempted=0,
            context={**parent.context, "mutation_type": "combination", "combo": combo},
            prerequisites=parent.prerequisites.copy(),
            unlocks=[],
            derived_from=f"mutation:combo:{parent.id}",
            parent_goal=parent.id,
            difficulty=parent.difficulty * 1.15
        )
    
    def _efficiency_mutation(self, parent: Goal, goal_counter: int) -> Goal:
        """Optimize for efficiency."""
        optimizations = ["more efficiently", "using better tools", 
                        "with less risk", "in bulk"]
        opt = random.choice(optimizations)
        
        return Goal(
            id=f"goal_{goal_counter}",
            category=parent.category,
            description=f"Complete {parent.description} {opt}",
            priority=parent.priority * 0.9,
            fitness=parent.fitness * 0.85,
            attempts=0,
            successes=0,
            created_at=time.time(),
            last_attempted=0,
            context={**parent.context, "mutation_type": "efficiency", "optimization": opt},
            prerequisites=parent.prerequisites.copy(),
            unlocks=[],
            derived_from=f"mutation:efficiency:{parent.id}",
            parent_goal=parent.id,
            difficulty=parent.difficulty * 0.95
        )


class EvolutionaryGoalSystem:
    """Enhanced evolutionary goal generation with priorities, chaining, and mutations.
    
    Goals are generated from:
    1. Survival instincts (highest priority)
    2. Safety concerns
    3. Tool/equipment progression
    4. Resource gathering
    5. Skill development
    6. Exploration
    7. Social learning
    8. Creativity/novelty
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
        self.completed_goal_ids: Set[str] = set()
        self.failed_goal_ids: Set[str] = set()
        self.goal_history: Dict[str, Goal] = {}
        
        # State tracking
        self.survival_state = SurvivalState()
        self.exploration_memory = ExplorationMemory()
        
        # Chain and mutation managers
        self.chain_manager = GoalChainManager()
        self.mutator = GoalMutator(mutation_rate=self.config.goal_mutation_rate)
        
        # Fitness tracking for goal types
        self.category_fitness: Dict[str, List[float]] = defaultdict(list)
        self.category_weights: Dict[str, float] = {
            GoalCategory.SURVIVAL.value: self.config.survival_weight,
            GoalCategory.SAFETY.value: 0.25,
            GoalCategory.EXPLORATION.value: self.config.exploration_weight,
            GoalCategory.SOCIAL_LEARNING.value: self.config.social_learning_weight,
            GoalCategory.SKILL_DEVELOPMENT.value: self.config.skill_development_weight,
            GoalCategory.CREATIVITY.value: self.config.creativity_weight,
            GoalCategory.RESOURCE_GATHERING.value: 0.15,
            GoalCategory.TOOL_PROGRESSION.value: 0.2
        }
        
        # Goal ID counter
        self.goal_counter = 0
        
        # Recent context for better goal generation
        self.recent_inventory: Dict[str, int] = {}
        self.current_position: Optional[Tuple[float, float, float]] = None
        
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
                self.completed_goal_ids = set(data.get("completed_goal_ids", []))
                self.failed_goal_ids = set(data.get("failed_goal_ids", []))
                self.category_weights = data.get("category_weights", self.category_weights)
                self.category_fitness = defaultdict(list, data.get("category_fitness", {}))
                self.goal_counter = data.get("goal_counter", 0)
                
                print(f"\033[36mLoaded {len(self.goal_history)} goals, "
                      f"{len(self.completed_goal_ids)} completed\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: Could not load goal state: {e}\033[0m")
    
    def save_state(self):
        """Save goal state to disk."""
        state = {
            "goal_history": {k: v.to_dict() for k, v in self.goal_history.items()},
            "completed_goal_ids": list(self.completed_goal_ids),
            "failed_goal_ids": list(self.failed_goal_ids),
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
                if "lightLevel" in status:
                    self.survival_state.light_level = status["lightLevel"]
                
                # Check for threats
                entities = event[1].get("nearbyEntities", [])
                hostile_mobs = ["zombie", "skeleton", "creeper", "spider", "enderman", 
                              "witch", "pillager", "vindicator"]
                threats = [e.get("name", "") for e in entities 
                          if any(mob in e.get("name", "").lower() for mob in hostile_mobs)]
                self.survival_state.nearby_threats = threats
                self.survival_state.is_safe = len(threats) == 0
                
                # Update exploration memory
                if "position" in status:
                    pos = status["position"]
                    self.current_position = (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
                    chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
                    self.exploration_memory.visited_chunks.add(chunk)
                    self.exploration_memory.last_visited[chunk] = time.time()
                
                if "biome" in event[1]:
                    biome = event[1]["biome"]
                    self.exploration_memory.discovered_biomes.add(biome)
                    self.survival_state.biome = biome
                
                # Update inventory
                if "inventory" in event[1]:
                    self.recent_inventory = {
                        item.get("name", ""): item.get("count", 0) 
                        for item in event[1]["inventory"] 
                        if isinstance(item, dict)
                    }
    
    def generate_next_goal(self, events: List, completed_tasks: List[str], 
                           inventory: Dict) -> Tuple[str, str, Dict]:
        """Generate the next goal using priority system.
        
        Returns:
            (task_description, context, goal_metadata)
        """
        self.update_survival_state(events)
        self.recent_inventory = inventory
        
        # Generate candidate goals from different sources (in priority order)
        all_candidates = []
        
        # 1. SURVIVAL goals (highest priority when needed)
        survival_urgency = self.survival_state.overall_urgency()
        if survival_urgency > 0.3:
            survival_goals = self._generate_survival_goals(inventory)
            for g in survival_goals:
                g.priority *= (1 + survival_urgency)  # Boost by urgency
            all_candidates.extend(survival_goals)
        
        # 2. SAFETY goals
        if not self.survival_state.is_safe:
            safety_goals = self._generate_safety_goals()
            all_candidates.extend(safety_goals)
        
        # 3. TOOL PROGRESSION goals
        tool_goals = self._generate_tool_progression_goals(inventory)
        all_candidates.extend(tool_goals)
        
        # 4. RESOURCE GATHERING goals
        resource_goals = self._generate_resource_goals(inventory)
        all_candidates.extend(resource_goals)
        
        # 5. SKILL DEVELOPMENT goals
        skill_goals = self._generate_skill_goals(completed_tasks, inventory)
        all_candidates.extend(skill_goals)
        
        # 6. EXPLORATION goals
        exploration_goals = self._generate_exploration_goals(events)
        all_candidates.extend(exploration_goals)
        
        # 7. SOCIAL LEARNING goals
        if self.observer:
            social_goals = self._generate_social_learning_goals()
            all_candidates.extend(social_goals)
        
        # 8. CREATIVITY/MUTATION goals
        creative_goals = self._generate_creative_goals(completed_tasks)
        all_candidates.extend(creative_goals)
        
        # Filter by prerequisites
        all_candidates = [g for g in all_candidates if g.is_unlocked(self.completed_goal_ids)]
        
        if not all_candidates:
            return self._fallback_goal()
        
        # Select goal based on effective priority and personality
        selected = self._select_goal(all_candidates)
        
        # Store goal
        selected.status = GoalStatus.ACTIVE
        self.active_goals.append(selected)
        self.goal_history[selected.id] = selected
        
        return selected.description, self._generate_context(selected), {
            "goal_id": selected.id,
            "category": selected.category.value,
            "priority": selected.effective_priority(),
            "difficulty": selected.difficulty,
            "derived_from": selected.derived_from,
            "prerequisites": selected.prerequisites,
            "unlocks": selected.unlocks
        }
    
    def _generate_survival_goals(self, inventory: Dict) -> List[Goal]:
        """Generate high-priority survival goals."""
        goals = []
        current_time = time.time()
        
        # Critical health - immediate healing
        if self.survival_state.health < 6:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.SURVIVAL,
                description="URGENT: Find food and heal immediately - health is critical!",
                priority=1.0,
                fitness=0.7,
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"urgency": "critical_health", "health": self.survival_state.health},
                prerequisites=[], unlocks=[],
                derived_from="instinct:survival:critical",
                difficulty=0.4
            ))
        
        # Low health
        elif self.survival_state.health < self.config.health_threshold:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.SURVIVAL,
                description="Find food and heal: health is low",
                priority=0.8,
                fitness=0.7,
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"urgency": "health", "health": self.survival_state.health},
                prerequisites=[], unlocks=[],
                derived_from="instinct:survival:health",
                difficulty=0.3
            ))
        
        # Hunger management
        if self.survival_state.hunger < self.config.hunger_threshold:
            food_items = ["bread", "cooked", "apple", "steak", "porkchop", 
                         "chicken", "carrot", "potato", "melon", "berries"]
            has_food = any(any(f in item.lower() for f in food_items) 
                         for item in inventory.keys())
            
            if has_food:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SURVIVAL,
                    description="Eat food from inventory to restore hunger",
                    priority=0.7,
                    fitness=0.95,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"urgency": "hunger", "has_food": True},
                    prerequisites=[], unlocks=[],
                    derived_from="instinct:survival:eat",
                    difficulty=0.1
                ))
            else:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SURVIVAL,
                    description="Hunt animals or find food - hunger is getting low",
                    priority=0.75,
                    fitness=0.6,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"urgency": "hunger", "has_food": False},
                    prerequisites=[], unlocks=["cook_food"],
                    derived_from="instinct:survival:find_food",
                    difficulty=0.4
                ))
        
        # Night shelter
        if (self.config.shelter_priority_at_night and 
            self.survival_state.time_of_day == "night" and
            not self.survival_state.has_shelter):
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.SURVIVAL,
                description="Build emergency shelter - it's night and dangerous!",
                priority=0.85,
                fitness=0.6,
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"urgency": "shelter", "time": "night"},
                prerequisites=[], unlocks=["craft_bed"],
                derived_from="instinct:survival:shelter",
                difficulty=0.5
            ))
        
        return goals
    
    def _generate_safety_goals(self) -> List[Goal]:
        """Generate safety goals when threats are nearby."""
        goals = []
        current_time = time.time()
        
        threats = self.survival_state.nearby_threats
        
        if threats:
            # Decide fight or flight based on personality
            caution = self.personality.traits.get("caution", 0.5) if self.personality else 0.5
            
            if caution > 0.6 or len(threats) > 2:
                # Flight
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SAFETY,
                    description=f"Escape from threats: {', '.join(threats[:3])}",
                    priority=0.9,
                    fitness=0.7,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"threats": threats, "action": "flee"},
                    prerequisites=[], unlocks=[],
                    derived_from="instinct:safety:flee",
                    difficulty=0.4
                ))
            else:
                # Fight
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SAFETY,
                    description=f"Fight nearby threats: {', '.join(threats[:2])}",
                    priority=0.85,
                    fitness=0.5,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"threats": threats, "action": "fight"},
                    prerequisites=[], unlocks=[],
                    derived_from="instinct:safety:fight",
                    difficulty=0.6
                ))
        
        return goals
    
    def _generate_tool_progression_goals(self, inventory: Dict) -> List[Goal]:
        """Generate tool progression goals."""
        goals = []
        current_time = time.time()
        
        # Analyze current tools
        tiers = ["wooden", "stone", "iron", "diamond", "netherite"]
        tools = ["pickaxe", "sword", "axe", "shovel"]
        
        best_tier: Dict[str, int] = {tool: -1 for tool in tools}
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
                prerequisites = []
                
                # Add prerequisites based on tier
                if next_tier == "stone" and current < 0:
                    prerequisites = ["craft_wooden_pickaxe"]
                elif next_tier == "iron":
                    prerequisites = ["find_iron_ore"]
                elif next_tier == "diamond":
                    prerequisites = ["mine_deep"]
                
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.TOOL_PROGRESSION,
                    description=f"Craft a {next_tier} {tool}",
                    priority=0.5 + (current + 1) * 0.1,
                    fitness=0.6,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"tool": tool, "current_tier": tiers[current] if current >= 0 else "none", 
                            "target_tier": next_tier},
                    prerequisites=prerequisites,
                    unlocks=[f"craft_{tiers[min(current+2, len(tiers)-1)]}_{tool}"],
                    derived_from="progression:tools",
                    difficulty=0.3 + (current + 1) * 0.15
                ))
        
        return goals
    
    def _generate_resource_goals(self, inventory: Dict) -> List[Goal]:
        """Generate resource gathering goals."""
        goals = []
        current_time = time.time()
        
        # Check for basic resources
        resources = {
            "wood": ("log", 16),
            "cobblestone": ("cobblestone", 32),
            "iron": ("iron", 8),
            "coal": ("coal", 16),
            "food": (["cooked", "bread", "apple"], 8)
        }
        
        for resource_name, (item_pattern, min_amount) in resources.items():
            if isinstance(item_pattern, list):
                current = sum(inventory.get(k, 0) for k in inventory.keys() 
                             if any(p in k.lower() for p in item_pattern))
            else:
                current = sum(v for k, v in inventory.items() if item_pattern in k.lower())
            
            if current < min_amount:
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.RESOURCE_GATHERING,
                    description=f"Gather more {resource_name} (have {current}, need {min_amount})",
                    priority=0.4 + (1 - current / min_amount) * 0.2,
                    fitness=0.65,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"resource": resource_name, "current": current, "target": min_amount},
                    prerequisites=[], unlocks=[],
                    derived_from="resource:shortage",
                    difficulty=0.3
                ))
        
        return goals
    
    def _generate_skill_goals(self, completed_tasks: List[str], inventory: Dict) -> List[Goal]:
        """Generate skill development goals."""
        goals = []
        current_time = time.time()
        
        # Analyze completed tasks
        skill_categories = defaultdict(int)
        for task in completed_tasks:
            task_lower = task.lower()
            if "mine" in task_lower: skill_categories["mining"] += 1
            if "craft" in task_lower: skill_categories["crafting"] += 1
            if "build" in task_lower: skill_categories["building"] += 1
            if "farm" in task_lower: skill_categories["farming"] += 1
            if "smelt" in task_lower: skill_categories["smelting"] += 1
            if "fight" in task_lower or "kill" in task_lower: skill_categories["combat"] += 1
        
        # Find underdeveloped skills
        all_skills = ["mining", "crafting", "building", "farming", "smelting", "combat"]
        for skill in all_skills:
            count = skill_categories[skill]
            if count < 5:  # Threshold for "underdeveloped"
                persistence = self.personality.traits.get("persistence", 0.5) if self.personality else 0.5
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SKILL_DEVELOPMENT,
                    description=f"Practice {skill} to improve this skill (level: {count}/5)",
                    priority=0.3 + persistence * 0.2,
                    fitness=self._get_category_fitness(GoalCategory.SKILL_DEVELOPMENT),
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"skill": skill, "current_level": count},
                    prerequisites=[], unlocks=[],
                    derived_from="skill:development",
                    difficulty=0.4 + count * 0.05
                ))
        
        return goals
    
    def _generate_exploration_goals(self, events: List) -> List[Goal]:
        """Generate exploration goals."""
        goals = []
        current_time = time.time()
        
        curiosity = self.personality.traits.get("curiosity", 0.5) if self.personality else 0.5
        
        # Calculate exploration need
        explored_ratio = len(self.exploration_memory.visited_chunks) / 100
        exploration_need = max(0.3, 1.0 - explored_ratio)
        
        if curiosity > 0.4 or exploration_need > 0.6:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description="Explore in a new direction to discover new areas",
                priority=0.35 + curiosity * 0.2,
                fitness=self._get_category_fitness(GoalCategory.EXPLORATION),
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "curiosity", "exploration_need": exploration_need},
                prerequisites=[], unlocks=[],
                derived_from="curiosity:exploration",
                difficulty=0.3
            ))
        
        # Biome exploration
        num_biomes = len(self.exploration_memory.discovered_biomes)
        if num_biomes < 5 and curiosity > 0.3:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description=f"Travel to find a new biome type (discovered: {num_biomes}/5)",
                priority=0.3 + self.config.biome_diversity_bonus,
                fitness=0.5,
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "biome_diversity", 
                        "known_biomes": list(self.exploration_memory.discovered_biomes)},
                prerequisites=[], unlocks=[],
                derived_from="curiosity:biome",
                difficulty=0.4
            ))
        
        # Cave exploration
        if curiosity > 0.6:
            goals.append(Goal(
                id=self._next_goal_id(),
                category=GoalCategory.EXPLORATION,
                description="Find and explore a cave system",
                priority=0.3 + curiosity * 0.15,
                fitness=0.4,
                attempts=0, successes=0,
                created_at=current_time,
                last_attempted=0,
                context={"motivation": "adventure"},
                prerequisites=["craft_torches"], unlocks=["find_diamonds"],
                derived_from="curiosity:caves",
                difficulty=0.5
            ))
        
        return goals
    
    def _generate_social_learning_goals(self) -> List[Goal]:
        """Generate goals based on observing other players."""
        goals = []
        current_time = time.time()
        
        if not self.observer:
            return goals
        
        from .player_observer import PlayerActivity
        
        # Get observed behaviors with good relevance
        activity_counts = defaultdict(int)
        for behavior in self.observer.observed_behaviors:
            if behavior.relevance_score() > 0.4:
                activity_counts[behavior.activity] += 1
        
        sociability = self.personality.traits.get("sociability", 0.5) if self.personality else 0.5
        imitation = self.personality.traits.get("imitation_tendency", 0.5) if self.personality else 0.5
        
        activity_to_goal = {
            PlayerActivity.MINING: ("Try mining like the observed players do", "mining"),
            PlayerActivity.BUILDING: ("Build a structure like observed players", "building"),
            PlayerActivity.FARMING: ("Start a farm like the observed players", "farming"),
            PlayerActivity.EXPLORING: ("Explore the area like observed players", "exploration"),
            PlayerActivity.GATHERING: ("Gather resources like observed players", "gathering"),
        }
        
        for activity, count in activity_counts.items():
            if count < self.config.observe_before_imitate:
                continue
            
            if activity in activity_to_goal:
                desc, skill = activity_to_goal[activity]
                goals.append(Goal(
                    id=self._next_goal_id(),
                    category=GoalCategory.SOCIAL_LEARNING,
                    description=desc,
                    priority=0.3 + imitation * 0.2 + self.config.success_imitation_boost,
                    fitness=0.6,
                    attempts=0, successes=0,
                    created_at=current_time,
                    last_attempted=0,
                    context={"learned_from": "player_observation", "activity": activity.value,
                            "observation_count": count},
                    prerequisites=[], unlocks=[],
                    derived_from=f"social:{activity.value}",
                    difficulty=0.4
                ))
        
        return goals
    
    def _generate_creative_goals(self, completed_tasks: List[str]) -> List[Goal]:
        """Generate creative/mutated goals based on past successes."""
        goals = []
        
        creativity = self.personality.traits.get("creativity", 0.5) if self.personality else 0.5
        
        if creativity < 0.3:
            return goals
        
        # Find successful goals to mutate
        successful_goals = [g for g in self.goal_history.values() 
                           if g.success_rate() > 0.6 and g.attempts >= 2]
        
        if not successful_goals or not self.mutator.should_mutate():
            return goals
        
        # Select a goal to mutate
        parent = random.choice(successful_goals)
        mutated = self.mutator.mutate_goal(parent, self.goal_counter)
        
        if mutated:
            self.goal_counter += 1
            parent.mutations.append(mutated.id)
            goals.append(mutated)
        
        return goals
    
    def _select_goal(self, candidates: List[Goal]) -> Goal:
        """Select the best goal using weighted scoring."""
        if not candidates:
            raise ValueError("No candidates to select from")
        
        # Score each candidate
        scored = []
        for goal in candidates:
            score = goal.effective_priority()
            
            # Apply category weight
            category_weight = self.category_weights.get(goal.category.value, 0.1)
            score *= (0.5 + category_weight * 0.5)
            
            # Apply fitness
            score *= (0.6 + goal.fitness * 0.4)
            
            # Personality modifier
            if self.personality:
                personality_mod = self.personality.get_goal_preference_modifier(goal.category.value)
                score *= personality_mod
            
            # Freshness bonus
            if goal.id in self.goal_history:
                time_since = time.time() - goal.last_attempted
                freshness = min(1.0, time_since / 300)
                score *= (0.7 + freshness * 0.3)
            
            # Difficulty adjustment (prefer achievable goals)
            difficulty_mod = 1.0 - abs(goal.difficulty - 0.5) * 0.2
            score *= difficulty_mod
            
            scored.append((goal, score))
        
        # Softmax selection
        impulsivity = self.personality.traits.get("impulsivity", 0.3) if self.personality else 0.3
        temperature = 0.3 + impulsivity * 0.5
        
        exp_scores = [math.exp(s / temperature) for _, s in scored]
        total = sum(exp_scores)
        probabilities = [s / total for s in exp_scores]
        
        selected_idx = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        return candidates[selected_idx]
    
    def _generate_context(self, goal: Goal) -> str:
        """Generate context string for a goal."""
        parts = []
        
        if goal.derived_from:
            if "instinct:survival" in goal.derived_from:
                parts.append("🚨 URGENT survival need!")
            elif "instinct:safety" in goal.derived_from:
                parts.append("⚠️ Safety concern!")
            elif "social" in goal.derived_from:
                parts.append("👀 Learned from watching other players.")
            elif "curiosity" in goal.derived_from:
                parts.append("🔍 Driven by curiosity.")
            elif "mutation" in goal.derived_from:
                parts.append("🧬 Creative variation of a successful approach.")
            elif "progression" in goal.derived_from:
                parts.append("📈 Part of natural progression.")
        
        if goal.context:
            if "urgency" in goal.context:
                parts.append(f"Urgency: {goal.context['urgency']}")
            if goal.difficulty > 0.7:
                parts.append("⚡ This will be challenging!")
        
        if goal.prerequisites:
            parts.append(f"Prerequisites: {', '.join(goal.prerequisites)}")
        
        return " ".join(parts)
    
    def record_goal_result(self, goal_id: str, success: bool, 
                           duration: float = 0, failure_reason: str = None):
        """Record the result of attempting a goal."""
        if goal_id not in self.goal_history:
            return
        
        goal = self.goal_history[goal_id]
        goal.record_outcome(success, duration, failure_reason)
        goal.last_attempted = time.time()
        
        if success:
            goal.status = GoalStatus.COMPLETED
            self.completed_goal_ids.add(goal_id)
            
            # Check for chain progression
            next_goal = self.chain_manager.get_next_in_chain(goal_id)
            if next_goal:
                print(f"\033[36mUnlocked next goal in chain: {next_goal}\033[0m")
        else:
            if goal.attempts >= 3:
                goal.status = GoalStatus.FAILED
                self.failed_goal_ids.add(goal_id)
            else:
                goal.status = GoalStatus.PENDING
        
        # Update category fitness
        category = goal.category.value
        self.category_fitness[category].append(1.0 if success else 0.0)
        if len(self.category_fitness[category]) > self.config.fitness_memory_size:
            self.category_fitness[category] = \
                self.category_fitness[category][-self.config.fitness_memory_size:]
        
        # Adapt category weights
        if success:
            self.category_weights[category] *= (1 + self.config.adaptation_rate)
        else:
            self.category_weights[category] *= (1 - self.config.adaptation_rate * 0.5)
        
        # Normalize weights
        total = sum(self.category_weights.values())
        self.category_weights = {k: v/total for k, v in self.category_weights.items()}
        
        # Remove from active goals
        self.active_goals = [g for g in self.active_goals if g.id != goal_id]
        
        # Notify personality engine
        if self.personality:
            if success:
                self.personality.record_success(goal.description, goal.difficulty)
            else:
                self.personality.record_failure(goal.description, failure_reason or "")
    
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
        goal_id = self._next_goal_id()
        return (
            "Explore the surroundings and gather basic resources",
            "General exploration and resource gathering.",
            {"goal_id": goal_id, "category": "fallback", "priority": 0.5,
             "difficulty": 0.3, "derived_from": "fallback"}
        )
    
    def get_goal_statistics(self) -> Dict:
        """Get comprehensive statistics about goal system."""
        category_stats = {}
        for category in GoalCategory:
            cat_goals = [g for g in self.goal_history.values() if g.category == category]
            if cat_goals:
                category_stats[category.value] = {
                    "total": len(cat_goals),
                    "success_rate": sum(g.success_rate() for g in cat_goals) / len(cat_goals),
                    "avg_difficulty": sum(g.difficulty for g in cat_goals) / len(cat_goals),
                    "weight": self.category_weights.get(category.value, 0)
                }
        
        return {
            "total_goals_generated": len(self.goal_history),
            "completed": len(self.completed_goal_ids),
            "failed": len(self.failed_goal_ids),
            "active": len(self.active_goals),
            "category_stats": category_stats,
            "category_weights": self.category_weights,
            "exploration_stats": {
                "chunks_visited": len(self.exploration_memory.visited_chunks),
                "biomes_discovered": list(self.exploration_memory.discovered_biomes)
            },
            "survival_state": {
                "health": self.survival_state.health,
                "hunger": self.survival_state.hunger,
                "is_safe": self.survival_state.is_safe,
                "urgency": self.survival_state.overall_urgency()
            }
        }
