"""
Enhanced Semantic Memory System for Voyager Evolved

Implements a cognitively-realistic semantic memory inspired by:
- Collins & Quillian's hierarchical network model
- Collins & Loftus's spreading activation theory
- Frames and schema theory

Features:
1. Concept Hierarchies - IS-A relationships forming taxonomies
2. Relationship Graphs - Multiple relation types between concepts
3. Abstraction & Generalization - Learning general rules from specifics
4. Inference & Reasoning - Deriving new knowledge from existing
5. Spreading Activation - Associative retrieval
"""

import time
import threading
import json
import os
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set, Callable
from collections import defaultdict
import heapq


# ============================================================================
# RELATION TYPES
# ============================================================================

class RelationType(Enum):
    """Types of relationships between concepts."""
    
    # Taxonomic (hierarchical)
    IS_A = auto()           # cat IS_A animal
    INSTANCE_OF = auto()    # Mittens INSTANCE_OF cat
    SUBTYPE_OF = auto()     # oak SUBTYPE_OF tree
    
    # Compositional
    HAS_PART = auto()       # car HAS_PART wheel
    PART_OF = auto()        # wheel PART_OF car
    MADE_OF = auto()        # sword MADE_OF iron
    
    # Spatial
    LOCATED_IN = auto()     # diamonds LOCATED_IN caves
    FOUND_NEAR = auto()     # zombies FOUND_NEAR spawners
    
    # Functional
    USED_FOR = auto()       # pickaxe USED_FOR mining
    REQUIRES = auto()       # diamond_pickaxe REQUIRES diamonds
    PRODUCES = auto()       # furnace PRODUCES iron_ingot
    ENABLES = auto()        # torch ENABLES seeing
    
    # Causal
    CAUSES = auto()         # fire CAUSES damage
    PREVENTS = auto()       # armor PREVENTS damage
    
    # Associative
    RELATED_TO = auto()     # generic association
    SIMILAR_TO = auto()     # similarity relation
    OPPOSITE_OF = auto()    # opposite/contrasting
    
    # Temporal
    BEFORE = auto()         # wood BEFORE planks (in progression)
    AFTER = auto()          # iron AFTER stone (in progression)
    
    # Agent-specific
    CAN_PERFORM = auto()    # player CAN_PERFORM mine
    VULNERABLE_TO = auto()  # zombie VULNERABLE_TO sunlight


@dataclass
class Relation:
    """A directional relationship between two concepts."""
    source: str
    relation_type: RelationType
    target: str
    strength: float = 1.0  # 0-1
    confidence: float = 1.0  # How certain we are
    learned_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def decay(self, rate: float = 0.01):
        """Apply strength decay."""
        age_hours = (time.time() - self.learned_at) / 3600
        self.strength *= (1 - rate) ** (age_hours / 24)
    
    def reinforce(self, amount: float = 0.1):
        """Reinforce the relation from use."""
        self.strength = min(1.0, self.strength + amount)
        self.access_count += 1


@dataclass
class ConceptProperty:
    """A property of a concept."""
    name: str
    value: Any
    is_defining: bool = True  # Defining vs characteristic property
    inherited: bool = False  # Inherited from parent concept
    confidence: float = 1.0


@dataclass
class Concept:
    """
    A semantic concept (category, object, action, etc.).
    
    Combines features from:
    - Prototype theory (typical features)
    - Classical theory (defining features)
    - Exemplar theory (specific instances)
    """
    id: str
    name: str
    category: str = "general"  # e.g., "entity", "item", "action", "biome"
    
    # Properties
    properties: Dict[str, ConceptProperty] = field(default_factory=dict)
    
    # Hierarchy
    parents: Set[str] = field(default_factory=set)  # IS_A relations
    children: Set[str] = field(default_factory=set)  # Subtypes
    
    # Relations (non-hierarchical)
    outgoing_relations: List[str] = field(default_factory=list)  # Relation IDs
    incoming_relations: List[str] = field(default_factory=list)  # Relation IDs
    
    # Activation
    activation: float = 0.0  # Current activation level
    resting_activation: float = 0.0  # Base activation
    
    # Learning
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    exemplars: List[str] = field(default_factory=list)  # Specific instances
    
    # Abstraction level
    abstraction_level: int = 0  # 0 = concrete, higher = more abstract
    
    # Embedding for similarity
    embedding: Optional[List[float]] = None
    
    def get_property(self, name: str) -> Optional[Any]:
        """Get property value, returns None if not found."""
        prop = self.properties.get(name)
        return prop.value if prop else None
    
    def set_property(
        self, 
        name: str, 
        value: Any, 
        is_defining: bool = False,
        inherited: bool = False
    ):
        """Set a property."""
        self.properties[name] = ConceptProperty(
            name=name,
            value=value,
            is_defining=is_defining,
            inherited=inherited
        )
    
    def decay_activation(self, rate: float = 0.1):
        """Decay activation towards resting level."""
        self.activation = (
            self.activation * (1 - rate) + 
            self.resting_activation * rate
        )


# ============================================================================
# ONTOLOGY / CONCEPT HIERARCHY
# ============================================================================

class ConceptOntology:
    """
    Hierarchical organization of concepts.
    
    Provides efficient IS-A queries and property inheritance.
    """
    
    def __init__(self):
        # Root concepts for each domain
        self.roots: Dict[str, str] = {}  # category -> root concept ID
        
        # Level index for breadth-first operations
        self.by_level: Dict[int, Set[str]] = defaultdict(set)
    
    def set_root(self, category: str, concept_id: str):
        """Set root concept for a category."""
        self.roots[category] = concept_id
    
    def add_to_level(self, concept_id: str, level: int):
        """Index concept by abstraction level."""
        self.by_level[level].add(concept_id)
    
    def get_ancestors(
        self, 
        concepts: Dict[str, Concept], 
        concept_id: str
    ) -> List[str]:
        """Get all ancestors (parents, grandparents, etc.)."""
        ancestors = []
        visited = set()
        queue = list(concepts.get(concept_id, Concept(id="", name="")).parents)
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ancestors.append(current)
            
            if current in concepts:
                queue.extend(concepts[current].parents)
        
        return ancestors
    
    def get_descendants(
        self, 
        concepts: Dict[str, Concept], 
        concept_id: str
    ) -> List[str]:
        """Get all descendants (children, grandchildren, etc.)."""
        descendants = []
        visited = set()
        queue = list(concepts.get(concept_id, Concept(id="", name="")).children)
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            
            if current in concepts:
                queue.extend(concepts[current].children)
        
        return descendants
    
    def is_ancestor(
        self, 
        concepts: Dict[str, Concept],
        potential_ancestor: str, 
        concept_id: str
    ) -> bool:
        """Check if one concept is an ancestor of another."""
        return potential_ancestor in self.get_ancestors(concepts, concept_id)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """
    Performs reasoning over the semantic network.
    
    Supports:
    - Property inheritance
    - Transitive inference
    - Default reasoning
    - Analogical reasoning
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.inference_cache: Dict[str, Any] = {}
    
    def inherit_property(
        self,
        concepts: Dict[str, Concept],
        ontology: ConceptOntology,
        concept_id: str,
        property_name: str
    ) -> Optional[Tuple[Any, float]]:
        """
        Inherit a property from ancestors.
        
        Returns (value, confidence) or None.
        """
        if concept_id not in concepts:
            return None
        
        concept = concepts[concept_id]
        
        # Check if concept has property directly
        if property_name in concept.properties:
            prop = concept.properties[property_name]
            return (prop.value, prop.confidence)
        
        # Search ancestors (nearest first)
        ancestors = ontology.get_ancestors(concepts, concept_id)
        
        for ancestor_id in ancestors:
            if ancestor_id not in concepts:
                continue
            
            ancestor = concepts[ancestor_id]
            if property_name in ancestor.properties:
                prop = ancestor.properties[property_name]
                # Reduce confidence for inherited properties
                inherited_confidence = prop.confidence * 0.8
                return (prop.value, inherited_confidence)
        
        return None
    
    def transitive_inference(
        self,
        relations: Dict[str, Relation],
        concept_relations: Dict[str, List[str]],
        start: str,
        relation_type: RelationType,
        max_depth: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Follow transitive relations.
        
        E.g., if A IS_A B and B IS_A C, infer A IS_A C.
        """
        results = []
        visited = set()
        queue = [(start, 1.0, 0)]  # (concept, confidence, depth)
        
        while queue:
            current, conf, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            if current in visited:
                continue
            visited.add(current)
            
            # Get relations of this type from current concept
            rel_ids = concept_relations.get(current, [])
            for rel_id in rel_ids:
                if rel_id not in relations:
                    continue
                
                rel = relations[rel_id]
                if rel.relation_type != relation_type:
                    continue
                
                target = rel.target
                new_conf = conf * rel.strength * rel.confidence
                
                if new_conf >= self.confidence_threshold:
                    results.append((target, new_conf))
                    queue.append((target, new_conf, depth + 1))
        
        return results
    
    def infer_relation(
        self,
        concepts: Dict[str, Concept],
        relations: Dict[str, Relation],
        concept_a: str,
        concept_b: str
    ) -> List[Tuple[RelationType, float]]:
        """
        Infer possible relations between two concepts.
        
        Uses multiple inference strategies.
        """
        inferred = []
        
        # Check shared parents (siblings)
        if concept_a in concepts and concept_b in concepts:
            parents_a = concepts[concept_a].parents
            parents_b = concepts[concept_b].parents
            shared_parents = parents_a & parents_b
            
            if shared_parents:
                inferred.append((RelationType.SIMILAR_TO, 0.6))
        
        # Check if one is ancestor of other
        # (would need ontology reference)
        
        return inferred
    
    def clear_cache(self):
        """Clear inference cache."""
        self.inference_cache.clear()


# ============================================================================
# SPREADING ACTIVATION
# ============================================================================

class SpreadingActivation:
    """
    Implements spreading activation for associative retrieval.
    
    Based on Collins & Loftus (1975).
    """
    
    def __init__(
        self,
        decay_rate: float = 0.2,
        firing_threshold: float = 0.3,
        max_iterations: int = 10
    ):
        self.decay_rate = decay_rate
        self.firing_threshold = firing_threshold
        self.max_iterations = max_iterations
    
    def activate(
        self,
        concepts: Dict[str, Concept],
        relations: Dict[str, Relation],
        seed_concepts: List[Tuple[str, float]],  # (concept_id, initial_activation)
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Spread activation from seed concepts.
        
        Returns top-k activated concepts.
        """
        # Initialize activation
        for concept in concepts.values():
            concept.activation = concept.resting_activation
        
        # Set seed activations
        for concept_id, activation in seed_concepts:
            if concept_id in concepts:
                concepts[concept_id].activation = activation
        
        # Spread activation
        for iteration in range(self.max_iterations):
            # Collect activation to spread
            activation_delta: Dict[str, float] = defaultdict(float)
            
            for concept_id, concept in concepts.items():
                if concept.activation < self.firing_threshold:
                    continue
                
                # Spread through relations
                for rel_id in concept.outgoing_relations:
                    if rel_id not in relations:
                        continue
                    
                    rel = relations[rel_id]
                    spread_amount = concept.activation * rel.strength * 0.3
                    activation_delta[rel.target] += spread_amount
                
                # Spread to children
                for child_id in concept.children:
                    spread_amount = concept.activation * 0.4
                    activation_delta[child_id] += spread_amount
                
                # Spread to parents (less)
                for parent_id in concept.parents:
                    spread_amount = concept.activation * 0.2
                    activation_delta[parent_id] += spread_amount
            
            # Apply activation and decay
            for concept_id, concept in concepts.items():
                concept.activation += activation_delta.get(concept_id, 0)
                concept.decay_activation(self.decay_rate)
            
            # Check convergence (optional early stop)
            if not activation_delta:
                break
        
        # Get top-k activated concepts
        activated = [
            (cid, c.activation) 
            for cid, c in concepts.items() 
            if c.activation > self.firing_threshold
        ]
        activated.sort(key=lambda x: x[1], reverse=True)
        
        return activated[:top_k]


# ============================================================================
# ENHANCED SEMANTIC MEMORY - Main Class
# ============================================================================

class EnhancedSemanticMemory:
    """
    Enhanced semantic memory with human-like knowledge organization.
    
    Features:
    - Concept hierarchies (IS-A taxonomies)
    - Multiple relation types
    - Property inheritance
    - Spreading activation for retrieval
    - Inference and reasoning
    - Abstraction and generalization
    """
    
    def __init__(
        self,
        max_concepts: int = 10000,
        max_relations: int = 50000,
        decay_rate: float = 0.005,
        persist_path: Optional[str] = None,
        embedding_fn: Optional[Callable] = None
    ):
        self.max_concepts = max_concepts
        self.max_relations = max_relations
        self.decay_rate = decay_rate
        self.persist_path = persist_path
        self.embedding_fn = embedding_fn
        
        # Storage
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}
        self._concept_count = 0
        self._relation_count = 0
        
        # Indices
        self._by_name: Dict[str, str] = {}  # name -> concept_id
        self._by_category: Dict[str, List[str]] = defaultdict(list)
        self._relations_by_source: Dict[str, List[str]] = defaultdict(list)
        self._relations_by_target: Dict[str, List[str]] = defaultdict(list)
        self._relations_by_type: Dict[RelationType, List[str]] = defaultdict(list)
        
        # Ontology
        self.ontology = ConceptOntology()
        
        # Inference and activation
        self.inference = InferenceEngine()
        self.activation = SpreadingActivation()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        if self.persist_path and os.path.exists(self.persist_path):
            self._load()
        else:
            # Initialize with Minecraft ontology
            self._initialize_minecraft_ontology()
    
    def _generate_concept_id(self) -> str:
        self._concept_count += 1
        return f"concept_{self._concept_count}"
    
    def _generate_relation_id(self) -> str:
        self._relation_count += 1
        return f"rel_{self._relation_count}"
    
    def add_concept(
        self,
        name: str,
        category: str = "general",
        properties: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
        abstraction_level: int = 0
    ) -> str:
        """
        Add a new concept to semantic memory.
        
        Args:
            name: Concept name
            category: Category (entity, item, action, biome, etc.)
            properties: Initial properties
            parents: Parent concept names (for IS-A relations)
            abstraction_level: How abstract (0=concrete)
            
        Returns:
            Concept ID
        """
        with self._lock:
            # Check if exists
            if name in self._by_name:
                return self._by_name[name]
            
            concept_id = self._generate_concept_id()
            
            concept = Concept(
                id=concept_id,
                name=name,
                category=category,
                abstraction_level=abstraction_level
            )
            
            # Add properties
            if properties:
                for prop_name, prop_value in properties.items():
                    concept.set_property(prop_name, prop_value)
            
            # Generate embedding
            if self.embedding_fn:
                try:
                    concept.embedding = self.embedding_fn(name)
                except Exception:
                    pass
            
            # Store
            self.concepts[concept_id] = concept
            self._by_name[name] = concept_id
            self._by_category[category].append(concept_id)
            self.ontology.add_to_level(concept_id, abstraction_level)
            
            # Add parent relations
            if parents:
                for parent_name in parents:
                    parent_id = self._by_name.get(parent_name)
                    if parent_id and parent_id in self.concepts:
                        concept.parents.add(parent_id)
                        self.concepts[parent_id].children.add(concept_id)
                        
                        # Create IS-A relation
                        self.add_relation(
                            source=name,
                            relation_type=RelationType.IS_A,
                            target=parent_name
                        )
            
            # Cleanup if over capacity
            if len(self.concepts) > self.max_concepts:
                self._cleanup_concepts()
            
            return concept_id
    
    def add_relation(
        self,
        source: str,
        relation_type: RelationType,
        target: str,
        strength: float = 1.0,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Add a relation between concepts.
        
        Args:
            source: Source concept name
            relation_type: Type of relation
            target: Target concept name
            strength: Relation strength (0-1)
            confidence: Confidence in relation (0-1)
            metadata: Additional metadata
            
        Returns:
            Relation ID or None if concepts don't exist
        """
        with self._lock:
            source_id = self._by_name.get(source)
            target_id = self._by_name.get(target)
            
            if not source_id or not target_id:
                return None
            
            # Check for existing relation
            existing = self._find_relation(source_id, relation_type, target_id)
            if existing:
                # Reinforce existing relation
                existing.reinforce(strength * 0.1)
                return existing.source  # Return relation ID
            
            relation_id = self._generate_relation_id()
            
            relation = Relation(
                source=source_id,
                relation_type=relation_type,
                target=target_id,
                strength=strength,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            self.relations[relation_id] = relation
            
            # Index
            self._relations_by_source[source_id].append(relation_id)
            self._relations_by_target[target_id].append(relation_id)
            self._relations_by_type[relation_type].append(relation_id)
            
            # Update concept relation lists
            self.concepts[source_id].outgoing_relations.append(relation_id)
            self.concepts[target_id].incoming_relations.append(relation_id)
            
            # Cleanup if over capacity
            if len(self.relations) > self.max_relations:
                self._cleanup_relations()
            
            return relation_id
    
    def _find_relation(
        self, 
        source_id: str, 
        rel_type: RelationType, 
        target_id: str
    ) -> Optional[Relation]:
        """Find existing relation between concepts."""
        for rel_id in self._relations_by_source.get(source_id, []):
            rel = self.relations.get(rel_id)
            if rel and rel.relation_type == rel_type and rel.target == target_id:
                return rel
        return None
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """Get concept by name."""
        with self._lock:
            concept_id = self._by_name.get(name)
            if concept_id:
                concept = self.concepts.get(concept_id)
                if concept:
                    concept.access_count += 1
                return concept
            return None
    
    def get_property(
        self, 
        concept_name: str, 
        property_name: str,
        inherit: bool = True
    ) -> Optional[Any]:
        """
        Get a property of a concept.
        
        Args:
            concept_name: Concept name
            property_name: Property to retrieve
            inherit: Whether to check ancestors
        """
        with self._lock:
            concept_id = self._by_name.get(concept_name)
            if not concept_id:
                return None
            
            concept = self.concepts.get(concept_id)
            if not concept:
                return None
            
            # Check direct property
            if property_name in concept.properties:
                return concept.properties[property_name].value
            
            # Inherit from ancestors
            if inherit:
                result = self.inference.inherit_property(
                    self.concepts, self.ontology, concept_id, property_name
                )
                if result:
                    return result[0]
            
            return None
    
    def get_related(
        self,
        concept_name: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing",  # "outgoing", "incoming", "both"
        limit: int = 20
    ) -> List[Tuple[str, RelationType, float]]:
        """
        Get concepts related to a given concept.
        
        Returns list of (concept_name, relation_type, strength).
        """
        with self._lock:
            concept_id = self._by_name.get(concept_name)
            if not concept_id or concept_id not in self.concepts:
                return []
            
            concept = self.concepts[concept_id]
            results = []
            
            # Outgoing relations
            if direction in ["outgoing", "both"]:
                for rel_id in concept.outgoing_relations:
                    rel = self.relations.get(rel_id)
                    if not rel:
                        continue
                    if relation_type and rel.relation_type != relation_type:
                        continue
                    
                    target_concept = self.concepts.get(rel.target)
                    if target_concept:
                        results.append((
                            target_concept.name,
                            rel.relation_type,
                            rel.strength
                        ))
            
            # Incoming relations
            if direction in ["incoming", "both"]:
                for rel_id in concept.incoming_relations:
                    rel = self.relations.get(rel_id)
                    if not rel:
                        continue
                    if relation_type and rel.relation_type != relation_type:
                        continue
                    
                    source_concept = self.concepts.get(rel.source)
                    if source_concept:
                        results.append((
                            source_concept.name,
                            rel.relation_type,
                            rel.strength
                        ))
            
            # Sort by strength
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:limit]
    
    def query_associative(
        self,
        seed_concepts: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Associative retrieval using spreading activation.
        
        Args:
            seed_concepts: Starting concepts
            top_k: Number of results
            
        Returns:
            List of (concept_name, activation_level)
        """
        with self._lock:
            seeds = []
            for name in seed_concepts:
                concept_id = self._by_name.get(name)
                if concept_id:
                    seeds.append((concept_id, 1.0))
            
            if not seeds:
                return []
            
            activated = self.activation.activate(
                self.concepts,
                self.relations,
                seeds,
                top_k=top_k
            )
            
            # Convert IDs to names
            return [
                (self.concepts[cid].name, activation)
                for cid, activation in activated
                if cid in self.concepts
            ]
    
    def is_a(self, concept_name: str, category_name: str) -> bool:
        """
        Check if concept IS-A category (including through inheritance).
        """
        with self._lock:
            concept_id = self._by_name.get(concept_name)
            category_id = self._by_name.get(category_name)
            
            if not concept_id or not category_id:
                return False
            
            if concept_id == category_id:
                return True
            
            concept = self.concepts.get(concept_id)
            if not concept:
                return False
            
            # Check direct parents
            if category_id in concept.parents:
                return True
            
            # Check transitive
            return self.ontology.is_ancestor(
                self.concepts, category_id, concept_id
            )
    
    def generalize(
        self,
        concept_names: List[str],
        create_if_missing: bool = True
    ) -> Optional[str]:
        """
        Find or create a common generalization of concepts.
        
        E.g., generalize(["diamond_pickaxe", "iron_pickaxe"]) -> "pickaxe"
        """
        with self._lock:
            concept_ids = [
                self._by_name.get(name) 
                for name in concept_names 
                if name in self._by_name
            ]
            
            if len(concept_ids) < 2:
                return None
            
            # Find common ancestors
            ancestor_counts: Dict[str, int] = defaultdict(int)
            
            for cid in concept_ids:
                ancestors = self.ontology.get_ancestors(self.concepts, cid)
                for ancestor in ancestors:
                    ancestor_counts[ancestor] += 1
            
            # Find most specific common ancestor
            common = [
                aid for aid, count in ancestor_counts.items()
                if count == len(concept_ids)
            ]
            
            if common:
                # Return most specific (lowest abstraction level)
                common_concepts = [
                    (aid, self.concepts[aid].abstraction_level)
                    for aid in common
                    if aid in self.concepts
                ]
                common_concepts.sort(key=lambda x: x[1])
                return self.concepts[common_concepts[0][0]].name
            
            return None
    
    def learn_from_example(
        self,
        example: Dict[str, Any],
        category: str
    ):
        """
        Learn properties from an example instance.
        
        Updates typical properties of a category based on examples.
        """
        with self._lock:
            category_id = self._by_name.get(category)
            if not category_id or category_id not in self.concepts:
                return
            
            concept = self.concepts[category_id]
            
            for prop_name, prop_value in example.items():
                if prop_name in concept.properties:
                    # Reinforce existing property
                    prop = concept.properties[prop_name]
                    prop.confidence = min(1.0, prop.confidence + 0.1)
                else:
                    # Add new characteristic property
                    concept.set_property(
                        prop_name, 
                        prop_value, 
                        is_defining=False
                    )
    
    def _initialize_minecraft_ontology(self):
        """Initialize base Minecraft ontology."""
        # Root concepts
        self.add_concept("entity", "root", abstraction_level=3)
        self.add_concept("item", "root", abstraction_level=3)
        self.add_concept("block", "root", abstraction_level=3)
        self.add_concept("biome", "root", abstraction_level=3)
        self.add_concept("action", "root", abstraction_level=3)
        
        # Set ontology roots
        self.ontology.set_root("entity", self._by_name["entity"])
        self.ontology.set_root("item", self._by_name["item"])
        self.ontology.set_root("block", self._by_name["block"])
        
        # Entity hierarchy
        self.add_concept("mob", "entity", parents=["entity"], abstraction_level=2)
        self.add_concept("hostile_mob", "entity", parents=["mob"], abstraction_level=1)
        self.add_concept("passive_mob", "entity", parents=["mob"], abstraction_level=1)
        self.add_concept("player", "entity", parents=["entity"], abstraction_level=0)
        
        # Hostile mobs
        for mob in ["zombie", "skeleton", "creeper", "spider", "enderman"]:
            self.add_concept(mob, "entity", parents=["hostile_mob"])
            self.add_relation(mob, RelationType.CAUSES, "damage")
        
        # Passive mobs
        for mob in ["cow", "sheep", "pig", "chicken"]:
            self.add_concept(mob, "entity", parents=["passive_mob"])
        
        # Item hierarchy
        self.add_concept("tool", "item", parents=["item"], abstraction_level=2)
        self.add_concept("weapon", "item", parents=["item"], abstraction_level=2)
        self.add_concept("armor", "item", parents=["item"], abstraction_level=2)
        self.add_concept("food", "item", parents=["item"], abstraction_level=2)
        self.add_concept("material", "item", parents=["item"], abstraction_level=2)
        
        # Tools
        self.add_concept("pickaxe", "item", parents=["tool"], abstraction_level=1)
        self.add_concept("axe", "item", parents=["tool"], abstraction_level=1)
        self.add_concept("shovel", "item", parents=["tool"], abstraction_level=1)
        self.add_concept("sword", "item", parents=["weapon"], abstraction_level=1)
        
        # Tool tiers
        for material in ["wooden", "stone", "iron", "diamond"]:
            for tool in ["pickaxe", "axe", "shovel", "sword"]:
                name = f"{material}_{tool}"
                self.add_concept(name, "item", parents=[tool])
                self.add_relation(name, RelationType.MADE_OF, material)
        
        # Materials
        for material in ["wood", "stone", "iron", "gold", "diamond", "coal"]:
            self.add_concept(material, "item", parents=["material"])
        
        # Blocks
        self.add_concept("ore", "block", parents=["block"], abstraction_level=1)
        for ore in ["coal_ore", "iron_ore", "gold_ore", "diamond_ore"]:
            self.add_concept(ore, "block", parents=["ore"])
        
        # Biomes
        for biome in ["plains", "forest", "desert", "mountains", "ocean", "swamp"]:
            self.add_concept(biome, "biome", parents=["biome"])
        
        # Actions
        for action in ["mine", "craft", "build", "fight", "explore", "eat"]:
            self.add_concept(action, "action", parents=["action"])
        
        # Relations
        self.add_relation("pickaxe", RelationType.USED_FOR, "mine")
        self.add_relation("axe", RelationType.USED_FOR, "chop")
        self.add_relation("sword", RelationType.USED_FOR, "fight")
        self.add_relation("armor", RelationType.PREVENTS, "damage")
        self.add_relation("food", RelationType.USED_FOR, "eat")
        self.add_relation("diamond", RelationType.FOUND_NEAR, "bedrock")
    
    def _cleanup_concepts(self):
        """Remove least accessed concepts."""
        # Score concepts for retention
        scored = []
        for cid, concept in self.concepts.items():
            # Don't remove root/high-level concepts
            if concept.abstraction_level >= 2:
                continue
            
            # Score based on access and relations
            score = concept.access_count + len(concept.outgoing_relations) * 2
            scored.append((score, cid))
        
        scored.sort(key=lambda x: x[0])
        
        # Remove lowest scoring
        to_remove = len(self.concepts) - int(self.max_concepts * 0.9)
        for i in range(min(to_remove, len(scored))):
            self._remove_concept(scored[i][1])
    
    def _cleanup_relations(self):
        """Remove weak relations."""
        scored = []
        for rid, rel in self.relations.items():
            score = rel.strength * rel.confidence * (1 + rel.access_count * 0.1)
            scored.append((score, rid))
        
        scored.sort(key=lambda x: x[0])
        
        to_remove = len(self.relations) - int(self.max_relations * 0.9)
        for i in range(min(to_remove, len(scored))):
            self._remove_relation(scored[i][1])
    
    def _remove_concept(self, concept_id: str):
        """Remove a concept and its relations."""
        if concept_id not in self.concepts:
            return
        
        concept = self.concepts[concept_id]
        
        # Remove relations
        for rel_id in concept.outgoing_relations + concept.incoming_relations:
            self._remove_relation(rel_id)
        
        # Remove from indices
        if concept.name in self._by_name:
            del self._by_name[concept.name]
        
        if concept_id in self._by_category.get(concept.category, []):
            self._by_category[concept.category].remove(concept_id)
        
        # Remove from parent/child relations
        for parent_id in concept.parents:
            if parent_id in self.concepts:
                self.concepts[parent_id].children.discard(concept_id)
        
        for child_id in concept.children:
            if child_id in self.concepts:
                self.concepts[child_id].parents.discard(concept_id)
        
        del self.concepts[concept_id]
    
    def _remove_relation(self, relation_id: str):
        """Remove a relation."""
        if relation_id not in self.relations:
            return
        
        rel = self.relations[relation_id]
        
        # Remove from indices
        if relation_id in self._relations_by_source.get(rel.source, []):
            self._relations_by_source[rel.source].remove(relation_id)
        if relation_id in self._relations_by_target.get(rel.target, []):
            self._relations_by_target[rel.target].remove(relation_id)
        if relation_id in self._relations_by_type.get(rel.relation_type, []):
            self._relations_by_type[rel.relation_type].remove(relation_id)
        
        # Remove from concepts
        if rel.source in self.concepts:
            concept = self.concepts[rel.source]
            if relation_id in concept.outgoing_relations:
                concept.outgoing_relations.remove(relation_id)
        
        if rel.target in self.concepts:
            concept = self.concepts[rel.target]
            if relation_id in concept.incoming_relations:
                concept.incoming_relations.remove(relation_id)
        
        del self.relations[relation_id]
    
    def apply_decay(self):
        """Apply decay to relations."""
        with self._lock:
            for rel in self.relations.values():
                rel.decay(self.decay_rate)
    
    def save(self):
        """Save to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            try:
                data = {
                    'concepts': {},
                    'relations': {},
                    'concept_count': self._concept_count,
                    'relation_count': self._relation_count
                }
                
                for cid, concept in self.concepts.items():
                    data['concepts'][cid] = {
                        'id': concept.id,
                        'name': concept.name,
                        'category': concept.category,
                        'properties': {
                            k: {'value': v.value, 'is_defining': v.is_defining}
                            for k, v in concept.properties.items()
                        },
                        'parents': list(concept.parents),
                        'children': list(concept.children),
                        'abstraction_level': concept.abstraction_level,
                        'access_count': concept.access_count
                    }
                
                for rid, rel in self.relations.items():
                    data['relations'][rid] = {
                        'source': rel.source,
                        'relation_type': rel.relation_type.name,
                        'target': rel.target,
                        'strength': rel.strength,
                        'confidence': rel.confidence
                    }
                
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save semantic memory: {e}")
    
    def _load(self):
        """Load from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self._concept_count = data.get('concept_count', 0)
            self._relation_count = data.get('relation_count', 0)
            
            # Load concepts
            for cid, cdata in data.get('concepts', {}).items():
                concept = Concept(
                    id=cdata['id'],
                    name=cdata['name'],
                    category=cdata['category'],
                    parents=set(cdata.get('parents', [])),
                    children=set(cdata.get('children', [])),
                    abstraction_level=cdata.get('abstraction_level', 0),
                    access_count=cdata.get('access_count', 0)
                )
                
                for pname, pdata in cdata.get('properties', {}).items():
                    concept.set_property(
                        pname, 
                        pdata['value'], 
                        is_defining=pdata.get('is_defining', False)
                    )
                
                self.concepts[cid] = concept
                self._by_name[concept.name] = cid
                self._by_category[concept.category].append(cid)
            
            # Load relations
            for rid, rdata in data.get('relations', {}).items():
                rel = Relation(
                    source=rdata['source'],
                    relation_type=RelationType[rdata['relation_type']],
                    target=rdata['target'],
                    strength=rdata.get('strength', 1.0),
                    confidence=rdata.get('confidence', 1.0)
                )
                
                self.relations[rid] = rel
                self._relations_by_source[rel.source].append(rid)
                self._relations_by_target[rel.target].append(rid)
                self._relations_by_type[rel.relation_type].append(rid)
                
                # Update concept relation lists
                if rel.source in self.concepts:
                    self.concepts[rel.source].outgoing_relations.append(rid)
                if rel.target in self.concepts:
                    self.concepts[rel.target].incoming_relations.append(rid)
                    
        except Exception as e:
            print(f"Failed to load semantic memory: {e}")
            self._initialize_minecraft_ontology()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                'total_concepts': len(self.concepts),
                'total_relations': len(self.relations),
                'concepts_by_category': {
                    cat: len(ids) for cat, ids in self._by_category.items()
                },
                'relations_by_type': {
                    rt.name: len(ids) for rt, ids in self._relations_by_type.items()
                }
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_semantic_instance: Optional[EnhancedSemanticMemory] = None
_semantic_lock = threading.Lock()


def get_semantic_memory(
    persist_path: Optional[str] = None,
    **kwargs
) -> EnhancedSemanticMemory:
    """Get or create the global enhanced semantic memory instance."""
    global _semantic_instance
    
    with _semantic_lock:
        if _semantic_instance is None:
            _semantic_instance = EnhancedSemanticMemory(
                persist_path=persist_path or os.path.expanduser(
                    "~/.voyager_evolved/semantic_memory.json"
                ),
                **kwargs
            )
        return _semantic_instance


def reset_semantic_memory():
    """Reset the global semantic memory instance."""
    global _semantic_instance
    
    with _semantic_lock:
        if _semantic_instance:
            _semantic_instance.save()
        _semantic_instance = None
