"""Semantic memory for concepts and knowledge.

Stores abstract knowledge and relationships between concepts.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import numpy as np

from voyager.memory.vector_store import VectorStore, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of concept relationships."""
    IS_A = "is_a"              # Hierarchical (dog IS_A animal)
    HAS_A = "has_a"            # Composition (house HAS_A door)
    PART_OF = "part_of"        # Part relationship (wheel PART_OF car)
    USED_FOR = "used_for"      # Purpose (pickaxe USED_FOR mining)
    FOUND_IN = "found_in"      # Location (diamond FOUND_IN deep caves)
    MADE_FROM = "made_from"    # Material (sword MADE_FROM iron)
    CAUSES = "causes"          # Causation (fire CAUSES damage)
    REQUIRES = "requires"      # Prerequisites (iron_pickaxe REQUIRES iron)
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Opposition
    RELATED_TO = "related_to"  # General relation


@dataclass
class ConceptRelation:
    """A relationship between two concepts."""
    source: str
    target: str
    relation_type: RelationType
    strength: float = 1.0  # 0-1
    confidence: float = 1.0  # 0-1
    evidence_count: int = 1
    timestamp: float = field(default_factory=time.time)
    
    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce this relation."""
        self.strength = min(1.0, self.strength + amount)
        self.evidence_count += 1
        self.timestamp = time.time()
    
    def decay(self, rate: float = 0.01) -> None:
        """Apply decay to relation strength."""
        self.strength *= (1 - rate)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptRelation':
        return cls(
            source=data['source'],
            target=data['target'],
            relation_type=RelationType(data['relation_type']),
            strength=data.get('strength', 1.0),
            confidence=data.get('confidence', 1.0),
            evidence_count=data.get('evidence_count', 1),
            timestamp=data.get('timestamp', time.time()),
        )


@dataclass
class SemanticConcept:
    """A semantic concept with properties and relations."""
    id: str
    name: str
    category: str  # block, entity, item, biome, action, etc.
    description: str = ""
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Relations
    relations: List[ConceptRelation] = field(default_factory=list)
    
    # Memory metadata
    activation: float = 1.0  # Current activation level
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created: float = field(default_factory=time.time)
    
    # Learning
    learned_from: List[str] = field(default_factory=list)  # Experience IDs
    
    def activate(self, amount: float = 0.2) -> None:
        """Activate this concept."""
        self.activation = min(1.0, self.activation + amount)
        self.access_count += 1
        self.last_accessed = time.time()
    
    def decay(self, rate: float = 0.01) -> None:
        """Apply decay to activation."""
        self.activation *= (1 - rate)
    
    def add_relation(
        self,
        target: str,
        relation_type: RelationType,
        strength: float = 1.0,
    ) -> ConceptRelation:
        """Add or reinforce a relation."""
        # Check if relation exists
        for rel in self.relations:
            if rel.target == target and rel.relation_type == relation_type:
                rel.reinforce()
                return rel
        
        # Create new relation
        rel = ConceptRelation(
            source=self.name,
            target=target,
            relation_type=relation_type,
            strength=strength,
        )
        self.relations.append(rel)
        return rel
    
    def get_relations(self, relation_type: Optional[RelationType] = None) -> List[ConceptRelation]:
        """Get relations, optionally filtered by type."""
        if relation_type is None:
            return self.relations
        return [r for r in self.relations if r.relation_type == relation_type]
    
    def to_text(self) -> str:
        """Convert to searchable text."""
        parts = [self.name, self.category]
        if self.description:
            parts.append(self.description)
        
        for key, value in self.properties.items():
            parts.append(f"{key}: {value}")
        
        for rel in self.relations[:5]:  # Limit relations in text
            parts.append(f"{rel.relation_type.value} {rel.target}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'properties': self.properties,
            'relations': [r.to_dict() for r in self.relations],
            'activation': self.activation,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'created': self.created,
            'learned_from': self.learned_from,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticConcept':
        return cls(
            id=data['id'],
            name=data['name'],
            category=data['category'],
            description=data.get('description', ''),
            properties=data.get('properties', {}),
            relations=[ConceptRelation.from_dict(r) for r in data.get('relations', [])],
            activation=data.get('activation', 1.0),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            created=data.get('created', time.time()),
            learned_from=data.get('learned_from', []),
        )


class SemanticMemory:
    """Semantic memory system for concepts and knowledge.
    
    Features:
    - Store and retrieve concepts
    - Manage concept relationships
    - Spreading activation for association
    - Inference and reasoning
    - Knowledge graph operations
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        max_concepts: int = 10000,
        min_relation_strength: float = 0.3,
        enable_inference: bool = True,
        decay_rate: float = 0.99,
    ):
        self.vector_store = vector_store
        self.max_concepts = max_concepts
        self.min_relation_strength = min_relation_strength
        self.enable_inference = enable_inference
        self.decay_rate = decay_rate
        
        self._concepts: Dict[str, SemanticConcept] = {}
        self._name_to_id: Dict[str, str] = {}
        self._lock = threading.Lock()
        
        # Load existing concepts
        self._load_concepts()
    
    def _load_concepts(self) -> None:
        """Load concepts from vector store."""
        try:
            entries = self.vector_store.get_by_type(
                MemoryType.CONCEPT,
                limit=self.max_concepts
            )
            
            for entry in entries:
                try:
                    concept_data = entry.metadata.get('concept_data')
                    if concept_data:
                        if isinstance(concept_data, str):
                            concept_data = json.loads(concept_data)
                        concept = SemanticConcept.from_dict(concept_data)
                        self._concepts[concept.id] = concept
                        self._name_to_id[concept.name.lower()] = concept.id
                except Exception as e:
                    logger.debug(f"Could not load concept: {e}")
            
            logger.info(f"Loaded {len(self._concepts)} concepts")
            
        except Exception as e:
            logger.warning(f"Error loading concepts: {e}")
    
    def add_concept(
        self,
        name: str,
        category: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> SemanticConcept:
        """Add or update a concept.
        
        Args:
            name: Concept name
            category: Concept category
            description: Description
            properties: Concept properties
            importance: Importance score
        
        Returns:
            Created or updated concept
        """
        import uuid
        
        with self._lock:
            # Check if concept exists
            existing_id = self._name_to_id.get(name.lower())
            if existing_id:
                concept = self._concepts[existing_id]
                concept.activate()
                if description:
                    concept.description = description
                if properties:
                    concept.properties.update(properties)
                return concept
            
            # Create new concept
            concept = SemanticConcept(
                id=str(uuid.uuid4()),
                name=name,
                category=category,
                description=description,
                properties=properties or {},
                importance=importance,
            )
            
            # Store in vector database
            self.vector_store.add(
                content=concept.to_text(),
                memory_type=MemoryType.CONCEPT,
                importance=importance,
                metadata={'concept_data': json.dumps(concept.to_dict())},
                entry_id=concept.id,
            )
            
            # Cache locally
            self._concepts[concept.id] = concept
            self._name_to_id[name.lower()] = concept.id
            
            logger.debug(f"Added concept: {name}")
            
            return concept
    
    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation_type: RelationType,
        strength: float = 1.0,
        bidirectional: bool = False,
    ) -> Optional[ConceptRelation]:
        """Add a relation between concepts.
        
        Args:
            source_name: Source concept name
            target_name: Target concept name
            relation_type: Type of relation
            strength: Relation strength
            bidirectional: Add reverse relation too
        
        Returns:
            Created relation or None
        """
        with self._lock:
            source_id = self._name_to_id.get(source_name.lower())
            target_id = self._name_to_id.get(target_name.lower())
            
            if not source_id:
                # Auto-create source concept
                source = self.add_concept(source_name, "unknown")
                source_id = source.id
            
            if not target_id:
                # Auto-create target concept
                target = self.add_concept(target_name, "unknown")
                target_id = target.id
            
            source_concept = self._concepts[source_id]
            rel = source_concept.add_relation(target_name, relation_type, strength)
            
            # Add bidirectional
            if bidirectional:
                target_concept = self._concepts[target_id]
                reverse_type = self._get_reverse_relation(relation_type)
                target_concept.add_relation(source_name, reverse_type, strength)
            
            return rel
    
    def _get_reverse_relation(self, relation_type: RelationType) -> RelationType:
        """Get reverse relation type."""
        reverse_map = {
            RelationType.IS_A: RelationType.HAS_A,
            RelationType.HAS_A: RelationType.PART_OF,
            RelationType.PART_OF: RelationType.HAS_A,
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,
            RelationType.OPPOSITE_OF: RelationType.OPPOSITE_OF,
            RelationType.RELATED_TO: RelationType.RELATED_TO,
        }
        return reverse_map.get(relation_type, RelationType.RELATED_TO)
    
    def get_concept(self, name: str) -> Optional[SemanticConcept]:
        """Get concept by name."""
        with self._lock:
            concept_id = self._name_to_id.get(name.lower())
            if concept_id:
                concept = self._concepts[concept_id]
                concept.activate()
                return concept
            return None
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        category: Optional[str] = None,
    ) -> List[Tuple[SemanticConcept, float]]:
        """Search for concepts."""
        with self._lock:
            results = self.vector_store.search(
                query=query,
                n_results=n_results,
                memory_types=[MemoryType.CONCEPT],
            )
            
            concepts = []
            for entry, score in results:
                try:
                    concept_data = entry.metadata.get('concept_data')
                    if concept_data:
                        if isinstance(concept_data, str):
                            concept_data = json.loads(concept_data)
                        concept = SemanticConcept.from_dict(concept_data)
                        
                        if category and concept.category != category:
                            continue
                        
                        concept.activate()
                        concepts.append((concept, score))
                        
                except Exception as e:
                    logger.debug(f"Error processing concept: {e}")
            
            return concepts
    
    def spreading_activation(
        self,
        start_concepts: List[str],
        depth: int = 2,
        decay: float = 0.5,
    ) -> Dict[str, float]:
        """Perform spreading activation from starting concepts.
        
        Args:
            start_concepts: Names of starting concepts
            depth: How many hops to spread
            decay: Activation decay per hop
        
        Returns:
            Dict mapping concept names to activation levels
        """
        activations: Dict[str, float] = {}
        
        # Initialize starting concepts
        current_level = {}
        for name in start_concepts:
            current_level[name.lower()] = 1.0
            activations[name.lower()] = 1.0
        
        # Spread activation
        for _ in range(depth):
            next_level = {}
            
            for name, activation in current_level.items():
                concept = self.get_concept(name)
                if not concept:
                    continue
                
                for rel in concept.relations:
                    if rel.strength < self.min_relation_strength:
                        continue
                    
                    target_name = rel.target.lower()
                    spread_activation = activation * decay * rel.strength
                    
                    if target_name in activations:
                        activations[target_name] = max(
                            activations[target_name],
                            spread_activation
                        )
                    else:
                        activations[target_name] = spread_activation
                    
                    if target_name not in current_level:
                        next_level[target_name] = spread_activation
            
            current_level = next_level
        
        return activations
    
    def get_related_concepts(
        self,
        name: str,
        relation_types: Optional[List[RelationType]] = None,
        min_strength: float = 0.3,
    ) -> List[Tuple[str, RelationType, float]]:
        """Get concepts related to a given concept."""
        concept = self.get_concept(name)
        if not concept:
            return []
        
        results = []
        for rel in concept.relations:
            if rel.strength < min_strength:
                continue
            if relation_types and rel.relation_type not in relation_types:
                continue
            results.append((rel.target, rel.relation_type, rel.strength))
        
        return sorted(results, key=lambda x: x[2], reverse=True)
    
    def infer_relations(
        self,
        source_name: str,
        target_name: str,
    ) -> List[Tuple[RelationType, float]]:
        """Infer possible relations between two concepts."""
        if not self.enable_inference:
            return []
        
        source = self.get_concept(source_name)
        target = self.get_concept(target_name)
        
        if not source or not target:
            return []
        
        inferences = []
        
        # Direct relations
        for rel in source.relations:
            if rel.target.lower() == target_name.lower():
                inferences.append((rel.relation_type, rel.strength))
        
        # Transitive IS_A inference
        for rel in source.get_relations(RelationType.IS_A):
            parent = self.get_concept(rel.target)
            if parent:
                for parent_rel in parent.relations:
                    if parent_rel.target.lower() == target_name.lower():
                        confidence = rel.strength * parent_rel.strength * 0.8
                        if confidence > 0.3:
                            inferences.append((parent_rel.relation_type, confidence))
        
        return sorted(inferences, key=lambda x: x[1], reverse=True)
    
    def apply_decay(self) -> int:
        """Apply decay to all concepts and relations."""
        with self._lock:
            decayed = 0
            
            for concept in list(self._concepts.values()):
                concept.decay(1 - self.decay_rate)
                
                # Decay relations
                weak_relations = []
                for rel in concept.relations:
                    rel.decay(1 - self.decay_rate)
                    if rel.strength < self.min_relation_strength:
                        weak_relations.append(rel)
                
                # Remove weak relations
                for rel in weak_relations:
                    concept.relations.remove(rel)
                    decayed += 1
            
            return decayed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_relations = sum(
            len(c.relations) for c in self._concepts.values()
        )
        
        return {
            'total_concepts': len(self._concepts),
            'total_relations': total_relations,
            'categories': list(set(
                c.category for c in self._concepts.values()
            )),
        }
