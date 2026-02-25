"""Memory module for Voyager Evolved.

Comprehensive cognitive memory system featuring:
- Vector-based long-term memory (ChromaDB)
- Working memory (7±2 item capacity, human-like)
- Enhanced episodic memory (autobiographical, temporal, emotional)
- Enhanced semantic memory (concept hierarchies, inference)
"""

from voyager.memory.vector_store import (
    VectorStore,
    MemoryEntry,
    MemoryType,
    get_vector_store,
)
from voyager.memory.experience import (
    ExperienceMemory,
    Experience,
    ExperienceType,
)
from voyager.memory.semantic import (
    SemanticMemory,
    SemanticConcept,
    ConceptRelation,
)
from voyager.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationStrategy,
)
from voyager.memory.forgetting import (
    ForgettingCurve,
    MemoryStrength,
)
from voyager.memory.config import MemoryConfig

# Phase 1B: Enhanced Cognitive Memory Systems
from voyager.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryItem,
    MemoryItemType,
    PhonologicalLoop,
    VisuospatialSketchpad,
    EpisodicBuffer,
    CentralExecutive,
    get_working_memory,
    reset_working_memory,
)
from voyager.memory.episodic_enhanced import (
    EnhancedEpisodicMemory,
    Episode,
    EpisodeType,
    EpisodeChain,
    EmotionalValence,
    EmotionalTag,
    TemporalContext,
    SpatialContext,
    AutobiographicalMemory,
    get_episodic_memory,
    reset_episodic_memory,
)
from voyager.memory.semantic_enhanced import (
    EnhancedSemanticMemory,
    Concept,
    ConceptProperty,
    Relation,
    RelationType,
    ConceptOntology,
    InferenceEngine,
    SpreadingActivation,
    get_semantic_memory,
    reset_semantic_memory,
)

__all__ = [
    # Vector Store
    'VectorStore',
    'MemoryEntry',
    'MemoryType',
    'get_vector_store',
    # Experience (basic)
    'ExperienceMemory',
    'Experience',
    'ExperienceType',
    # Semantic (basic)
    'SemanticMemory',
    'SemanticConcept',
    'ConceptRelation',
    # Consolidation
    'MemoryConsolidator',
    'ConsolidationStrategy',
    # Forgetting
    'ForgettingCurve',
    'MemoryStrength',
    # Config
    'MemoryConfig',
    
    # Phase 1B: Working Memory
    'WorkingMemory',
    'WorkingMemoryItem',
    'MemoryItemType',
    'PhonologicalLoop',
    'VisuospatialSketchpad',
    'EpisodicBuffer',
    'CentralExecutive',
    'get_working_memory',
    'reset_working_memory',
    
    # Phase 1B: Enhanced Episodic Memory
    'EnhancedEpisodicMemory',
    'Episode',
    'EpisodeType',
    'EpisodeChain',
    'EmotionalValence',
    'EmotionalTag',
    'TemporalContext',
    'SpatialContext',
    'AutobiographicalMemory',
    'get_episodic_memory',
    'reset_episodic_memory',
    
    # Phase 1B: Enhanced Semantic Memory
    'EnhancedSemanticMemory',
    'Concept',
    'ConceptProperty',
    'Relation',
    'RelationType',
    'ConceptOntology',
    'InferenceEngine',
    'SpreadingActivation',
    'get_semantic_memory',
    'reset_semantic_memory',
]

__version__ = '1.1.0'
