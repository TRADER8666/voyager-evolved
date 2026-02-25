"""Memory module for Voyager Evolved.

Vector-based long-term memory system using ChromaDB.
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

__all__ = [
    # Vector Store
    'VectorStore',
    'MemoryEntry',
    'MemoryType',
    'get_vector_store',
    # Experience
    'ExperienceMemory',
    'Experience',
    'ExperienceType',
    # Semantic
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
]

__version__ = '1.0.0'
