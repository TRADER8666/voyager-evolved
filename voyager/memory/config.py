"""Memory system configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import os


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    collection_name: str = "voyager_memory"
    persist_directory: str = "memory_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    distance_metric: str = "cosine"  # cosine, l2, ip
    max_entries: int = 100000
    

@dataclass
class ExperienceConfig:
    """Experience memory configuration."""
    max_experiences: int = 50000
    experience_ttl_days: int = 30  # Auto-delete old experiences
    importance_threshold: float = 0.3
    enable_auto_tagging: bool = True
    enable_emotion_tracking: bool = True


@dataclass
class SemanticConfig:
    """Semantic memory configuration."""
    max_concepts: int = 10000
    min_relation_strength: float = 0.3
    enable_inference: bool = True
    concept_decay_rate: float = 0.99


@dataclass
class ConsolidationConfig:
    """Memory consolidation configuration."""
    enabled: bool = True
    consolidation_interval: int = 3600  # seconds
    similarity_threshold: float = 0.85
    min_cluster_size: int = 3
    max_cluster_size: int = 20
    enable_abstraction: bool = True


@dataclass
class ForgettingConfig:
    """Forgetting curve configuration."""
    enabled: bool = True
    base_decay_rate: float = 0.1
    rehearsal_boost: float = 0.3
    importance_factor: float = 0.5
    emotional_factor: float = 0.3
    min_strength: float = 0.01  # Delete below this
    review_intervals: list = field(default_factory=lambda: [1, 3, 7, 14, 30])  # days


@dataclass
class MemoryConfig:
    """Main memory system configuration."""
    enabled: bool = True
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    
    # Performance settings
    async_operations: bool = True
    batch_size: int = 100
    cache_size: int = 1000
    
    # Persistence
    auto_save: bool = True
    save_interval: int = 300  # seconds
    backup_enabled: bool = True
    max_backups: int = 5
    
    def validate(self) -> list:
        """Validate configuration."""
        errors = []
        
        if self.vector_store.max_entries < 1000:
            errors.append("Vector store max_entries should be at least 1000")
        
        if self.consolidation.similarity_threshold < 0.5:
            errors.append("Consolidation similarity threshold too low")
        
        if self.forgetting.base_decay_rate <= 0 or self.forgetting.base_decay_rate >= 1:
            errors.append("Forgetting decay rate must be between 0 and 1")
        
        return errors
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MemoryConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'vector_store': {
                'collection_name': self.vector_store.collection_name,
                'persist_directory': self.vector_store.persist_directory,
                'embedding_model': self.vector_store.embedding_model,
                'embedding_dim': self.vector_store.embedding_dim,
                'distance_metric': self.vector_store.distance_metric,
                'max_entries': self.vector_store.max_entries,
            },
            'experience': {
                'max_experiences': self.experience.max_experiences,
                'experience_ttl_days': self.experience.experience_ttl_days,
                'importance_threshold': self.experience.importance_threshold,
                'enable_auto_tagging': self.experience.enable_auto_tagging,
                'enable_emotion_tracking': self.experience.enable_emotion_tracking,
            },
            'semantic': {
                'max_concepts': self.semantic.max_concepts,
                'min_relation_strength': self.semantic.min_relation_strength,
                'enable_inference': self.semantic.enable_inference,
                'concept_decay_rate': self.semantic.concept_decay_rate,
            },
            'consolidation': {
                'enabled': self.consolidation.enabled,
                'consolidation_interval': self.consolidation.consolidation_interval,
                'similarity_threshold': self.consolidation.similarity_threshold,
                'min_cluster_size': self.consolidation.min_cluster_size,
                'max_cluster_size': self.consolidation.max_cluster_size,
                'enable_abstraction': self.consolidation.enable_abstraction,
            },
            'forgetting': {
                'enabled': self.forgetting.enabled,
                'base_decay_rate': self.forgetting.base_decay_rate,
                'rehearsal_boost': self.forgetting.rehearsal_boost,
                'importance_factor': self.forgetting.importance_factor,
                'emotional_factor': self.forgetting.emotional_factor,
                'min_strength': self.forgetting.min_strength,
                'review_intervals': self.forgetting.review_intervals,
            },
            'async_operations': self.async_operations,
            'batch_size': self.batch_size,
            'cache_size': self.cache_size,
            'auto_save': self.auto_save,
            'save_interval': self.save_interval,
            'backup_enabled': self.backup_enabled,
            'max_backups': self.max_backups,
        }
    
    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> 'MemoryConfig':
        """Create from dictionary."""
        return cls(
            enabled=d.get('enabled', True),
            vector_store=VectorStoreConfig(
                collection_name=d.get('vector_store', {}).get('collection_name', 'voyager_memory'),
                persist_directory=d.get('vector_store', {}).get('persist_directory', 'memory_db'),
                embedding_model=d.get('vector_store', {}).get('embedding_model', 'all-MiniLM-L6-v2'),
                embedding_dim=d.get('vector_store', {}).get('embedding_dim', 384),
                distance_metric=d.get('vector_store', {}).get('distance_metric', 'cosine'),
                max_entries=d.get('vector_store', {}).get('max_entries', 100000),
            ),
            experience=ExperienceConfig(
                max_experiences=d.get('experience', {}).get('max_experiences', 50000),
                experience_ttl_days=d.get('experience', {}).get('experience_ttl_days', 30),
                importance_threshold=d.get('experience', {}).get('importance_threshold', 0.3),
                enable_auto_tagging=d.get('experience', {}).get('enable_auto_tagging', True),
                enable_emotion_tracking=d.get('experience', {}).get('enable_emotion_tracking', True),
            ),
            semantic=SemanticConfig(
                max_concepts=d.get('semantic', {}).get('max_concepts', 10000),
                min_relation_strength=d.get('semantic', {}).get('min_relation_strength', 0.3),
                enable_inference=d.get('semantic', {}).get('enable_inference', True),
                concept_decay_rate=d.get('semantic', {}).get('concept_decay_rate', 0.99),
            ),
            consolidation=ConsolidationConfig(
                enabled=d.get('consolidation', {}).get('enabled', True),
                consolidation_interval=d.get('consolidation', {}).get('consolidation_interval', 3600),
                similarity_threshold=d.get('consolidation', {}).get('similarity_threshold', 0.85),
                min_cluster_size=d.get('consolidation', {}).get('min_cluster_size', 3),
                max_cluster_size=d.get('consolidation', {}).get('max_cluster_size', 20),
                enable_abstraction=d.get('consolidation', {}).get('enable_abstraction', True),
            ),
            forgetting=ForgettingConfig(
                enabled=d.get('forgetting', {}).get('enabled', True),
                base_decay_rate=d.get('forgetting', {}).get('base_decay_rate', 0.1),
                rehearsal_boost=d.get('forgetting', {}).get('rehearsal_boost', 0.3),
                importance_factor=d.get('forgetting', {}).get('importance_factor', 0.5),
                emotional_factor=d.get('forgetting', {}).get('emotional_factor', 0.3),
                min_strength=d.get('forgetting', {}).get('min_strength', 0.01),
                review_intervals=d.get('forgetting', {}).get('review_intervals', [1, 3, 7, 14, 30]),
            ),
            async_operations=d.get('async_operations', True),
            batch_size=d.get('batch_size', 100),
            cache_size=d.get('cache_size', 1000),
            auto_save=d.get('auto_save', True),
            save_interval=d.get('save_interval', 300),
            backup_enabled=d.get('backup_enabled', True),
            max_backups=d.get('max_backups', 5),
        )
    
    @classmethod
    def create_high_performance(cls) -> 'MemoryConfig':
        """Create high-performance config for powerful hardware."""
        config = cls()
        config.vector_store.max_entries = 500000
        config.experience.max_experiences = 100000
        config.semantic.max_concepts = 50000
        config.batch_size = 200
        config.cache_size = 5000
        return config
    
    @classmethod
    def create_minimal(cls) -> 'MemoryConfig':
        """Create minimal config for constrained systems."""
        config = cls()
        config.vector_store.max_entries = 10000
        config.experience.max_experiences = 5000
        config.semantic.max_concepts = 1000
        config.consolidation.enabled = False
        config.batch_size = 50
        config.cache_size = 500
        return config
