"""Vector store implementation using ChromaDB.

Provides persistent vector storage for all memory types.
"""

import logging
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import os

import numpy as np

logger = logging.getLogger(__name__)

# Global vector store instance
_vector_store: Optional['VectorStore'] = None
_store_lock = threading.Lock()


class MemoryType(Enum):
    """Types of memory stored in vector database."""
    EXPERIENCE = "experience"      # Episodic experiences
    OBSERVATION = "observation"    # Visual/sensory observations
    SKILL = "skill"               # Learned skills
    CONCEPT = "concept"           # Semantic concepts
    LOCATION = "location"         # Places and landmarks
    PLAYER = "player"             # Other players
    ENTITY = "entity"             # Entities encountered
    EVENT = "event"               # Significant events


@dataclass
class MemoryEntry:
    """A single entry in the vector store."""
    id: str
    memory_type: MemoryType
    content: str  # Text description
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0  # Memory strength (forgetting curve)
    importance: float = 0.5
    emotional_valence: float = 0.0  # -1 to 1
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Record an access to this memory."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'strength': self.strength,
            'importance': self.importance,
            'emotional_valence': self.emotional_valence,
            'tags': self.tags,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            memory_type=MemoryType(data['memory_type']),
            content=data['content'],
            embedding=embedding,
            timestamp=data.get('timestamp', time.time()),
            last_accessed=data.get('last_accessed', time.time()),
            access_count=data.get('access_count', 0),
            strength=data.get('strength', 1.0),
            importance=data.get('importance', 0.5),
            emotional_valence=data.get('emotional_valence', 0.0),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
        )


class VectorStore:
    """ChromaDB-based vector store for memory.
    
    Features:
    - Persistent vector storage
    - Semantic similarity search
    - Metadata filtering
    - Batch operations
    - Multiple collections for different memory types
    """
    
    def __init__(
        self,
        collection_name: str = "voyager_memory",
        persist_directory: str = "memory_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        distance_metric: str = "cosine",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        self._client = None
        self._collection = None
        self._embedder = None
        self._lock = threading.Lock()
        self._initialized = False
        
        self._stats = {
            'total_entries': 0,
            'queries': 0,
            'inserts': 0,
        }
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize ChromaDB and embedding model."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            self._stats['total_entries'] = self._collection.count()
            
            logger.info(
                f"ChromaDB initialized: {self.collection_name} "
                f"({self._stats['total_entries']} entries)"
            )
            
            # Initialize embedding model
            self._initialize_embedder()
            
            self._initialized = True
            
        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def _initialize_embedder(self) -> None:
        """Initialize the sentence embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._embedder = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded embedding model: {self.embedding_model}")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback embeddings."
            )
            self._embedder = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self._embedder is not None:
            return self._embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Repeat to fill embedding dimension
            repeated = (hash_bytes * (self.embedding_dim // len(hash_bytes) + 1))[:self.embedding_dim]
            embedding = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
            # Normalize
            return embedding / (np.linalg.norm(embedding) + 1e-10)
    
    def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Add a memory entry.
        
        Args:
            content: Text content of memory
            memory_type: Type of memory
            importance: Importance score (0-1)
            emotional_valence: Emotional association (-1 to 1)
            tags: Optional tags for filtering
            metadata: Additional metadata
            entry_id: Optional custom ID
        
        Returns:
            Created MemoryEntry
        """
        if not self._initialized:
            logger.warning("Vector store not initialized")
            return None
        
        with self._lock:
            # Generate ID if not provided
            if entry_id is None:
                entry_id = str(uuid.uuid4())
            
            # Create embedding
            embedding = self._get_embedding(content)
            
            # Create entry
            entry = MemoryEntry(
                id=entry_id,
                memory_type=memory_type,
                content=content,
                embedding=embedding,
                importance=importance,
                emotional_valence=emotional_valence,
                tags=tags or [],
                metadata=metadata or {},
            )
            
            # Store in ChromaDB
            self._collection.add(
                ids=[entry_id],
                embeddings=[embedding.tolist()],
                metadatas=[entry.to_dict()],
                documents=[content],
            )
            
            self._stats['inserts'] += 1
            self._stats['total_entries'] += 1
            
            return entry
    
    def add_batch(
        self,
        entries: List[Tuple[str, MemoryType, Dict[str, Any]]],
    ) -> List[MemoryEntry]:
        """Add multiple entries in batch.
        
        Args:
            entries: List of (content, memory_type, kwargs) tuples
        
        Returns:
            List of created MemoryEntry objects
        """
        if not self._initialized:
            return []
        
        with self._lock:
            results = []
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for content, memory_type, kwargs in entries:
                entry_id = kwargs.get('entry_id', str(uuid.uuid4()))
                embedding = self._get_embedding(content)
                
                entry = MemoryEntry(
                    id=entry_id,
                    memory_type=memory_type,
                    content=content,
                    embedding=embedding,
                    importance=kwargs.get('importance', 0.5),
                    emotional_valence=kwargs.get('emotional_valence', 0.0),
                    tags=kwargs.get('tags', []),
                    metadata=kwargs.get('metadata', {}),
                )
                
                results.append(entry)
                ids.append(entry_id)
                embeddings.append(embedding.tolist())
                metadatas.append(entry.to_dict())
                documents.append(content)
            
            # Batch add to ChromaDB
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            
            self._stats['inserts'] += len(entries)
            self._stats['total_entries'] += len(entries)
            
            return results
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            tags: Filter by tags (any match)
            time_range: Filter by timestamp range (start, end)
        
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        if not self._initialized:
            return []
        
        with self._lock:
            # Build where clause for filtering
            where = {}
            where_document = None
            
            if memory_types:
                if len(memory_types) == 1:
                    where['memory_type'] = memory_types[0].value
                else:
                    where['memory_type'] = {'$in': [mt.value for mt in memory_types]}
            
            if min_importance > 0:
                where['importance'] = {'$gte': min_importance}
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where if where else None,
            )
            
            self._stats['queries'] += 1
            
            # Process results
            entries = []
            if results['ids'] and results['ids'][0]:
                for i, entry_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    # Convert distance to similarity
                    if self.distance_metric == 'cosine':
                        similarity = 1 - distance
                    else:
                        similarity = 1 / (1 + distance)
                    
                    entry = MemoryEntry.from_dict(metadata)
                    entry.access()  # Record access
                    
                    # Apply additional filters
                    if time_range:
                        if not (time_range[0] <= entry.timestamp <= time_range[1]):
                            continue
                    
                    if tags:
                        if not any(tag in entry.tags for tag in tags):
                            continue
                    
                    entries.append((entry, similarity))
            
            return entries
    
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        if not self._initialized:
            return None
        
        with self._lock:
            result = self._collection.get(
                ids=[entry_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if result['ids']:
                metadata = result['metadatas'][0]
                embedding = np.array(result['embeddings'][0]) if result['embeddings'] else None
                return MemoryEntry.from_dict(metadata, embedding)
            
            return None
    
    def update(
        self,
        entry_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        strength: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing entry."""
        if not self._initialized:
            return False
        
        with self._lock:
            # Get existing entry
            entry = self.get_by_id(entry_id)
            if not entry:
                return False
            
            # Update fields
            if content is not None:
                entry.content = content
                entry.embedding = self._get_embedding(content)
            if importance is not None:
                entry.importance = importance
            if strength is not None:
                entry.strength = strength
            if tags is not None:
                entry.tags = tags
            if metadata is not None:
                entry.metadata.update(metadata)
            
            # Update in ChromaDB
            self._collection.update(
                ids=[entry_id],
                embeddings=[entry.embedding.tolist()] if entry.embedding is not None else None,
                metadatas=[entry.to_dict()],
                documents=[entry.content],
            )
            
            return True
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        if not self._initialized:
            return False
        
        with self._lock:
            try:
                self._collection.delete(ids=[entry_id])
                self._stats['total_entries'] -= 1
                return True
            except:
                return False
    
    def delete_batch(self, entry_ids: List[str]) -> int:
        """Delete multiple entries."""
        if not self._initialized:
            return 0
        
        with self._lock:
            try:
                self._collection.delete(ids=entry_ids)
                deleted = len(entry_ids)
                self._stats['total_entries'] -= deleted
                return deleted
            except:
                return 0
    
    def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """Get all entries of a specific type."""
        if not self._initialized:
            return []
        
        with self._lock:
            results = self._collection.get(
                where={'memory_type': memory_type.value},
                limit=limit,
                include=['metadatas']
            )
            
            entries = []
            for metadata in results['metadatas']:
                entries.append(MemoryEntry.from_dict(metadata))
            
            return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            **self._stats,
            'initialized': self._initialized,
            'collection_name': self.collection_name,
        }
    
    def clear(self) -> None:
        """Clear all entries."""
        if not self._initialized:
            return
        
        with self._lock:
            # Delete and recreate collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            self._stats['total_entries'] = 0


def get_vector_store(
    collection_name: str = "voyager_memory",
    persist_directory: str = "memory_db",
    **kwargs
) -> VectorStore:
    """Get or create global vector store instance."""
    global _vector_store
    
    with _store_lock:
        if _vector_store is None:
            _vector_store = VectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                **kwargs
            )
        return _vector_store


def reset_vector_store() -> None:
    """Reset global vector store."""
    global _vector_store
    
    with _store_lock:
        _vector_store = None
