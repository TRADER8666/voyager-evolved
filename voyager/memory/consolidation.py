"""Memory consolidation system.

Merges similar memories and creates abstractions.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import numpy as np

from voyager.memory.vector_store import VectorStore, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""
    MERGE = "merge"           # Merge similar memories into one
    ABSTRACT = "abstract"     # Create abstraction from cluster
    COMPRESS = "compress"     # Compress redundant information
    PRUNE = "prune"          # Remove low-value memories


@dataclass
class MemoryCluster:
    """A cluster of similar memories."""
    cluster_id: str
    memories: List[str]  # Memory IDs
    centroid: Optional[np.ndarray] = None
    avg_similarity: float = 0.0
    created: float = field(default_factory=time.time)
    
    # Abstraction
    abstraction: Optional[str] = None  # Abstract summary
    common_themes: List[str] = field(default_factory=list)


class MemoryConsolidator:
    """Consolidates memories to reduce redundancy.
    
    Features:
    - Cluster similar memories
    - Merge redundant memories
    - Create abstractions from patterns
    - Compress old memories
    - Remove low-value memories
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        max_cluster_size: int = 20,
        enable_abstraction: bool = True,
    ):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.enable_abstraction = enable_abstraction
        
        self._clusters: Dict[str, MemoryCluster] = {}
        self._lock = threading.Lock()
        
        self._stats = {
            'total_consolidations': 0,
            'memories_merged': 0,
            'abstractions_created': 0,
            'memories_pruned': 0,
        }
    
    def consolidate(
        self,
        memory_type: Optional[MemoryType] = None,
        strategy: ConsolidationStrategy = ConsolidationStrategy.MERGE,
        max_memories: int = 1000,
    ) -> Dict[str, Any]:
        """Run consolidation on memories.
        
        Args:
            memory_type: Type of memories to consolidate
            strategy: Consolidation strategy
            max_memories: Max memories to process
        
        Returns:
            Consolidation results
        """
        with self._lock:
            results = {
                'clusters_found': 0,
                'memories_affected': 0,
                'strategy': strategy.value,
            }
            
            # Get memories to consolidate
            if memory_type:
                memories = self.vector_store.get_by_type(memory_type, limit=max_memories)
            else:
                # Get all recent memories
                memories = []
                for mt in MemoryType:
                    memories.extend(
                        self.vector_store.get_by_type(mt, limit=max_memories // len(MemoryType))
                    )
            
            if len(memories) < self.min_cluster_size:
                return results
            
            # Find clusters of similar memories
            clusters = self._find_clusters(memories)
            results['clusters_found'] = len(clusters)
            
            # Apply consolidation strategy
            for cluster in clusters:
                if strategy == ConsolidationStrategy.MERGE:
                    affected = self._merge_cluster(cluster)
                elif strategy == ConsolidationStrategy.ABSTRACT:
                    affected = self._abstract_cluster(cluster)
                elif strategy == ConsolidationStrategy.COMPRESS:
                    affected = self._compress_cluster(cluster)
                elif strategy == ConsolidationStrategy.PRUNE:
                    affected = self._prune_cluster(cluster)
                else:
                    affected = 0
                
                results['memories_affected'] += affected
            
            self._stats['total_consolidations'] += 1
            
            return results
    
    def _find_clusters(
        self,
        memories: List[MemoryEntry],
    ) -> List[MemoryCluster]:
        """Find clusters of similar memories using DBSCAN-like approach."""
        import uuid
        
        if not memories:
            return []
        
        # Get embeddings
        embeddings = []
        memory_ids = []
        
        for memory in memories:
            # Get embedding from vector store
            entry = self.vector_store.get_by_id(memory.id)
            if entry and entry.embedding is not None:
                embeddings.append(entry.embedding)
                memory_ids.append(memory.id)
        
        if len(embeddings) < self.min_cluster_size:
            return []
        
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Simple clustering: find groups above threshold
        clusters = []
        used = set()
        
        for i in range(len(memory_ids)):
            if i in used:
                continue
            
            # Find similar memories
            similar = []
            for j in range(len(memory_ids)):
                if j not in used and similarity_matrix[i, j] >= self.similarity_threshold:
                    similar.append(j)
            
            if len(similar) >= self.min_cluster_size:
                # Create cluster
                cluster_memories = [memory_ids[j] for j in similar[:self.max_cluster_size]]
                cluster_embeddings = embeddings[similar[:self.max_cluster_size]]
                
                cluster = MemoryCluster(
                    cluster_id=str(uuid.uuid4()),
                    memories=cluster_memories,
                    centroid=np.mean(cluster_embeddings, axis=0),
                    avg_similarity=float(np.mean([
                        similarity_matrix[i, j]
                        for j in similar[:self.max_cluster_size]
                    ])),
                )
                
                clusters.append(cluster)
                used.update(similar[:self.max_cluster_size])
        
        return clusters
    
    def _merge_cluster(self, cluster: MemoryCluster) -> int:
        """Merge memories in a cluster into one."""
        if len(cluster.memories) < 2:
            return 0
        
        # Get all memory contents
        contents = []
        total_importance = 0
        total_strength = 0
        
        for memory_id in cluster.memories:
            entry = self.vector_store.get_by_id(memory_id)
            if entry:
                contents.append(entry.content)
                total_importance += entry.importance
                total_strength += entry.strength
        
        if not contents:
            return 0
        
        # Create merged content (take first as base, add unique info from others)
        merged_content = contents[0]
        for content in contents[1:]:
            # Simple deduplication - could be improved with LLM summarization
            if content not in merged_content:
                # Add unique parts
                unique_parts = [p for p in content.split('|') if p.strip() not in merged_content]
                if unique_parts:
                    merged_content += " | " + " | ".join(unique_parts[:2])
        
        # Create merged memory
        import uuid
        merged_id = str(uuid.uuid4())
        
        # Determine memory type from first entry
        first_entry = self.vector_store.get_by_id(cluster.memories[0])
        memory_type = MemoryType(first_entry.memory_type.value) if first_entry else MemoryType.EXPERIENCE
        
        self.vector_store.add(
            content=merged_content[:2000],  # Limit length
            memory_type=memory_type,
            importance=total_importance / len(cluster.memories),
            metadata={
                'merged_from': cluster.memories,
                'merge_time': time.time(),
            },
            entry_id=merged_id,
        )
        
        # Delete original memories
        self.vector_store.delete_batch(cluster.memories)
        
        self._stats['memories_merged'] += len(cluster.memories)
        
        return len(cluster.memories)
    
    def _abstract_cluster(self, cluster: MemoryCluster) -> int:
        """Create an abstraction from a cluster of memories."""
        if not self.enable_abstraction:
            return 0
        
        # Get all memory contents
        contents = []
        tags = set()
        
        for memory_id in cluster.memories:
            entry = self.vector_store.get_by_id(memory_id)
            if entry:
                contents.append(entry.content)
                tags.update(entry.tags)
        
        if not contents:
            return 0
        
        # Extract common themes (simple approach)
        words = {}
        for content in contents:
            for word in content.lower().split():
                if len(word) > 3:
                    words[word] = words.get(word, 0) + 1
        
        # Find words that appear in most memories
        common_words = [
            word for word, count in words.items()
            if count >= len(contents) * 0.5  # Appears in at least half
        ][:10]
        
        cluster.common_themes = common_words
        
        # Create abstraction
        abstraction = f"Pattern ({len(cluster.memories)} instances): {' '.join(common_words)}"
        cluster.abstraction = abstraction
        
        # Store abstraction as new concept
        import uuid
        self.vector_store.add(
            content=abstraction,
            memory_type=MemoryType.CONCEPT,
            importance=0.7,  # Abstractions are valuable
            tags=list(tags)[:5],
            metadata={
                'is_abstraction': True,
                'source_memories': cluster.memories,
                'common_themes': common_words,
            },
            entry_id=str(uuid.uuid4()),
        )
        
        self._clusters[cluster.cluster_id] = cluster
        self._stats['abstractions_created'] += 1
        
        return len(cluster.memories)
    
    def _compress_cluster(self, cluster: MemoryCluster) -> int:
        """Compress memories in cluster by reducing detail."""
        affected = 0
        
        for memory_id in cluster.memories:
            entry = self.vector_store.get_by_id(memory_id)
            if entry:
                # Shorten content
                original_length = len(entry.content)
                compressed = entry.content[:original_length // 2]
                
                # Reduce importance of compressed memories
                self.vector_store.update(
                    memory_id,
                    content=compressed,
                    importance=entry.importance * 0.8,
                )
                affected += 1
        
        return affected
    
    def _prune_cluster(self, cluster: MemoryCluster) -> int:
        """Prune low-value memories from cluster, keeping the best."""
        if len(cluster.memories) <= self.min_cluster_size:
            return 0
        
        # Score memories
        scored = []
        for memory_id in cluster.memories:
            entry = self.vector_store.get_by_id(memory_id)
            if entry:
                score = entry.importance * entry.strength * (1 + entry.access_count * 0.1)
                scored.append((memory_id, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top memories, prune rest
        to_keep = [m[0] for m in scored[:self.min_cluster_size]]
        to_prune = [m[0] for m in scored[self.min_cluster_size:]]
        
        if to_prune:
            self.vector_store.delete_batch(to_prune)
            self._stats['memories_pruned'] += len(to_prune)
        
        return len(to_prune)
    
    def find_redundant_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of highly similar memories.
        
        Returns:
            List of (memory_id_1, memory_id_2, similarity) tuples
        """
        if memory_type:
            memories = self.vector_store.get_by_type(memory_type, limit=limit)
        else:
            memories = []
            for mt in MemoryType:
                memories.extend(
                    self.vector_store.get_by_type(mt, limit=limit // len(MemoryType))
                )
        
        redundant = []
        checked = set()
        
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                if (mem1.id, mem2.id) in checked:
                    continue
                
                # Search for similar memories
                results = self.vector_store.search(
                    mem1.content,
                    n_results=5
                )
                
                for entry, similarity in results:
                    if entry.id == mem2.id and similarity >= self.similarity_threshold:
                        redundant.append((mem1.id, mem2.id, similarity))
                
                checked.add((mem1.id, mem2.id))
        
        return sorted(redundant, key=lambda x: x[2], reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            **self._stats,
            'active_clusters': len(self._clusters),
        }
