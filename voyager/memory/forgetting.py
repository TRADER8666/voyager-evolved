"""Forgetting curve implementation.

Implements Ebbinghaus forgetting curve for realistic memory decay.
"""

import logging
import time
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

import numpy as np

from voyager.memory.vector_store import VectorStore, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class MemoryStrength:
    """Tracks the strength of a memory over time."""
    memory_id: str
    initial_strength: float = 1.0
    current_strength: float = 1.0
    stability: float = 1.0  # How resistant to forgetting
    
    # Review history
    creation_time: float = field(default_factory=time.time)
    last_review: float = field(default_factory=time.time)
    review_count: int = 0
    review_history: List[float] = field(default_factory=list)  # Timestamps
    
    # Factors affecting retention
    importance: float = 0.5
    emotional_strength: float = 0.0  # Strong emotions improve retention
    rehearsal_count: int = 0
    
    def calculate_retention(self, current_time: Optional[float] = None) -> float:
        """Calculate current retention using Ebbinghaus curve.
        
        R = e^(-t/S) where:
        - R = retention (0-1)
        - t = time since last review
        - S = stability (increases with reviews)
        """
        if current_time is None:
            current_time = time.time()
        
        time_elapsed = current_time - self.last_review
        time_hours = time_elapsed / 3600  # Convert to hours
        
        # Modified forgetting curve
        # Higher stability = slower forgetting
        effective_stability = self.stability * (
            1 + self.importance * 0.5 +
            abs(self.emotional_strength) * 0.3 +
            min(self.rehearsal_count * 0.1, 0.5)
        )
        
        # Ebbinghaus formula
        retention = math.exp(-time_hours / (effective_stability * 24))  # Scale to days
        
        return max(0.0, min(1.0, retention))
    
    def review(self, success: bool = True) -> None:
        """Record a review/recall of this memory.
        
        Successful reviews increase stability (spaced repetition).
        """
        current_time = time.time()
        
        if success:
            # Increase stability based on spacing
            time_since_last = current_time - self.last_review
            spacing_bonus = min(time_since_last / (24 * 3600), 1.0)  # Max 1 day
            
            self.stability *= (1 + 0.5 * spacing_bonus)  # Up to 50% increase
            self.stability = min(self.stability, 10.0)  # Cap stability
            
            self.current_strength = min(1.0, self.current_strength + 0.2)
        else:
            # Failed recall - minor stability decrease
            self.stability *= 0.9
        
        self.review_count += 1
        self.last_review = current_time
        self.review_history.append(current_time)
        
        # Keep only last 10 reviews
        if len(self.review_history) > 10:
            self.review_history = self.review_history[-10:]
    
    def get_optimal_review_time(self) -> float:
        """Calculate optimal time for next review.
        
        Based on when retention will drop to ~90%.
        """
        target_retention = 0.9
        
        # Solve for t: 0.9 = e^(-t/S)
        # t = -S * ln(0.9)
        time_hours = -self.stability * 24 * math.log(target_retention)
        
        return self.last_review + (time_hours * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'initial_strength': self.initial_strength,
            'current_strength': self.current_strength,
            'stability': self.stability,
            'creation_time': self.creation_time,
            'last_review': self.last_review,
            'review_count': self.review_count,
            'review_history': self.review_history,
            'importance': self.importance,
            'emotional_strength': self.emotional_strength,
            'rehearsal_count': self.rehearsal_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryStrength':
        return cls(
            memory_id=data['memory_id'],
            initial_strength=data.get('initial_strength', 1.0),
            current_strength=data.get('current_strength', 1.0),
            stability=data.get('stability', 1.0),
            creation_time=data.get('creation_time', time.time()),
            last_review=data.get('last_review', time.time()),
            review_count=data.get('review_count', 0),
            review_history=data.get('review_history', []),
            importance=data.get('importance', 0.5),
            emotional_strength=data.get('emotional_strength', 0.0),
            rehearsal_count=data.get('rehearsal_count', 0),
        )


class ForgettingCurve:
    """Manages forgetting for all memories.
    
    Features:
    - Ebbinghaus forgetting curve
    - Spaced repetition scheduling
    - Importance-weighted retention
    - Emotional memory bonus
    - Automatic pruning of forgotten memories
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        base_decay_rate: float = 0.1,
        rehearsal_boost: float = 0.3,
        importance_factor: float = 0.5,
        emotional_factor: float = 0.3,
        min_strength: float = 0.01,
        review_intervals: Optional[List[int]] = None,
    ):
        self.vector_store = vector_store
        self.base_decay_rate = base_decay_rate
        self.rehearsal_boost = rehearsal_boost
        self.importance_factor = importance_factor
        self.emotional_factor = emotional_factor
        self.min_strength = min_strength
        self.review_intervals = review_intervals or [1, 3, 7, 14, 30]  # days
        
        self._memory_strengths: Dict[str, MemoryStrength] = {}
        self._lock = threading.Lock()
        
        self._stats = {
            'memories_forgotten': 0,
            'reviews_scheduled': 0,
            'total_reviews': 0,
        }
    
    def register_memory(
        self,
        memory_id: str,
        importance: float = 0.5,
        emotional_strength: float = 0.0,
    ) -> MemoryStrength:
        """Register a new memory for forgetting tracking."""
        with self._lock:
            strength = MemoryStrength(
                memory_id=memory_id,
                importance=importance,
                emotional_strength=emotional_strength,
            )
            self._memory_strengths[memory_id] = strength
            return strength
    
    def get_retention(self, memory_id: str) -> float:
        """Get current retention level for a memory."""
        with self._lock:
            if memory_id not in self._memory_strengths:
                # Unknown memory, assume full retention
                return 1.0
            
            return self._memory_strengths[memory_id].calculate_retention()
    
    def record_access(
        self,
        memory_id: str,
        successful_recall: bool = True,
    ) -> None:
        """Record that a memory was accessed/recalled."""
        with self._lock:
            if memory_id in self._memory_strengths:
                self._memory_strengths[memory_id].review(successful_recall)
                self._stats['total_reviews'] += 1
    
    def apply_forgetting(
        self,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> Dict[str, Any]:
        """Apply forgetting to memories and remove forgotten ones.
        
        Returns:
            Statistics about the forgetting process
        """
        with self._lock:
            results = {
                'memories_checked': 0,
                'memories_weakened': 0,
                'memories_forgotten': 0,
            }
            
            # Get memories from vector store
            if memory_types:
                for mt in memory_types:
                    self._apply_forgetting_to_type(mt, results)
            else:
                for mt in MemoryType:
                    self._apply_forgetting_to_type(mt, results)
            
            return results
    
    def _apply_forgetting_to_type(
        self,
        memory_type: MemoryType,
        results: Dict[str, int],
    ) -> None:
        """Apply forgetting to a specific memory type."""
        memories = self.vector_store.get_by_type(memory_type, limit=1000)
        
        to_delete = []
        
        for memory in memories:
            results['memories_checked'] += 1
            
            # Get or create strength tracker
            if memory.id not in self._memory_strengths:
                self._memory_strengths[memory.id] = MemoryStrength(
                    memory_id=memory.id,
                    importance=memory.importance,
                    emotional_strength=memory.emotional_valence,
                    creation_time=memory.timestamp,
                    last_review=memory.last_accessed,
                )
            
            strength = self._memory_strengths[memory.id]
            retention = strength.calculate_retention()
            
            # Update strength in vector store
            new_strength = retention * strength.current_strength
            
            if new_strength < self.min_strength:
                # Memory forgotten
                to_delete.append(memory.id)
                results['memories_forgotten'] += 1
            elif new_strength < memory.strength:
                # Memory weakened
                self.vector_store.update(
                    memory.id,
                    strength=new_strength,
                )
                results['memories_weakened'] += 1
        
        # Delete forgotten memories
        if to_delete:
            self.vector_store.delete_batch(to_delete)
            for mid in to_delete:
                if mid in self._memory_strengths:
                    del self._memory_strengths[mid]
            
            self._stats['memories_forgotten'] += len(to_delete)
    
    def get_memories_due_for_review(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get memories that are due for review.
        
        Returns:
            List of (memory_id, current_retention) tuples
        """
        current_time = time.time()
        due_for_review = []
        
        with self._lock:
            for memory_id, strength in self._memory_strengths.items():
                # Check if review is due
                optimal_time = strength.get_optimal_review_time()
                
                if current_time >= optimal_time:
                    retention = strength.calculate_retention(current_time)
                    
                    # Only include if retention is dropping
                    if retention < 0.95:
                        due_for_review.append((memory_id, retention))
        
        # Sort by retention (lowest first = most urgent)
        due_for_review.sort(key=lambda x: x[1])
        
        return due_for_review[:limit]
    
    def get_spaced_repetition_schedule(
        self,
        memory_id: str,
    ) -> List[float]:
        """Get suggested review schedule for a memory."""
        if memory_id not in self._memory_strengths:
            return []
        
        strength = self._memory_strengths[memory_id]
        current_time = time.time()
        
        schedule = []
        review_time = current_time
        
        for interval_days in self.review_intervals:
            review_time = current_time + (interval_days * 24 * 3600)
            schedule.append(review_time)
        
        return schedule
    
    def estimate_long_term_retention(
        self,
        memory_id: str,
        days_ahead: int = 30,
    ) -> List[Tuple[int, float]]:
        """Estimate retention over time.
        
        Returns:
            List of (days, retention) tuples
        """
        if memory_id not in self._memory_strengths:
            return []
        
        strength = self._memory_strengths[memory_id]
        current_time = time.time()
        
        retention_curve = []
        for day in range(days_ahead + 1):
            future_time = current_time + (day * 24 * 3600)
            retention = strength.calculate_retention(future_time)
            retention_curve.append((day, retention))
        
        return retention_curve
    
    def boost_memory(
        self,
        memory_id: str,
        boost_amount: float = 0.2,
    ) -> None:
        """Artificially boost a memory's strength."""
        with self._lock:
            if memory_id in self._memory_strengths:
                strength = self._memory_strengths[memory_id]
                strength.current_strength = min(1.0, strength.current_strength + boost_amount)
                strength.stability *= 1.2  # Also increase stability
    
    def get_memory_health(
        self,
        memory_id: str,
    ) -> Dict[str, Any]:
        """Get detailed health information about a memory."""
        if memory_id not in self._memory_strengths:
            return {'status': 'unknown'}
        
        strength = self._memory_strengths[memory_id]
        retention = strength.calculate_retention()
        
        if retention > 0.9:
            status = 'strong'
        elif retention > 0.7:
            status = 'healthy'
        elif retention > 0.5:
            status = 'fading'
        elif retention > 0.2:
            status = 'weak'
        else:
            status = 'nearly_forgotten'
        
        return {
            'status': status,
            'retention': retention,
            'stability': strength.stability,
            'review_count': strength.review_count,
            'days_since_review': (time.time() - strength.last_review) / (24 * 3600),
            'next_review_due': strength.get_optimal_review_time(),
            'importance': strength.importance,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get forgetting system statistics."""
        retentions = [
            s.calculate_retention()
            for s in self._memory_strengths.values()
        ]
        
        return {
            **self._stats,
            'tracked_memories': len(self._memory_strengths),
            'avg_retention': np.mean(retentions) if retentions else 1.0,
            'min_retention': min(retentions) if retentions else 1.0,
        }
