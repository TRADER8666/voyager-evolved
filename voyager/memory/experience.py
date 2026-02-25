"""Experience memory for episodic memories.

Stores and retrieves experiences (what happened) for the AI agent.
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


class ExperienceType(Enum):
    """Types of experiences."""
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    DISCOVERY = "discovery"
    ENCOUNTER = "encounter"
    COMBAT = "combat"
    CRAFTING = "crafting"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    DEATH = "death"
    MILESTONE = "milestone"


@dataclass
class Experience:
    """A single experience/episode."""
    id: str
    experience_type: ExperienceType
    description: str
    timestamp: float = field(default_factory=time.time)
    
    # Context
    location: Optional[Dict[str, Any]] = None  # x, y, z, biome
    task: Optional[str] = None
    outcome: Optional[str] = None
    
    # Emotional/evaluative
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.5  # 0 (calm) to 1 (excited)
    importance: float = 0.5
    
    # Related entities
    entities_involved: List[str] = field(default_factory=list)
    players_involved: List[str] = field(default_factory=list)
    items_involved: List[str] = field(default_factory=list)
    
    # Learning
    lessons_learned: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    skills_learned: List[str] = field(default_factory=list)
    
    # Memory metadata
    recall_count: int = 0
    last_recalled: Optional[float] = None
    strength: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    def recall(self) -> None:
        """Record that this experience was recalled."""
        self.recall_count += 1
        self.last_recalled = time.time()
        # Strengthen memory on recall
        self.strength = min(1.0, self.strength + 0.1)
    
    def to_text(self) -> str:
        """Convert to searchable text."""
        parts = [self.description]
        
        if self.task:
            parts.append(f"Task: {self.task}")
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        if self.location:
            loc = self.location
            parts.append(f"Location: {loc.get('biome', 'unknown')}")
        if self.lessons_learned:
            parts.append(f"Learned: {', '.join(self.lessons_learned)}")
        if self.entities_involved:
            parts.append(f"Entities: {', '.join(self.entities_involved)}")
        if self.players_involved:
            parts.append(f"Players: {', '.join(self.players_involved)}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'experience_type': self.experience_type.value,
            'description': self.description,
            'timestamp': self.timestamp,
            'location': self.location,
            'task': self.task,
            'outcome': self.outcome,
            'valence': self.valence,
            'arousal': self.arousal,
            'importance': self.importance,
            'entities_involved': self.entities_involved,
            'players_involved': self.players_involved,
            'items_involved': self.items_involved,
            'lessons_learned': self.lessons_learned,
            'skills_used': self.skills_used,
            'skills_learned': self.skills_learned,
            'recall_count': self.recall_count,
            'last_recalled': self.last_recalled,
            'strength': self.strength,
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            experience_type=ExperienceType(data['experience_type']),
            description=data['description'],
            timestamp=data.get('timestamp', time.time()),
            location=data.get('location'),
            task=data.get('task'),
            outcome=data.get('outcome'),
            valence=data.get('valence', 0.0),
            arousal=data.get('arousal', 0.5),
            importance=data.get('importance', 0.5),
            entities_involved=data.get('entities_involved', []),
            players_involved=data.get('players_involved', []),
            items_involved=data.get('items_involved', []),
            lessons_learned=data.get('lessons_learned', []),
            skills_used=data.get('skills_used', []),
            skills_learned=data.get('skills_learned', []),
            recall_count=data.get('recall_count', 0),
            last_recalled=data.get('last_recalled'),
            strength=data.get('strength', 1.0),
            tags=data.get('tags', []),
        )


class ExperienceMemory:
    """Episodic memory system for experiences.
    
    Features:
    - Store and retrieve experiences
    - Contextual search (location, task, entities)
    - Emotional filtering
    - Learning from past experiences
    - Experience summarization
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        max_experiences: int = 50000,
        importance_threshold: float = 0.3,
        enable_auto_tagging: bool = True,
    ):
        self.vector_store = vector_store
        self.max_experiences = max_experiences
        self.importance_threshold = importance_threshold
        self.enable_auto_tagging = enable_auto_tagging
        
        self._experiences: Dict[str, Experience] = {}
        self._lock = threading.Lock()
        
        # Load existing experiences
        self._load_experiences()
    
    def _load_experiences(self) -> None:
        """Load experiences from vector store."""
        try:
            entries = self.vector_store.get_by_type(
                MemoryType.EXPERIENCE,
                limit=self.max_experiences
            )
            
            for entry in entries:
                try:
                    exp_data = entry.metadata.get('experience_data')
                    if exp_data:
                        if isinstance(exp_data, str):
                            exp_data = json.loads(exp_data)
                        exp = Experience.from_dict(exp_data)
                        self._experiences[exp.id] = exp
                except Exception as e:
                    logger.debug(f"Could not load experience: {e}")
            
            logger.info(f"Loaded {len(self._experiences)} experiences")
            
        except Exception as e:
            logger.warning(f"Error loading experiences: {e}")
    
    def record(
        self,
        description: str,
        experience_type: ExperienceType,
        location: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        outcome: Optional[str] = None,
        valence: float = 0.0,
        arousal: float = 0.5,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
        players: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        lessons: Optional[List[str]] = None,
        skills_used: Optional[List[str]] = None,
        skills_learned: Optional[List[str]] = None,
    ) -> Experience:
        """Record a new experience.
        
        Args:
            description: Natural language description
            experience_type: Type of experience
            location: Optional location data
            task: Optional related task
            outcome: Optional outcome description
            valence: Emotional valence (-1 to 1)
            arousal: Emotional arousal (0 to 1)
            importance: Importance score (0 to 1)
            entities: Entities involved
            players: Players involved
            items: Items involved
            lessons: Lessons learned
            skills_used: Skills that were used
            skills_learned: New skills learned
        
        Returns:
            Created Experience
        """
        import uuid
        
        with self._lock:
            # Create experience
            exp = Experience(
                id=str(uuid.uuid4()),
                experience_type=experience_type,
                description=description,
                location=location,
                task=task,
                outcome=outcome,
                valence=valence,
                arousal=arousal,
                importance=importance,
                entities_involved=entities or [],
                players_involved=players or [],
                items_involved=items or [],
                lessons_learned=lessons or [],
                skills_used=skills_used or [],
                skills_learned=skills_learned or [],
            )
            
            # Auto-tagging
            if self.enable_auto_tagging:
                exp.tags = self._generate_tags(exp)
            
            # Store in vector database
            self.vector_store.add(
                content=exp.to_text(),
                memory_type=MemoryType.EXPERIENCE,
                importance=importance,
                emotional_valence=valence,
                tags=exp.tags,
                metadata={'experience_data': json.dumps(exp.to_dict())},
                entry_id=exp.id,
            )
            
            # Cache locally
            self._experiences[exp.id] = exp
            
            # Cleanup if over limit
            if len(self._experiences) > self.max_experiences:
                self._cleanup_old_experiences()
            
            logger.debug(f"Recorded experience: {exp.experience_type.value}")
            
            return exp
    
    def _generate_tags(self, exp: Experience) -> List[str]:
        """Generate tags for an experience."""
        tags = [exp.experience_type.value]
        
        # Location-based tags
        if exp.location:
            biome = exp.location.get('biome')
            if biome:
                tags.append(f"biome:{biome}")
        
        # Entity-based tags
        for entity in exp.entities_involved[:3]:
            tags.append(f"entity:{entity}")
        
        # Player-based tags
        for player in exp.players_involved[:3]:
            tags.append(f"player:{player}")
        
        # Emotional tags
        if exp.valence > 0.5:
            tags.append("positive")
        elif exp.valence < -0.5:
            tags.append("negative")
        
        if exp.importance > 0.7:
            tags.append("important")
        
        return tags
    
    def recall(
        self,
        query: str,
        n_results: int = 5,
        experience_types: Optional[List[ExperienceType]] = None,
        min_importance: float = 0.0,
        location: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[Experience, float]]:
        """Recall relevant experiences.
        
        Args:
            query: Search query
            n_results: Number of results
            experience_types: Filter by types
            min_importance: Minimum importance
            location: Filter by location (biome)
            time_range: Filter by time
        
        Returns:
            List of (Experience, relevance_score) tuples
        """
        with self._lock:
            # Search vector store
            results = self.vector_store.search(
                query=query,
                n_results=n_results * 2,  # Get more for filtering
                memory_types=[MemoryType.EXPERIENCE],
                min_importance=min_importance,
                time_range=time_range,
            )
            
            experiences = []
            for entry, score in results:
                try:
                    exp_data = entry.metadata.get('experience_data')
                    if exp_data:
                        if isinstance(exp_data, str):
                            exp_data = json.loads(exp_data)
                        exp = Experience.from_dict(exp_data)
                        
                        # Apply filters
                        if experience_types:
                            if exp.experience_type not in experience_types:
                                continue
                        
                        if location:
                            if exp.location:
                                if exp.location.get('biome') != location.get('biome'):
                                    continue
                            else:
                                continue
                        
                        # Record recall
                        exp.recall()
                        
                        experiences.append((exp, score))
                        
                        if len(experiences) >= n_results:
                            break
                            
                except Exception as e:
                    logger.debug(f"Error processing experience: {e}")
            
            return experiences
    
    def recall_by_context(
        self,
        task: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None,
        players: Optional[List[str]] = None,
        n_results: int = 5,
    ) -> List[Experience]:
        """Recall experiences by context."""
        # Build query from context
        query_parts = []
        
        if task:
            query_parts.append(f"task: {task}")
        if location:
            query_parts.append(f"location: {location.get('biome', '')}")
        if entities:
            query_parts.append(f"entities: {', '.join(entities)}")
        if players:
            query_parts.append(f"players: {', '.join(players)}")
        
        if not query_parts:
            return []
        
        query = " ".join(query_parts)
        results = self.recall(query, n_results=n_results)
        
        return [exp for exp, _ in results]
    
    def get_lessons_for_task(self, task: str, n_results: int = 5) -> List[str]:
        """Get lessons learned from similar tasks."""
        experiences = self.recall(
            query=f"task: {task}",
            n_results=n_results,
            experience_types=[ExperienceType.TASK_SUCCESS, ExperienceType.TASK_FAILURE],
        )
        
        lessons = []
        for exp, _ in experiences:
            lessons.extend(exp.lessons_learned)
        
        return list(set(lessons))  # Deduplicate
    
    def get_experience_summary(
        self,
        experience_type: Optional[ExperienceType] = None,
        time_period_hours: float = 24,
    ) -> Dict[str, Any]:
        """Get summary of recent experiences."""
        cutoff = time.time() - (time_period_hours * 3600)
        
        experiences = [
            exp for exp in self._experiences.values()
            if exp.timestamp > cutoff
            and (experience_type is None or exp.experience_type == experience_type)
        ]
        
        if not experiences:
            return {'count': 0}
        
        return {
            'count': len(experiences),
            'types': {
                t.value: sum(1 for e in experiences if e.experience_type == t)
                for t in ExperienceType
            },
            'avg_valence': sum(e.valence for e in experiences) / len(experiences),
            'avg_importance': sum(e.importance for e in experiences) / len(experiences),
            'unique_entities': list(set(
                entity for e in experiences for entity in e.entities_involved
            ))[:10],
            'unique_players': list(set(
                player for e in experiences for player in e.players_involved
            )),
            'lessons_learned': list(set(
                lesson for e in experiences for lesson in e.lessons_learned
            ))[:10],
        }
    
    def _cleanup_old_experiences(self) -> None:
        """Remove old, low-importance experiences."""
        # Sort by importance * strength
        sorted_exps = sorted(
            self._experiences.items(),
            key=lambda x: x[1].importance * x[1].strength
        )
        
        # Remove lowest scoring 10%
        to_remove = int(len(sorted_exps) * 0.1)
        for exp_id, exp in sorted_exps[:to_remove]:
            del self._experiences[exp_id]
            self.vector_store.delete(exp_id)
        
        logger.debug(f"Cleaned up {to_remove} old experiences")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_experiences': len(self._experiences),
            'by_type': {
                t.value: sum(1 for e in self._experiences.values() if e.experience_type == t)
                for t in ExperienceType
            },
        }
