"""
Enhanced Episodic Memory System for Voyager Evolved

Implements autobiographical memory inspired by cognitive neuroscience:
- Tulving's episodic memory theory
- Conway's self-memory system
- Emotion-enhanced memory consolidation

Features:
1. Autobiographical Memory - Agent's personal narrative and identity
2. Temporal Tagging - When events happened relative to others
3. Emotional Salience - Memories weighted by emotional impact
4. Episode Reconstruction - Piecing together partial memories
5. Memory Chains - Linked experiences forming narratives
"""

import time
import threading
import hashlib
import json
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set
from collections import defaultdict
import heapq


# ============================================================================
# EPISODIC MEMORY TYPES AND STRUCTURES
# ============================================================================

class EpisodeType(Enum):
    """Types of episodic memories."""
    # Specific Events
    ACHIEVEMENT = auto()      # Completed tasks, goals reached
    FAILURE = auto()          # Failed attempts, mistakes
    DISCOVERY = auto()        # Finding new things, learning
    SOCIAL = auto()           # Interactions with players
    DANGER = auto()           # Threats, close calls
    ROUTINE = auto()          # Regular activities
    
    # Self-defining Memories
    FIRST_TIME = auto()       # First experiences (first diamond, first death)
    TURNING_POINT = auto()    # Significant changes in behavior/strategy
    PEAK_EXPERIENCE = auto()  # Highly positive moments
    NADIR = auto()            # Lowest points
    
    # Procedural Episodes
    SKILL_LEARNED = auto()    # Learning new abilities
    STRATEGY_FORMED = auto()  # Developing new approaches


class EmotionalValence(Enum):
    """Emotional valence of memories."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class EmotionalTag:
    """Emotional coloring of a memory."""
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    arousal: float = 0.5  # 0-1, intensity of emotion
    primary_emotion: str = "neutral"  # e.g., "joy", "fear", "curiosity"
    secondary_emotions: List[str] = field(default_factory=list)
    
    @property
    def salience(self) -> float:
        """Calculate emotional salience (affects memory strength)."""
        # Higher arousal and extreme valence = more memorable
        valence_factor = abs(self.valence.value) / 2.0
        return (valence_factor + self.arousal) / 2.0


@dataclass
class TemporalContext:
    """Temporal information about an episode."""
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    game_day: int = 0
    time_of_day: str = "unknown"  # dawn, day, dusk, night
    sequence_position: int = 0  # Position in episode chain
    
    # Relative temporal markers
    before_episodes: Set[str] = field(default_factory=set)
    after_episodes: Set[str] = field(default_factory=set)
    concurrent_episodes: Set[str] = field(default_factory=set)
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600


@dataclass
class SpatialContext:
    """Spatial information about an episode."""
    location: Optional[Dict[str, float]] = None  # x, y, z
    biome: str = "unknown"
    dimension: str = "overworld"
    nearby_structures: List[str] = field(default_factory=list)
    landmarks: List[str] = field(default_factory=list)


@dataclass
class Episode:
    """
    A single episodic memory - a remembered experience.
    
    Inspired by Tulving's concept of episodic memory as
    "mental time travel" - re-experiencing past events.
    """
    id: str
    description: str
    episode_type: EpisodeType
    
    # Temporal and spatial context
    temporal: TemporalContext = field(default_factory=TemporalContext)
    spatial: SpatialContext = field(default_factory=SpatialContext)
    
    # Emotional coloring
    emotional: EmotionalTag = field(default_factory=EmotionalTag)
    
    # Memory characteristics
    vividness: float = 1.0  # How clear the memory is (decays over time)
    importance: float = 0.5  # Self-assessed importance
    rehearsal_count: int = 0  # Times recalled
    
    # Content
    details: Dict[str, Any] = field(default_factory=dict)
    involved_entities: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    
    # Connections
    causal_antecedents: List[str] = field(default_factory=list)  # What led to this
    causal_consequences: List[str] = field(default_factory=list)  # What this led to
    associated_skills: List[str] = field(default_factory=list)
    associated_concepts: List[str] = field(default_factory=list)
    
    # Autobiographical significance
    is_self_defining: bool = False
    narrative_themes: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Vector embedding for similarity search
    embedding: Optional[List[float]] = None
    
    @property
    def effective_strength(self) -> float:
        """Calculate effective memory strength."""
        # Base from emotional salience
        emotional_factor = 0.5 + self.emotional.salience * 0.5
        
        # Rehearsal strengthens memory
        rehearsal_factor = min(1.0, 0.3 + self.rehearsal_count * 0.1)
        
        # Importance and vividness
        importance_factor = self.importance
        vividness_factor = self.vividness
        
        # Self-defining memories are stronger
        self_defining_bonus = 0.3 if self.is_self_defining else 0.0
        
        return min(1.0, (
            emotional_factor * 0.3 +
            rehearsal_factor * 0.2 +
            importance_factor * 0.2 +
            vividness_factor * 0.2 +
            self_defining_bonus
        ) / 0.9)
    
    def to_narrative(self) -> str:
        """Convert episode to narrative text for LLM context."""
        parts = [f"I remember {self.description}."]
        
        if self.spatial.biome != "unknown":
            parts.append(f"It happened in the {self.spatial.biome}.")
        
        if self.outcome:
            parts.append(f"The result was: {self.outcome}.")
        
        if self.lessons_learned:
            parts.append(f"I learned: {', '.join(self.lessons_learned)}.")
        
        return " ".join(parts)


@dataclass
class EpisodeChain:
    """
    A chain of linked episodes forming a narrative.
    
    Represents extended experiences like "my first day",
    "learning to mine", "the battle with the creeper".
    """
    id: str
    name: str
    theme: str
    episodes: List[str]  # Episode IDs in chronological order
    created_at: float = field(default_factory=time.time)
    is_complete: bool = False
    summary: Optional[str] = None
    
    # Narrative properties
    arc_type: str = "neutral"  # "growth", "struggle", "discovery", etc.
    overall_emotional_arc: List[EmotionalValence] = field(default_factory=list)


# ============================================================================
# AUTOBIOGRAPHICAL MEMORY SYSTEM
# ============================================================================

class AutobiographicalMemory:
    """
    Self-memory system that maintains the agent's life narrative.
    
    Implements Conway's self-memory system:
    - Lifetime periods (major life phases)
    - General events (repeated/extended events)
    - Event-specific knowledge (unique episodes)
    """
    
    def __init__(self):
        # Lifetime periods
        self.life_periods: List[Dict[str, Any]] = []
        self.current_period: Optional[Dict[str, Any]] = None
        
        # Self-defining memories
        self.self_defining_memories: List[str] = []  # Episode IDs
        
        # Personal themes and goals
        self.life_themes: Dict[str, float] = {}  # theme -> strength
        self.long_term_goals: List[str] = []
        
        # Identity elements
        self.personality_insights: List[str] = []
        self.strengths: List[str] = []
        self.weaknesses: List[str] = []
        
        # Statistics
        self.total_experiences: int = 0
        self.achievements: int = 0
        self.failures: int = 0
        self.discoveries: int = 0
    
    def start_life_period(self, name: str, characteristics: List[str]):
        """Start a new life period (e.g., 'early survival', 'iron age')."""
        if self.current_period:
            self.current_period['end_time'] = time.time()
            self.life_periods.append(self.current_period)
        
        self.current_period = {
            'name': name,
            'start_time': time.time(),
            'characteristics': characteristics,
            'key_events': [],
            'emotional_tone': EmotionalValence.NEUTRAL.name
        }
    
    def add_self_defining_memory(self, episode_id: str):
        """Mark a memory as self-defining."""
        if episode_id not in self.self_defining_memories:
            self.self_defining_memories.append(episode_id)
    
    def update_theme(self, theme: str, delta: float):
        """Update strength of a life theme."""
        self.life_themes[theme] = self.life_themes.get(theme, 0.0) + delta
    
    def get_life_narrative(self) -> str:
        """Generate a narrative summary of the agent's life."""
        parts = []
        
        if self.life_periods:
            parts.append(f"I have lived through {len(self.life_periods)} distinct periods.")
        
        if self.current_period:
            parts.append(f"Currently, I am in my '{self.current_period['name']}' phase.")
        
        parts.append(f"I have had {self.achievements} achievements and {self.failures} setbacks.")
        
        if self.strengths:
            parts.append(f"My strengths are: {', '.join(self.strengths[:3])}.")
        
        if self.life_themes:
            top_themes = sorted(self.life_themes.items(), key=lambda x: x[1], reverse=True)[:3]
            themes_str = ", ".join([t[0] for t in top_themes])
            parts.append(f"My life themes include: {themes_str}.")
        
        return " ".join(parts)


# ============================================================================
# ENHANCED EPISODIC MEMORY - Main Class
# ============================================================================

class EnhancedEpisodicMemory:
    """
    Enhanced episodic memory with human-like characteristics.
    
    Features:
    - Autobiographical memory and life narrative
    - Emotional salience weighting
    - Episode reconstruction from fragments
    - Memory chains linking related experiences
    - Temporal ordering and context
    """
    
    def __init__(
        self,
        max_episodes: int = 5000,
        decay_rate: float = 0.01,
        emotional_boost: float = 1.5,
        persist_path: Optional[str] = None,
        embedding_fn: Optional[callable] = None
    ):
        """
        Initialize enhanced episodic memory.
        
        Args:
            max_episodes: Maximum episodes to retain
            decay_rate: Rate of vividness decay
            emotional_boost: Multiplier for emotional memory strength
            persist_path: Path to persist memory
            embedding_fn: Function to generate embeddings
        """
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate
        self.emotional_boost = emotional_boost
        self.persist_path = persist_path
        self.embedding_fn = embedding_fn
        
        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        self._episode_count = 0
        
        # Indices for efficient retrieval
        self._by_type: Dict[EpisodeType, List[str]] = defaultdict(list)
        self._by_location: Dict[str, List[str]] = defaultdict(list)
        self._by_entity: Dict[str, List[str]] = defaultdict(list)
        self._temporal_index: List[Tuple[float, str]] = []  # (timestamp, id)
        
        # Episode chains
        self.chains: Dict[str, EpisodeChain] = {}
        self._active_chain: Optional[str] = None
        
        # Autobiographical memory
        self.autobiography = AutobiographicalMemory()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        if self.persist_path and os.path.exists(self.persist_path):
            self._load()
    
    def _generate_id(self) -> str:
        """Generate unique episode ID."""
        self._episode_count += 1
        return f"ep_{self._episode_count}_{int(time.time()*1000)}"
    
    def record_episode(
        self,
        description: str,
        episode_type: EpisodeType,
        location: Optional[Dict[str, float]] = None,
        biome: str = "unknown",
        emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
        arousal: float = 0.5,
        primary_emotion: str = "neutral",
        details: Optional[Dict[str, Any]] = None,
        involved_entities: Optional[List[str]] = None,
        actions_taken: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        importance: float = 0.5,
        is_self_defining: bool = False,
        lessons: Optional[List[str]] = None
    ) -> str:
        """
        Record a new episodic memory.
        
        Returns:
            Episode ID
        """
        with self._lock:
            episode_id = self._generate_id()
            
            # Create episode
            episode = Episode(
                id=episode_id,
                description=description,
                episode_type=episode_type,
                temporal=TemporalContext(
                    timestamp=time.time(),
                    sequence_position=self._episode_count
                ),
                spatial=SpatialContext(
                    location=location,
                    biome=biome
                ),
                emotional=EmotionalTag(
                    valence=emotional_valence,
                    arousal=arousal,
                    primary_emotion=primary_emotion
                ),
                importance=importance,
                details=details or {},
                involved_entities=involved_entities or [],
                actions_taken=actions_taken or [],
                outcome=outcome,
                is_self_defining=is_self_defining,
                lessons_learned=lessons or []
            )
            
            # Generate embedding if function available
            if self.embedding_fn:
                try:
                    episode.embedding = self.embedding_fn(description)
                except Exception:
                    pass
            
            # Store episode
            self.episodes[episode_id] = episode
            
            # Update indices
            self._by_type[episode_type].append(episode_id)
            if biome != "unknown":
                self._by_location[biome].append(episode_id)
            for entity in (involved_entities or []):
                self._by_entity[entity].append(episode_id)
            heapq.heappush(self._temporal_index, (episode.temporal.timestamp, episode_id))
            
            # Update autobiographical memory
            self._update_autobiography(episode)
            
            # Add to active chain if present
            if self._active_chain and self._active_chain in self.chains:
                self.chains[self._active_chain].episodes.append(episode_id)
            
            # Link to previous episode
            self._link_to_previous(episode)
            
            # Cleanup if over capacity
            if len(self.episodes) > self.max_episodes:
                self._cleanup_old_episodes()
            
            return episode_id
    
    def _update_autobiography(self, episode: Episode):
        """Update autobiographical memory with new episode."""
        # Update statistics
        self.autobiography.total_experiences += 1
        
        if episode.episode_type == EpisodeType.ACHIEVEMENT:
            self.autobiography.achievements += 1
        elif episode.episode_type == EpisodeType.FAILURE:
            self.autobiography.failures += 1
        elif episode.episode_type == EpisodeType.DISCOVERY:
            self.autobiography.discoveries += 1
        
        # Mark self-defining memories
        if episode.is_self_defining:
            self.autobiography.add_self_defining_memory(episode.id)
        
        # Update themes from narrative themes
        for theme in episode.narrative_themes:
            self.autobiography.update_theme(theme, 0.1)
        
        # Add to current life period
        if self.autobiography.current_period:
            if episode.emotional.salience > 0.7 or episode.is_self_defining:
                self.autobiography.current_period['key_events'].append(episode.id)
    
    def _link_to_previous(self, episode: Episode):
        """Link new episode to previous episode temporally."""
        if len(self._temporal_index) > 1:
            # Get previous episode
            prev_timestamp, prev_id = self._temporal_index[-2]
            if prev_id in self.episodes:
                prev_episode = self.episodes[prev_id]
                
                # Temporal linking
                episode.temporal.after_episodes.add(prev_id)
                prev_episode.temporal.before_episodes.add(episode.id)
    
    def recall(
        self,
        query: Optional[str] = None,
        episode_type: Optional[EpisodeType] = None,
        entity: Optional[str] = None,
        location: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        emotional_valence: Optional[EmotionalValence] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Episode]:
        """
        Recall episodes matching criteria.
        
        Simulates memory retrieval with reconstruction.
        """
        with self._lock:
            candidates = []
            
            # Filter by type
            if episode_type:
                candidate_ids = set(self._by_type.get(episode_type, []))
            else:
                candidate_ids = set(self.episodes.keys())
            
            # Filter by entity
            if entity:
                entity_ids = set(self._by_entity.get(entity, []))
                candidate_ids &= entity_ids
            
            # Filter by location
            if location:
                location_ids = set(self._by_location.get(location, []))
                candidate_ids &= location_ids
            
            # Get episodes and apply remaining filters
            for ep_id in candidate_ids:
                if ep_id not in self.episodes:
                    continue
                    
                episode = self.episodes[ep_id]
                
                # Time range filter
                if time_range:
                    if not (time_range[0] <= episode.temporal.timestamp <= time_range[1]):
                        continue
                
                # Emotional filter
                if emotional_valence and episode.emotional.valence != emotional_valence:
                    continue
                
                # Importance filter
                if episode.importance < min_importance:
                    continue
                
                candidates.append(episode)
            
            # Score by relevance and strength
            def score_episode(ep: Episode) -> float:
                score = ep.effective_strength
                
                # Boost emotional episodes
                score *= (1.0 + ep.emotional.salience * (self.emotional_boost - 1))
                
                # Recency boost
                age_hours = ep.temporal.age_hours
                recency = 1.0 / (1.0 + age_hours / 24)
                score *= (0.5 + 0.5 * recency)
                
                return score
            
            candidates.sort(key=score_episode, reverse=True)
            
            # Mark as recalled (rehearsal)
            result = candidates[:limit]
            for ep in result:
                ep.rehearsal_count += 1
            
            return result
    
    def recall_chain(self, episode_id: str) -> List[Episode]:
        """
        Recall a chain of related episodes.
        
        Follows temporal and causal links to reconstruct
        a narrative sequence.
        """
        with self._lock:
            if episode_id not in self.episodes:
                return []
            
            chain = []
            visited = set()
            queue = [episode_id]
            
            while queue and len(chain) < 20:  # Limit chain length
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                
                if current_id not in self.episodes:
                    continue
                    
                episode = self.episodes[current_id]
                chain.append(episode)
                
                # Add connected episodes
                queue.extend(episode.causal_antecedents)
                queue.extend(episode.causal_consequences)
                queue.extend(episode.temporal.before_episodes)
                queue.extend(episode.temporal.after_episodes)
            
            # Sort by temporal order
            chain.sort(key=lambda e: e.temporal.timestamp)
            
            return chain
    
    def start_episode_chain(self, name: str, theme: str) -> str:
        """Start a new episode chain (extended narrative)."""
        with self._lock:
            chain_id = f"chain_{len(self.chains)}_{int(time.time())}"
            
            self.chains[chain_id] = EpisodeChain(
                id=chain_id,
                name=name,
                theme=theme,
                episodes=[]
            )
            
            self._active_chain = chain_id
            return chain_id
    
    def end_episode_chain(self, summary: Optional[str] = None) -> Optional[EpisodeChain]:
        """End the current episode chain."""
        with self._lock:
            if not self._active_chain or self._active_chain not in self.chains:
                return None
            
            chain = self.chains[self._active_chain]
            chain.is_complete = True
            chain.summary = summary
            
            # Determine emotional arc
            if chain.episodes:
                arc = []
                for ep_id in chain.episodes:
                    if ep_id in self.episodes:
                        arc.append(self.episodes[ep_id].emotional.valence)
                chain.overall_emotional_arc = arc
            
            self._active_chain = None
            return chain
    
    def get_similar_episodes(
        self,
        episode_id: str,
        limit: int = 5
    ) -> List[Tuple[Episode, float]]:
        """
        Find episodes similar to a given episode.
        
        Uses embeddings if available, otherwise content matching.
        """
        with self._lock:
            if episode_id not in self.episodes:
                return []
            
            target = self.episodes[episode_id]
            similarities = []
            
            for ep_id, episode in self.episodes.items():
                if ep_id == episode_id:
                    continue
                
                # Calculate similarity
                sim = self._calculate_similarity(target, episode)
                similarities.append((episode, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
    
    def _calculate_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """Calculate similarity between two episodes."""
        sim = 0.0
        
        # Type similarity
        if ep1.episode_type == ep2.episode_type:
            sim += 0.3
        
        # Location similarity
        if ep1.spatial.biome == ep2.spatial.biome:
            sim += 0.2
        
        # Entity overlap
        entities1 = set(ep1.involved_entities)
        entities2 = set(ep2.involved_entities)
        if entities1 and entities2:
            overlap = len(entities1 & entities2) / len(entities1 | entities2)
            sim += overlap * 0.2
        
        # Emotional similarity
        if ep1.emotional.valence == ep2.emotional.valence:
            sim += 0.15
        
        # Embedding similarity if available
        if ep1.embedding and ep2.embedding:
            try:
                # Cosine similarity
                import math
                dot = sum(a * b for a, b in zip(ep1.embedding, ep2.embedding))
                norm1 = math.sqrt(sum(a * a for a in ep1.embedding))
                norm2 = math.sqrt(sum(b * b for b in ep2.embedding))
                if norm1 > 0 and norm2 > 0:
                    embed_sim = dot / (norm1 * norm2)
                    sim = 0.5 * sim + 0.5 * embed_sim  # Blend
            except Exception:
                pass
        
        return min(1.0, sim)
    
    def reconstruct_episode(
        self,
        partial_info: Dict[str, Any]
    ) -> Optional[Episode]:
        """
        Reconstruct an episode from partial information.
        
        Simulates how humans piece together memories
        from fragments.
        """
        with self._lock:
            # Find matching episodes
            candidates = []
            
            for episode in self.episodes.values():
                match_score = 0.0
                
                # Match by description keywords
                if 'description' in partial_info:
                    desc_words = set(partial_info['description'].lower().split())
                    ep_words = set(episode.description.lower().split())
                    if desc_words & ep_words:
                        match_score += len(desc_words & ep_words) / len(desc_words)
                
                # Match by entities
                if 'entities' in partial_info:
                    partial_entities = set(partial_info['entities'])
                    ep_entities = set(episode.involved_entities)
                    if partial_entities & ep_entities:
                        match_score += 0.5
                
                # Match by location
                if 'biome' in partial_info and episode.spatial.biome == partial_info['biome']:
                    match_score += 0.3
                
                # Match by time
                if 'approximate_time' in partial_info:
                    time_diff = abs(episode.temporal.timestamp - partial_info['approximate_time'])
                    if time_diff < 3600:  # Within an hour
                        match_score += 0.4
                
                if match_score > 0.3:
                    candidates.append((episode, match_score))
            
            if not candidates:
                return None
            
            # Return best match
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    
    def _cleanup_old_episodes(self):
        """Remove old, low-importance episodes when over capacity."""
        # Score episodes for retention
        scored = []
        for ep_id, episode in self.episodes.items():
            # Don't remove self-defining memories
            if episode.is_self_defining:
                continue
            
            retention_score = episode.effective_strength
            scored.append((retention_score, ep_id))
        
        # Sort by retention score (lowest first)
        scored.sort(key=lambda x: x[0])
        
        # Remove lowest scoring until under capacity
        to_remove = len(self.episodes) - int(self.max_episodes * 0.9)
        for i in range(min(to_remove, len(scored))):
            ep_id = scored[i][1]
            self._remove_episode(ep_id)
    
    def _remove_episode(self, episode_id: str):
        """Remove an episode and clean up indices."""
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # Remove from type index
        if episode.episode_type in self._by_type:
            self._by_type[episode.episode_type] = [
                eid for eid in self._by_type[episode.episode_type]
                if eid != episode_id
            ]
        
        # Remove from location index
        if episode.spatial.biome in self._by_location:
            self._by_location[episode.spatial.biome] = [
                eid for eid in self._by_location[episode.spatial.biome]
                if eid != episode_id
            ]
        
        # Remove from entity index
        for entity in episode.involved_entities:
            if entity in self._by_entity:
                self._by_entity[entity] = [
                    eid for eid in self._by_entity[entity]
                    if eid != episode_id
                ]
        
        del self.episodes[episode_id]
    
    def apply_decay(self):
        """Apply time-based vividness decay to all episodes."""
        with self._lock:
            for episode in self.episodes.values():
                # Decay vividness
                age_hours = episode.temporal.age_hours
                decay_factor = (1 - self.decay_rate) ** (age_hours / 24)
                
                # Emotional memories decay slower
                if episode.emotional.salience > 0.5:
                    decay_factor = decay_factor ** 0.7
                
                # Rehearsed memories decay slower
                rehearsal_protection = min(0.5, episode.rehearsal_count * 0.1)
                decay_factor = decay_factor * (1 - rehearsal_protection) + rehearsal_protection
                
                episode.vividness = max(0.1, episode.vividness * decay_factor)
    
    def get_life_story(self, detailed: bool = False) -> str:
        """Get a narrative summary of the agent's life."""
        with self._lock:
            parts = [self.autobiography.get_life_narrative()]
            
            # Add recent highlights
            recent = self.recall(
                min_importance=0.6,
                limit=5
            )
            
            if recent:
                parts.append("\nRecent memorable experiences:")
                for ep in recent:
                    parts.append(f"- {ep.to_narrative()}")
            
            if detailed and self.autobiography.self_defining_memories:
                parts.append("\nSelf-defining memories:")
                for ep_id in self.autobiography.self_defining_memories[:5]:
                    if ep_id in self.episodes:
                        ep = self.episodes[ep_id]
                        parts.append(f"- {ep.to_narrative()}")
            
            return "\n".join(parts)
    
    def save(self):
        """Save memory to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            try:
                data = {
                    'episodes': {
                        ep_id: self._episode_to_dict(ep)
                        for ep_id, ep in self.episodes.items()
                    },
                    'chains': {
                        chain_id: {
                            'id': chain.id,
                            'name': chain.name,
                            'theme': chain.theme,
                            'episodes': chain.episodes,
                            'is_complete': chain.is_complete,
                            'summary': chain.summary
                        }
                        for chain_id, chain in self.chains.items()
                    },
                    'autobiography': {
                        'life_periods': self.autobiography.life_periods,
                        'current_period': self.autobiography.current_period,
                        'self_defining_memories': self.autobiography.self_defining_memories,
                        'life_themes': self.autobiography.life_themes,
                        'total_experiences': self.autobiography.total_experiences,
                        'achievements': self.autobiography.achievements,
                        'failures': self.autobiography.failures,
                        'discoveries': self.autobiography.discoveries
                    },
                    'episode_count': self._episode_count
                }
                
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save episodic memory: {e}")
    
    def _episode_to_dict(self, ep: Episode) -> Dict[str, Any]:
        """Convert episode to serializable dict."""
        return {
            'id': ep.id,
            'description': ep.description,
            'episode_type': ep.episode_type.name,
            'temporal': {
                'timestamp': ep.temporal.timestamp,
                'duration_seconds': ep.temporal.duration_seconds,
                'game_day': ep.temporal.game_day,
                'time_of_day': ep.temporal.time_of_day
            },
            'spatial': {
                'location': ep.spatial.location,
                'biome': ep.spatial.biome,
                'dimension': ep.spatial.dimension
            },
            'emotional': {
                'valence': ep.emotional.valence.name,
                'arousal': ep.emotional.arousal,
                'primary_emotion': ep.emotional.primary_emotion
            },
            'vividness': ep.vividness,
            'importance': ep.importance,
            'rehearsal_count': ep.rehearsal_count,
            'details': ep.details,
            'involved_entities': ep.involved_entities,
            'actions_taken': ep.actions_taken,
            'outcome': ep.outcome,
            'is_self_defining': ep.is_self_defining,
            'lessons_learned': ep.lessons_learned
        }
    
    def _load(self):
        """Load memory from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self._episode_count = data.get('episode_count', 0)
            
            # Load episodes
            for ep_id, ep_data in data.get('episodes', {}).items():
                episode = Episode(
                    id=ep_data['id'],
                    description=ep_data['description'],
                    episode_type=EpisodeType[ep_data['episode_type']],
                    temporal=TemporalContext(
                        timestamp=ep_data['temporal']['timestamp'],
                        duration_seconds=ep_data['temporal'].get('duration_seconds', 0),
                        game_day=ep_data['temporal'].get('game_day', 0),
                        time_of_day=ep_data['temporal'].get('time_of_day', 'unknown')
                    ),
                    spatial=SpatialContext(
                        location=ep_data['spatial'].get('location'),
                        biome=ep_data['spatial'].get('biome', 'unknown'),
                        dimension=ep_data['spatial'].get('dimension', 'overworld')
                    ),
                    emotional=EmotionalTag(
                        valence=EmotionalValence[ep_data['emotional']['valence']],
                        arousal=ep_data['emotional'].get('arousal', 0.5),
                        primary_emotion=ep_data['emotional'].get('primary_emotion', 'neutral')
                    ),
                    vividness=ep_data.get('vividness', 1.0),
                    importance=ep_data.get('importance', 0.5),
                    rehearsal_count=ep_data.get('rehearsal_count', 0),
                    details=ep_data.get('details', {}),
                    involved_entities=ep_data.get('involved_entities', []),
                    actions_taken=ep_data.get('actions_taken', []),
                    outcome=ep_data.get('outcome'),
                    is_self_defining=ep_data.get('is_self_defining', False),
                    lessons_learned=ep_data.get('lessons_learned', [])
                )
                self.episodes[ep_id] = episode
                
                # Rebuild indices
                self._by_type[episode.episode_type].append(ep_id)
                if episode.spatial.biome != 'unknown':
                    self._by_location[episode.spatial.biome].append(ep_id)
                for entity in episode.involved_entities:
                    self._by_entity[entity].append(ep_id)
            
            # Load autobiography
            auto_data = data.get('autobiography', {})
            self.autobiography.life_periods = auto_data.get('life_periods', [])
            self.autobiography.current_period = auto_data.get('current_period')
            self.autobiography.self_defining_memories = auto_data.get('self_defining_memories', [])
            self.autobiography.life_themes = auto_data.get('life_themes', {})
            self.autobiography.total_experiences = auto_data.get('total_experiences', 0)
            self.autobiography.achievements = auto_data.get('achievements', 0)
            self.autobiography.failures = auto_data.get('failures', 0)
            self.autobiography.discoveries = auto_data.get('discoveries', 0)
            
        except Exception as e:
            print(f"Failed to load episodic memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                'total_episodes': len(self.episodes),
                'max_capacity': self.max_episodes,
                'episode_chains': len(self.chains),
                'self_defining_memories': len(self.autobiography.self_defining_memories),
                'life_periods': len(self.autobiography.life_periods),
                'by_type': {
                    et.name: len(ids) for et, ids in self._by_type.items()
                },
                'autobiography': {
                    'achievements': self.autobiography.achievements,
                    'failures': self.autobiography.failures,
                    'discoveries': self.autobiography.discoveries
                }
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_episodic_instance: Optional[EnhancedEpisodicMemory] = None
_episodic_lock = threading.Lock()


def get_episodic_memory(
    persist_path: Optional[str] = None,
    **kwargs
) -> EnhancedEpisodicMemory:
    """Get or create the global enhanced episodic memory instance."""
    global _episodic_instance
    
    with _episodic_lock:
        if _episodic_instance is None:
            _episodic_instance = EnhancedEpisodicMemory(
                persist_path=persist_path or os.path.expanduser(
                    "~/.voyager_evolved/episodic_memory.json"
                ),
                **kwargs
            )
        return _episodic_instance


def reset_episodic_memory():
    """Reset the global episodic memory instance."""
    global _episodic_instance
    
    with _episodic_lock:
        if _episodic_instance:
            _episodic_instance.save()
        _episodic_instance = None
