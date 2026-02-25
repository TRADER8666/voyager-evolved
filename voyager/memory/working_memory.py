"""
Working Memory System for Voyager Evolved

Implements a cognitively-realistic working memory system inspired by:
- Baddeley's model of working memory
- Miller's Law (7±2 item capacity)
- Cowan's embedded-processes model

Components:
1. Central Executive - Attention control and coordination
2. Phonological Loop - Verbal/textual information rehearsal
3. Visuospatial Sketchpad - Spatial and visual information
4. Episodic Buffer - Integration of multimodal information

Features:
- Limited capacity (5-9 items like humans)
- Attention-based item prioritization
- Automatic decay and rehearsal
- Transfer to long-term memory based on importance
- Context window management for LLM interactions
"""

import time
import threading
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple, Set
from collections import deque
import hashlib
import random


# ============================================================================
# WORKING MEMORY ITEM TYPES
# ============================================================================

class MemoryItemType(Enum):
    """Types of items that can be held in working memory."""
    VERBAL = auto()          # Text, instructions, conversations
    SPATIAL = auto()         # Locations, coordinates, spatial relations
    VISUAL = auto()          # Visual observations, object appearances
    PROCEDURAL = auto()      # Skills, actions, code snippets
    GOAL = auto()            # Current objectives, subgoals
    EPISODE = auto()         # Recent events, experiences
    FACT = auto()            # Semantic facts, knowledge
    PLAN = auto()            # Multi-step plans


@dataclass
class AttentionWeight:
    """Attention weight with decay characteristics."""
    value: float = 1.0
    last_attended: float = field(default_factory=time.time)
    attend_count: int = 1
    
    def decay(self, decay_rate: float = 0.1) -> float:
        """Apply time-based decay."""
        elapsed = time.time() - self.last_attended
        self.value *= (1 - decay_rate) ** elapsed
        return self.value
    
    def attend(self, boost: float = 0.3):
        """Boost attention from explicit focus."""
        self.value = min(1.0, self.value + boost)
        self.last_attended = time.time()
        self.attend_count += 1


@dataclass
class WorkingMemoryItem:
    """
    A single item in working memory.
    
    Inspired by chunks in cognitive architecture (ACT-R, SOAR).
    """
    id: str
    content: Any
    item_type: MemoryItemType
    attention: AttentionWeight = field(default_factory=AttentionWeight)
    created_at: float = field(default_factory=time.time)
    importance: float = 0.5  # 0-1, affects LTM transfer
    associations: Set[str] = field(default_factory=set)  # IDs of related items
    metadata: Dict[str, Any] = field(default_factory=dict)
    rehearsal_count: int = 0
    source: Optional[str] = None  # Where this item came from
    
    def __lt__(self, other):
        """For heap comparison - higher attention = higher priority."""
        return self.attention.value > other.attention.value
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def effective_priority(self) -> float:
        """Combined priority based on attention and importance."""
        return self.attention.value * 0.7 + self.importance * 0.3
    
    def to_text(self) -> str:
        """Convert to text representation for LLM context."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return str(self.content)
        else:
            return repr(self.content)


# ============================================================================
# WORKING MEMORY BUFFERS (Baddeley-inspired)
# ============================================================================

class PhonologicalLoop:
    """
    Handles verbal/textual information with rehearsal.
    
    Features:
    - Limited capacity for text
    - Subvocal rehearsal to maintain items
    - Decay without rehearsal
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.buffer: List[WorkingMemoryItem] = []
        self._rehearsal_pointer = 0
    
    def add(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """Add item, return displaced item if over capacity."""
        displaced = None
        if len(self.buffer) >= self.capacity:
            # Remove lowest attention item
            self.buffer.sort(key=lambda x: x.attention.value)
            displaced = self.buffer.pop(0)
        
        self.buffer.append(item)
        return displaced
    
    def rehearse(self) -> Optional[WorkingMemoryItem]:
        """Rehearse one item to prevent decay."""
        if not self.buffer:
            return None
        
        # Round-robin rehearsal
        self._rehearsal_pointer = self._rehearsal_pointer % len(self.buffer)
        item = self.buffer[self._rehearsal_pointer]
        item.attention.attend(boost=0.1)  # Small boost from rehearsal
        item.rehearsal_count += 1
        self._rehearsal_pointer += 1
        return item
    
    def get_items(self) -> List[WorkingMemoryItem]:
        return self.buffer.copy()


class VisuospatialSketchpad:
    """
    Handles spatial and visual information.
    
    Features:
    - Coordinate tracking
    - Visual scene representation
    - Mental rotation/manipulation placeholder
    """
    
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.buffer: List[WorkingMemoryItem] = []
        self.current_location: Optional[Dict[str, float]] = None
        self.landmark_memory: Dict[str, Dict[str, float]] = {}  # name -> coords
    
    def add(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """Add spatial/visual item."""
        displaced = None
        if len(self.buffer) >= self.capacity:
            self.buffer.sort(key=lambda x: x.attention.value)
            displaced = self.buffer.pop(0)
        
        self.buffer.append(item)
        
        # Extract location if present
        if item.item_type == MemoryItemType.SPATIAL:
            if isinstance(item.content, dict) and 'x' in item.content:
                self.current_location = item.content
        
        return displaced
    
    def update_location(self, x: float, y: float, z: float):
        """Update agent's current location."""
        self.current_location = {'x': x, 'y': y, 'z': z}
    
    def add_landmark(self, name: str, x: float, y: float, z: float):
        """Remember a landmark location."""
        self.landmark_memory[name] = {'x': x, 'y': y, 'z': z}
    
    def get_items(self) -> List[WorkingMemoryItem]:
        return self.buffer.copy()


class EpisodicBuffer:
    """
    Integration buffer that binds information from different sources.
    
    Features:
    - Multimodal integration
    - Temporal sequencing
    - Context binding
    """
    
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.current_episode: Optional[Dict[str, Any]] = None
    
    def add(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """Add integrated episode item."""
        displaced = None
        if len(self.buffer) >= self.capacity:
            displaced = self.buffer[0]
        
        self.buffer.append(item)
        return displaced
    
    def start_episode(self, context: Dict[str, Any]):
        """Start tracking a new episode."""
        self.current_episode = {
            'start_time': time.time(),
            'context': context,
            'events': []
        }
    
    def add_event(self, event: str):
        """Add event to current episode."""
        if self.current_episode:
            self.current_episode['events'].append({
                'time': time.time(),
                'event': event
            })
    
    def end_episode(self) -> Optional[Dict[str, Any]]:
        """End current episode and return it."""
        episode = self.current_episode
        self.current_episode = None
        return episode
    
    def get_items(self) -> List[WorkingMemoryItem]:
        return list(self.buffer)


# ============================================================================
# CENTRAL EXECUTIVE
# ============================================================================

class CentralExecutive:
    """
    Attention controller and coordinator for working memory.
    
    Responsibilities:
    - Attention allocation
    - Task switching
    - Inhibition of irrelevant information
    - Strategy selection
    """
    
    def __init__(self):
        self.current_focus: Optional[str] = None  # ID of focused item
        self.attention_history: deque = deque(maxlen=100)
        self.inhibited_items: Set[str] = set()
        self.task_stack: List[str] = []  # Stack of active tasks/goals
        
    def focus(self, item_id: str):
        """Direct attention to specific item."""
        self.current_focus = item_id
        self.attention_history.append((time.time(), item_id))
    
    def inhibit(self, item_id: str):
        """Inhibit an item (prevent it from capturing attention)."""
        self.inhibited_items.add(item_id)
    
    def release_inhibition(self, item_id: str):
        """Release inhibition on an item."""
        self.inhibited_items.discard(item_id)
    
    def push_task(self, task: str):
        """Push a task onto the task stack."""
        self.task_stack.append(task)
    
    def pop_task(self) -> Optional[str]:
        """Complete current task and return it."""
        if self.task_stack:
            return self.task_stack.pop()
        return None
    
    @property
    def current_task(self) -> Optional[str]:
        """Get current active task."""
        return self.task_stack[-1] if self.task_stack else None
    
    def is_inhibited(self, item_id: str) -> bool:
        return item_id in self.inhibited_items


# ============================================================================
# WORKING MEMORY SYSTEM - Main Class
# ============================================================================

class WorkingMemory:
    """
    Complete working memory system with human-like characteristics.
    
    Implements:
    - 7±2 item capacity (configurable 5-9)
    - Attention-based prioritization
    - Automatic decay with rehearsal
    - Transfer to long-term memory
    - Context window management for LLM
    """
    
    # Miller's magic number: 7±2
    DEFAULT_CAPACITY = 7
    MIN_CAPACITY = 5
    MAX_CAPACITY = 9
    
    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        decay_rate: float = 0.05,
        rehearsal_interval: float = 5.0,
        ltm_transfer_threshold: float = 0.7,
        ltm_callback: Optional[Callable[[WorkingMemoryItem], None]] = None,
        max_context_tokens: int = 4000
    ):
        """
        Initialize working memory.
        
        Args:
            capacity: Total WM capacity (5-9, default 7)
            decay_rate: Rate of attention decay
            rehearsal_interval: Seconds between rehearsals
            ltm_transfer_threshold: Importance threshold for LTM transfer
            ltm_callback: Callback when item is transferred to LTM
            max_context_tokens: Max tokens for LLM context window
        """
        # Enforce capacity bounds
        self.capacity = max(self.MIN_CAPACITY, min(self.MAX_CAPACITY, capacity))
        self.decay_rate = decay_rate
        self.rehearsal_interval = rehearsal_interval
        self.ltm_transfer_threshold = ltm_transfer_threshold
        self.ltm_callback = ltm_callback
        self.max_context_tokens = max_context_tokens
        
        # Initialize buffers (Baddeley model)
        phonological_capacity = max(2, self.capacity // 2)
        visuospatial_capacity = max(2, self.capacity // 3)
        episodic_capacity = max(2, self.capacity - phonological_capacity - visuospatial_capacity)
        
        self.phonological_loop = PhonologicalLoop(capacity=phonological_capacity)
        self.visuospatial_sketchpad = VisuospatialSketchpad(capacity=visuospatial_capacity)
        self.episodic_buffer = EpisodicBuffer(capacity=episodic_capacity)
        self.central_executive = CentralExecutive()
        
        # Unified item store for quick lookup
        self._items: Dict[str, WorkingMemoryItem] = {}
        self._item_count = 0
        
        # Background maintenance
        self._running = False
        self._maintenance_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'items_added': 0,
            'items_displaced': 0,
            'items_forgotten': 0,
            'items_transferred_ltm': 0,
            'rehearsals': 0,
            'attention_focuses': 0
        }
    
    def start(self):
        """Start background maintenance (decay & rehearsal)."""
        if self._running:
            return
        
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self._maintenance_thread.start()
    
    def stop(self):
        """Stop background maintenance."""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=2)
    
    def _maintenance_loop(self):
        """Background loop for decay and rehearsal."""
        while self._running:
            try:
                self._apply_decay()
                self._rehearse()
                self._cleanup_forgotten()
            except Exception as e:
                print(f"WM maintenance error: {e}")
            
            time.sleep(self.rehearsal_interval)
    
    def _generate_id(self) -> str:
        """Generate unique item ID."""
        self._item_count += 1
        return f"wm_{self._item_count}_{int(time.time()*1000)}"
    
    def add(
        self,
        content: Any,
        item_type: MemoryItemType,
        importance: float = 0.5,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add an item to working memory.
        
        Args:
            content: The content to remember
            item_type: Type of memory item
            importance: Importance for LTM transfer (0-1)
            source: Where this item came from
            metadata: Additional metadata
            
        Returns:
            ID of the added item
        """
        with self._lock:
            item_id = self._generate_id()
            
            item = WorkingMemoryItem(
                id=item_id,
                content=content,
                item_type=item_type,
                importance=min(1.0, max(0.0, importance)),
                source=source,
                metadata=metadata or {}
            )
            
            # Route to appropriate buffer
            displaced = self._route_to_buffer(item)
            
            # Store in unified lookup
            self._items[item_id] = item
            self.stats['items_added'] += 1
            
            # Handle displaced items
            if displaced:
                self._handle_displaced(displaced)
            
            return item_id
    
    def _route_to_buffer(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """Route item to appropriate buffer based on type."""
        if item.item_type in [MemoryItemType.VERBAL, MemoryItemType.FACT, MemoryItemType.PROCEDURAL]:
            return self.phonological_loop.add(item)
        elif item.item_type in [MemoryItemType.SPATIAL, MemoryItemType.VISUAL]:
            return self.visuospatial_sketchpad.add(item)
        else:  # EPISODE, GOAL, PLAN
            return self.episodic_buffer.add(item)
    
    def _handle_displaced(self, item: WorkingMemoryItem):
        """Handle an item displaced from a buffer."""
        self.stats['items_displaced'] += 1
        
        # Check if should transfer to LTM
        if self._should_transfer_to_ltm(item):
            self._transfer_to_ltm(item)
        
        # Remove from unified store
        if item.id in self._items:
            del self._items[item.id]
    
    def _should_transfer_to_ltm(self, item: WorkingMemoryItem) -> bool:
        """Determine if item should be transferred to LTM."""
        # Transfer if important enough
        if item.importance >= self.ltm_transfer_threshold:
            return True
        
        # Transfer if rehearsed multiple times (indicates importance)
        if item.rehearsal_count >= 3:
            return True
        
        # Transfer if attended multiple times
        if item.attention.attend_count >= 5:
            return True
        
        return False
    
    def _transfer_to_ltm(self, item: WorkingMemoryItem):
        """Transfer item to long-term memory."""
        self.stats['items_transferred_ltm'] += 1
        
        if self.ltm_callback:
            try:
                self.ltm_callback(item)
            except Exception as e:
                print(f"LTM transfer failed: {e}")
    
    def get(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Get an item by ID, boosting its attention."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.attention.attend()
                self.stats['attention_focuses'] += 1
            return item
    
    def attend(self, item_id: str) -> bool:
        """Focus attention on an item."""
        with self._lock:
            if item_id in self._items:
                self._items[item_id].attention.attend()
                self.central_executive.focus(item_id)
                self.stats['attention_focuses'] += 1
                return True
            return False
    
    def forget(self, item_id: str) -> bool:
        """Explicitly forget an item."""
        with self._lock:
            if item_id in self._items:
                item = self._items[item_id]
                
                # Remove from buffers
                self.phonological_loop.buffer = [
                    i for i in self.phonological_loop.buffer if i.id != item_id
                ]
                self.visuospatial_sketchpad.buffer = [
                    i for i in self.visuospatial_sketchpad.buffer if i.id != item_id
                ]
                self.episodic_buffer.buffer = deque(
                    [i for i in self.episodic_buffer.buffer if i.id != item_id],
                    maxlen=self.episodic_buffer.capacity
                )
                
                del self._items[item_id]
                self.stats['items_forgotten'] += 1
                return True
            return False
    
    def _apply_decay(self):
        """Apply time-based decay to all items."""
        with self._lock:
            for item in self._items.values():
                if not self.central_executive.is_inhibited(item.id):
                    item.attention.decay(self.decay_rate)
    
    def _rehearse(self):
        """Perform rehearsal to maintain items."""
        with self._lock:
            # Rehearse in phonological loop
            item = self.phonological_loop.rehearse()
            if item:
                self.stats['rehearsals'] += 1
    
    def _cleanup_forgotten(self):
        """Remove items that have decayed below threshold."""
        with self._lock:
            forgotten_threshold = 0.1
            to_remove = []
            
            for item_id, item in self._items.items():
                if item.attention.value < forgotten_threshold:
                    # Check for LTM transfer before removing
                    if self._should_transfer_to_ltm(item):
                        self._transfer_to_ltm(item)
                    to_remove.append(item_id)
            
            for item_id in to_remove:
                self.forget(item_id)
    
    def get_all_items(self) -> List[WorkingMemoryItem]:
        """Get all items in working memory."""
        with self._lock:
            return list(self._items.values())
    
    def get_by_type(self, item_type: MemoryItemType) -> List[WorkingMemoryItem]:
        """Get items of a specific type."""
        with self._lock:
            return [i for i in self._items.values() if i.item_type == item_type]
    
    def get_top_items(self, n: int = 5) -> List[WorkingMemoryItem]:
        """Get the n items with highest attention/priority."""
        with self._lock:
            items = list(self._items.values())
            items.sort(key=lambda x: x.effective_priority, reverse=True)
            return items[:n]
    
    def get_context_for_llm(self, max_tokens: Optional[int] = None) -> str:
        """
        Generate context string for LLM from working memory.
        
        Prioritizes high-attention items and includes metadata.
        """
        max_tokens = max_tokens or self.max_context_tokens
        
        with self._lock:
            # Get items sorted by priority
            items = self.get_top_items(self.capacity)
            
            context_parts = []
            estimated_tokens = 0
            
            for item in items:
                text = item.to_text()
                # Rough token estimation (4 chars per token)
                item_tokens = len(text) // 4 + 10  # +10 for formatting
                
                if estimated_tokens + item_tokens > max_tokens:
                    break
                
                # Format based on type
                type_label = item.item_type.name.lower()
                context_parts.append(f"[{type_label}] {text}")
                estimated_tokens += item_tokens
            
            return "\n".join(context_parts)
    
    def update_location(self, x: float, y: float, z: float):
        """Update spatial reference point."""
        self.visuospatial_sketchpad.update_location(x, y, z)
    
    def push_goal(self, goal: str) -> str:
        """Add a goal to working memory and task stack."""
        self.central_executive.push_task(goal)
        return self.add(
            content=goal,
            item_type=MemoryItemType.GOAL,
            importance=0.9
        )
    
    def pop_goal(self) -> Optional[str]:
        """Complete current goal."""
        return self.central_executive.pop_task()
    
    @property
    def current_goal(self) -> Optional[str]:
        """Get current active goal."""
        return self.central_executive.current_task
    
    def clear(self):
        """Clear all working memory."""
        with self._lock:
            self._items.clear()
            self.phonological_loop.buffer.clear()
            self.visuospatial_sketchpad.buffer.clear()
            self.episodic_buffer.buffer.clear()
            self.central_executive.task_stack.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self._lock:
            return {
                **self.stats,
                'current_items': len(self._items),
                'capacity': self.capacity,
                'utilization': f"{len(self._items) / self.capacity * 100:.1f}%",
                'phonological_items': len(self.phonological_loop.buffer),
                'visuospatial_items': len(self.visuospatial_sketchpad.buffer),
                'episodic_items': len(self.episodic_buffer.buffer),
                'active_goals': len(self.central_executive.task_stack)
            }
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_wm_instance: Optional[WorkingMemory] = None
_wm_lock = threading.Lock()


def get_working_memory(**kwargs) -> WorkingMemory:
    """Get or create the global working memory instance."""
    global _wm_instance
    
    with _wm_lock:
        if _wm_instance is None:
            _wm_instance = WorkingMemory(**kwargs)
            _wm_instance.start()
        return _wm_instance


def reset_working_memory():
    """Reset the global working memory instance."""
    global _wm_instance
    
    with _wm_lock:
        if _wm_instance:
            _wm_instance.stop()
        _wm_instance = None
