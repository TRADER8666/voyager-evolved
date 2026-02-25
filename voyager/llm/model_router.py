"""
Multi-Model Ensemble System for Voyager Evolved

Implements intelligent model routing that selects the best Ollama model
for each task type, inspired by how humans use different cognitive modes:

- Reasoning tasks: llama2/mistral (System 2 thinking - slow, deliberate)
- Code generation: codellama (specialized procedural knowledge)
- Conversation: neural-chat (social cognition)
- Fast decisions: tinyllama (System 1 thinking - fast, intuitive)

Features:
- Task-based model selection
- Performance tracking per model
- Parallel inference for consensus/best-of-N
- Automatic fallback on model unavailability
- Context-aware routing
"""

import os
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

# ============================================================================
# TASK TYPES - Cognitive Modes
# ============================================================================

class TaskType(Enum):
    """Task types mapped to cognitive processing modes."""
    
    # System 2 - Deliberate Reasoning
    REASONING = "reasoning"          # Complex logic, planning
    ANALYSIS = "analysis"            # Understanding situations
    PLANNING = "planning"            # Multi-step planning
    
    # Procedural Knowledge
    CODE_GENERATION = "code"         # Writing Mineflayer code
    CODE_REVIEW = "code_review"      # Checking code quality
    
    # Social Cognition
    CONVERSATION = "conversation"    # Natural dialogue
    INSTRUCTION = "instruction"      # Following/giving instructions
    
    # System 1 - Fast Intuition
    CLASSIFICATION = "classification"  # Quick categorization
    DECISION = "decision"             # Binary/simple choices
    EXTRACTION = "extraction"         # Pulling out specific info
    
    # Creative
    CREATIVE = "creative"            # Novel solutions
    
    # Default
    GENERAL = "general"              # Fallback for unknown tasks


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    task_types: List[TaskType]
    priority: int = 1  # Lower is higher priority
    temperature_override: Optional[float] = None
    max_tokens: Optional[int] = None
    warmup_required: bool = False
    estimated_speed: float = 1.0  # Relative speed (1.0 = baseline)
    
    
@dataclass
class ModelPerformance:
    """Tracks performance metrics for a model."""
    model_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    task_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    last_used: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls
    
    def record_call(self, task_type: str, success: bool, latency_ms: float, 
                   tokens: int = 0, quality_score: float = 1.0):
        """Record a model call outcome."""
        self.total_calls += 1
        self.last_used = time.time()
        
        if success:
            self.successful_calls += 1
            self.total_latency_ms += latency_ms
            self.total_tokens += tokens
            
            # Track per-task performance
            if task_type not in self.task_performance:
                self.task_performance[task_type] = {
                    "calls": 0, "success": 0, "latency": 0, "quality": 0
                }
            
            self.task_performance[task_type]["calls"] += 1
            self.task_performance[task_type]["success"] += 1
            self.task_performance[task_type]["latency"] += latency_ms
            self.task_performance[task_type]["quality"] += quality_score
        else:
            self.failed_calls += 1
            if task_type not in self.task_performance:
                self.task_performance[task_type] = {
                    "calls": 0, "success": 0, "latency": 0, "quality": 0
                }
            self.task_performance[task_type]["calls"] += 1
    
    def get_task_score(self, task_type: str) -> float:
        """Get composite score for a task type (higher is better)."""
        if task_type not in self.task_performance:
            return 0.5  # Neutral score for unknown
        
        perf = self.task_performance[task_type]
        if perf["calls"] == 0:
            return 0.5
        
        success_rate = perf["success"] / perf["calls"]
        avg_quality = perf["quality"] / max(perf["success"], 1)
        
        # Composite: 60% quality, 40% success rate
        return 0.6 * avg_quality + 0.4 * success_rate


# ============================================================================
# DEFAULT MODEL CONFIGURATIONS
# ============================================================================

DEFAULT_MODEL_CONFIGS = [
    # Primary reasoning model
    ModelConfig(
        name="llama2",
        task_types=[TaskType.REASONING, TaskType.ANALYSIS, TaskType.PLANNING, TaskType.GENERAL],
        priority=1,
        estimated_speed=1.0
    ),
    
    # Alternative reasoning (often faster)
    ModelConfig(
        name="mistral",
        task_types=[TaskType.REASONING, TaskType.ANALYSIS, TaskType.INSTRUCTION],
        priority=2,
        estimated_speed=1.2
    ),
    
    # Code specialist
    ModelConfig(
        name="codellama",
        task_types=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW],
        priority=1,
        temperature_override=0.1,  # Lower temp for code
        estimated_speed=1.0
    ),
    
    # Conversational model
    ModelConfig(
        name="neural-chat",
        task_types=[TaskType.CONVERSATION, TaskType.INSTRUCTION],
        priority=1,
        temperature_override=0.7,
        estimated_speed=1.1
    ),
    
    # Fast model for quick decisions
    ModelConfig(
        name="tinyllama",
        task_types=[TaskType.CLASSIFICATION, TaskType.DECISION, TaskType.EXTRACTION],
        priority=1,
        estimated_speed=3.0  # Much faster
    ),
    
    # Creative tasks
    ModelConfig(
        name="llama2",
        task_types=[TaskType.CREATIVE],
        priority=1,
        temperature_override=0.8
    ),
]


# ============================================================================
# MODEL ROUTER - Main Class
# ============================================================================

class ModelRouter:
    """
    Intelligent model router that selects the best Ollama model for each task.
    
    Mimics human cognitive flexibility:
    - Uses fast intuitive processing for simple decisions
    - Engages deliberate reasoning for complex problems
    - Specializes for code vs. conversation
    """
    
    def __init__(
        self,
        model_configs: Optional[List[ModelConfig]] = None,
        ollama_base_url: Optional[str] = None,
        max_parallel_workers: int = 4,
        enable_performance_tracking: bool = True,
        enable_fallback: bool = True,
        cache_size: int = 100,
        persist_path: Optional[str] = None
    ):
        """
        Initialize the model router.
        
        Args:
            model_configs: List of model configurations
            ollama_base_url: Ollama server URL
            max_parallel_workers: Max workers for parallel inference
            enable_performance_tracking: Track model performance
            enable_fallback: Enable automatic fallback
            cache_size: LRU cache size for responses
            persist_path: Path to persist performance data
        """
        self.ollama_base_url = ollama_base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model_configs = model_configs or DEFAULT_MODEL_CONFIGS
        self.max_parallel_workers = max_parallel_workers
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_fallback = enable_fallback
        self.persist_path = persist_path
        
        # Build task -> models mapping
        self.task_models: Dict[TaskType, List[ModelConfig]] = {}
        for config in self.model_configs:
            for task_type in config.task_types:
                if task_type not in self.task_models:
                    self.task_models[task_type] = []
                self.task_models[task_type].append(config)
        
        # Sort by priority
        for task_type in self.task_models:
            self.task_models[task_type].sort(key=lambda c: c.priority)
        
        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self._available_models: Optional[List[str]] = None
        self._availability_checked_at: float = 0
        
        # Response cache
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()
        
        # Thread pool for parallel inference
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_workers)
        
        # Load persisted data
        if self.persist_path:
            self._load_performance_data()
    
    def _get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of models available on Ollama server."""
        # Cache for 60 seconds
        if (not force_refresh and 
            self._available_models is not None and 
            time.time() - self._availability_checked_at < 60):
            return self._available_models
        
        from .providers import list_ollama_models
        self._available_models = list_ollama_models(self.ollama_base_url)
        self._availability_checked_at = time.time()
        return self._available_models
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available = self._get_available_models()
        model_base = model_name.split(":")[0]
        for avail in available:
            if avail == model_name or avail.startswith(f"{model_base}:"):
                return True
        return False
    
    def select_model(
        self,
        task_type: TaskType,
        context: Optional[Dict[str, Any]] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False
    ) -> Tuple[str, ModelConfig]:
        """
        Select the best model for a task.
        
        Args:
            task_type: Type of task to perform
            context: Additional context for selection
            prefer_speed: Prefer faster models
            prefer_quality: Prefer higher quality models
            
        Returns:
            Tuple of (model_name, model_config)
        """
        candidates = self.task_models.get(task_type, [])
        
        # Fallback to general if no specific models
        if not candidates:
            candidates = self.task_models.get(TaskType.GENERAL, [])
        
        if not candidates:
            # Ultimate fallback
            return ("llama2", ModelConfig(name="llama2", task_types=[TaskType.GENERAL]))
        
        # Filter by availability
        available_candidates = []
        for config in candidates:
            if self._is_model_available(config.name):
                available_candidates.append(config)
        
        # If none available, use fallback if enabled
        if not available_candidates:
            if self.enable_fallback:
                available = self._get_available_models()
                if available:
                    fallback_name = available[0]
                    return (fallback_name, ModelConfig(
                        name=fallback_name, 
                        task_types=[task_type]
                    ))
            # Return first candidate anyway (will error at call time)
            return (candidates[0].name, candidates[0])
        
        # Score candidates
        scored = []
        for config in available_candidates:
            score = 0.0
            
            # Base priority score (lower priority = higher score)
            score += (10 - config.priority) * 10
            
            # Performance-based scoring
            if self.enable_performance_tracking and config.name in self.model_performance:
                perf = self.model_performance[config.name]
                task_score = perf.get_task_score(task_type.value)
                score += task_score * 30  # Up to 30 points for quality
                
                # Speed factor
                if prefer_speed:
                    score += config.estimated_speed * 10
                    score -= perf.avg_latency_ms / 1000  # Penalize slow
            
            # Quality preference
            if prefer_quality and config.name in ["llama2:13b", "mistral"]:
                score += 20
            
            scored.append((score, config))
        
        # Select highest scoring
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[0][1]
        
        return (selected.name, selected)
    
    def get_llm(
        self,
        task_type: TaskType = TaskType.GENERAL,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Get an LLM instance optimized for the task type.
        
        Args:
            task_type: Type of task
            model_name: Override model selection
            temperature: Override temperature
            **kwargs: Additional args for ChatOllama
            
        Returns:
            ChatOllama instance
        """
        from .providers import get_llm as base_get_llm
        
        if model_name:
            selected_name = model_name
            config = None
        else:
            selected_name, config = self.select_model(task_type)
        
        # Determine temperature
        if temperature is not None:
            final_temp = temperature
        elif config and config.temperature_override is not None:
            final_temp = config.temperature_override
        else:
            final_temp = 0.0
        
        return base_get_llm(
            model_name=selected_name,
            temperature=final_temp,
            base_url=self.ollama_base_url,
            **kwargs
        )
    
    def infer(
        self,
        prompt: str,
        task_type: TaskType = TaskType.GENERAL,
        model_name: Optional[str] = None,
        use_cache: bool = True,
        track_performance: bool = True,
        **kwargs
    ) -> str:
        """
        Perform inference with automatic model selection.
        
        Args:
            prompt: Input prompt
            task_type: Type of task
            model_name: Override model
            use_cache: Use response cache
            track_performance: Track this call's performance
            **kwargs: Additional args
            
        Returns:
            Model response string
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._make_cache_key(prompt, task_type, model_name)
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Select and get model
        llm = self.get_llm(task_type, model_name, **kwargs)
        selected_name = model_name or self.select_model(task_type)[0]
        
        # Perform inference with tracking
        start_time = time.time()
        success = False
        response = ""
        
        try:
            result = llm.invoke(prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            success = True
        except Exception as e:
            response = f"Error: {str(e)}"
            success = False
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Track performance
        if track_performance and self.enable_performance_tracking:
            self._track_call(selected_name, task_type.value, success, latency_ms)
        
        # Cache response
        if use_cache and cache_key and success:
            self._cache_response(cache_key, response)
        
        return response
    
    def parallel_infer(
        self,
        prompt: str,
        task_type: TaskType = TaskType.GENERAL,
        models: Optional[List[str]] = None,
        strategy: str = "first",  # "first", "best", "consensus"
        timeout: float = 60.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Perform parallel inference across multiple models.
        
        Args:
            prompt: Input prompt
            task_type: Task type for model selection
            models: Specific models to use (or auto-select)
            strategy: How to combine results
                - "first": Return first successful response
                - "best": Return best response (requires scoring)
                - "consensus": Return most common response
            timeout: Timeout for all models
            
        Returns:
            Tuple of (response, metadata)
        """
        # Determine models to use
        if models is None:
            candidates = self.task_models.get(task_type, [])
            models = [c.name for c in candidates[:3] if self._is_model_available(c.name)]
            if not models:
                models = ["llama2"]  # Fallback
        
        results = {}
        futures = {}
        
        # Submit parallel tasks
        for model in models:
            future = self._executor.submit(
                self._single_infer,
                prompt, model, task_type
            )
            futures[future] = model
        
        # Collect results based on strategy
        if strategy == "first":
            for future in as_completed(futures, timeout=timeout):
                model = futures[future]
                try:
                    response, latency = future.result()
                    if response and not response.startswith("Error:"):
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return response, {"model": model, "latency_ms": latency}
                except Exception:
                    continue
        else:
            # Wait for all
            for future in as_completed(futures, timeout=timeout):
                model = futures[future]
                try:
                    response, latency = future.result()
                    results[model] = {"response": response, "latency": latency}
                except Exception as e:
                    results[model] = {"response": None, "error": str(e)}
        
        if strategy == "consensus":
            # Find most common response (simplified)
            responses = [r["response"] for r in results.values() if r.get("response")]
            if responses:
                # Simple majority
                from collections import Counter
                common = Counter(responses).most_common(1)
                if common:
                    return common[0][0], {"strategy": "consensus", "models": list(results.keys())}
        
        elif strategy == "best":
            # Return longest response (heuristic for quality)
            best_response = ""
            best_model = None
            for model, data in results.items():
                resp = data.get("response", "")
                if resp and len(resp) > len(best_response):
                    best_response = resp
                    best_model = model
            if best_response:
                return best_response, {"model": best_model, "strategy": "best"}
        
        # Fallback: return any successful response
        for model, data in results.items():
            if data.get("response"):
                return data["response"], {"model": model}
        
        return "Error: All models failed", {"error": True}
    
    def _single_infer(
        self, 
        prompt: str, 
        model_name: str, 
        task_type: TaskType
    ) -> Tuple[str, float]:
        """Single model inference for parallel execution."""
        start = time.time()
        try:
            llm = self.get_llm(task_type, model_name=model_name)
            result = llm.invoke(prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            latency = (time.time() - start) * 1000
            self._track_call(model_name, task_type.value, True, latency)
            return response, latency
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track_call(model_name, task_type.value, False, latency)
            return f"Error: {str(e)}", latency
    
    def _track_call(
        self, 
        model_name: str, 
        task_type: str, 
        success: bool, 
        latency_ms: float,
        quality_score: float = 1.0
    ):
        """Track a model call for performance analysis."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance(model_name=model_name)
        
        self.model_performance[model_name].record_call(
            task_type, success, latency_ms, quality_score=quality_score
        )
        
        # Periodically persist
        if self.persist_path and self.model_performance[model_name].total_calls % 10 == 0:
            self._save_performance_data()
    
    def _make_cache_key(
        self, 
        prompt: str, 
        task_type: TaskType, 
        model: Optional[str]
    ) -> str:
        """Create cache key for a request."""
        key_data = f"{prompt}:{task_type.value}:{model or 'auto'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[str]:
        """Get cached response."""
        with self._cache_lock:
            return self._cache.get(key)
    
    def _cache_response(self, key: str, response: str):
        """Cache a response with LRU eviction."""
        with self._cache_lock:
            if key in self._cache:
                self._cache_order.remove(key)
            elif len(self._cache) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            
            self._cache[key] = response
            self._cache_order.append(key)
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        if not self.persist_path:
            return
        
        try:
            data = {}
            for name, perf in self.model_performance.items():
                data[name] = {
                    "total_calls": perf.total_calls,
                    "successful_calls": perf.successful_calls,
                    "failed_calls": perf.failed_calls,
                    "total_latency_ms": perf.total_latency_ms,
                    "task_performance": perf.task_performance
                }
            
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save performance data: {e}")
    
    def _load_performance_data(self):
        """Load persisted performance data."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            for name, perf_data in data.items():
                self.model_performance[name] = ModelPerformance(
                    model_name=name,
                    total_calls=perf_data.get("total_calls", 0),
                    successful_calls=perf_data.get("successful_calls", 0),
                    failed_calls=perf_data.get("failed_calls", 0),
                    total_latency_ms=perf_data.get("total_latency_ms", 0),
                    task_performance=perf_data.get("task_performance", {})
                )
        except Exception as e:
            print(f"Warning: Failed to load performance data: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "available_models": self._get_available_models(),
            "models": {}
        }
        
        for name, perf in self.model_performance.items():
            report["models"][name] = {
                "total_calls": perf.total_calls,
                "success_rate": f"{perf.success_rate:.2%}",
                "avg_latency_ms": f"{perf.avg_latency_ms:.1f}",
                "task_performance": {
                    task: {
                        "score": f"{perf.get_task_score(task):.2f}"
                    }
                    for task in perf.task_performance
                }
            }
        
        return report
    
    def clear_cache(self):
        """Clear the response cache."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()
    
    def shutdown(self):
        """Shutdown the router and save state."""
        self._save_performance_data()
        self._executor.shutdown(wait=False)


# ============================================================================
# GLOBAL ROUTER INSTANCE
# ============================================================================

_router_instance: Optional[ModelRouter] = None
_router_lock = threading.Lock()


def get_model_router(
    persist_path: Optional[str] = None,
    **kwargs
) -> ModelRouter:
    """Get or create the global model router instance."""
    global _router_instance
    
    with _router_lock:
        if _router_instance is None:
            _router_instance = ModelRouter(
                persist_path=persist_path or os.path.expanduser(
                    "~/.voyager_evolved/model_performance.json"
                ),
                **kwargs
            )
        return _router_instance


def reset_router():
    """Reset the global router instance."""
    global _router_instance
    with _router_lock:
        if _router_instance:
            _router_instance.shutdown()
        _router_instance = None
