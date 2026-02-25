"""
LLM Abstraction Layer for Voyager - Ollama Only

This module provides a simple interface for Ollama LLM integration.
Ollama runs locally and requires no API key!

Default Configuration:
- Model: llama2 (good general-purpose model)
- Endpoint: http://localhost:11434 (default Ollama endpoint)

Phase 1B Features:
- Multi-Model Ensemble: Automatic model selection based on task type
- Performance Tracking: Learn which models work best for each task
- Parallel Inference: Run multiple models simultaneously

Usage:
    from voyager.llm import get_llm, get_embeddings
    
    llm = get_llm()  # Uses default llama2
    llm = get_llm(model_name="mistral")  # Use specific model
    
    embeddings = get_embeddings()  # Uses nomic-embed-text
    
    # Phase 1B: Model Router
    from voyager.llm import get_model_router, TaskType
    router = get_model_router()
    llm = router.get_llm(TaskType.CODE_GENERATION)  # Auto-selects codellama
"""

from .providers import (
    get_llm,
    get_embeddings,
    DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    check_ollama_connection,
    list_ollama_models,
    ensure_ollama_model,
)

# Phase 1B: Multi-Model Ensemble
from .model_router import (
    ModelRouter,
    TaskType,
    ModelConfig,
    ModelPerformance,
    get_model_router,
    reset_router,
)

__all__ = [
    # Basic LLM functions
    "get_llm",
    "get_embeddings",
    "DEFAULT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "check_ollama_connection",
    "list_ollama_models",
    "ensure_ollama_model",
    
    # Phase 1B: Model Router
    "ModelRouter",
    "TaskType",
    "ModelConfig",
    "ModelPerformance",
    "get_model_router",
    "reset_router",
]
