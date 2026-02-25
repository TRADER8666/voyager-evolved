"""
LLM Abstraction Layer for Voyager - Ollama Only

This module provides a simple interface for Ollama LLM integration.
Ollama runs locally and requires no API key!

Default Configuration:
- Model: llama2 (good general-purpose model)
- Endpoint: http://localhost:11434 (default Ollama endpoint)

Usage:
    from voyager.llm import get_llm, get_embeddings
    
    llm = get_llm()  # Uses default llama2
    llm = get_llm(model_name="mistral")  # Use specific model
    
    embeddings = get_embeddings()  # Uses nomic-embed-text
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

__all__ = [
    "get_llm",
    "get_embeddings",
    "DEFAULT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "check_ollama_connection",
    "list_ollama_models",
    "ensure_ollama_model",
]
