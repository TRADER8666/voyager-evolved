"""
LLM Abstraction Layer for Voyager

This module provides a unified interface for different LLM providers,
making it easy to switch between OpenAI, Ollama, and other providers.

Default Configuration:
- Provider: Ollama (runs locally, no API key required)
- Model: llama2 (good general-purpose model)
- Endpoint: http://localhost:11434 (default Ollama endpoint)

To use OpenAI instead:
    from voyager.llm import get_llm, get_embeddings
    llm = get_llm(provider="openai", model_name="gpt-4")
    embeddings = get_embeddings(provider="openai")
"""

from .providers import (
    get_llm,
    get_embeddings,
    LLMProvider,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
)

__all__ = [
    "get_llm",
    "get_embeddings",
    "LLMProvider",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "OLLAMA_BASE_URL",
]
