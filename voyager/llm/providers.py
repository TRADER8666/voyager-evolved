"""
LLM Provider implementations for Voyager

Supports:
- Ollama (default, local LLM - no API key required)
- OpenAI (requires API key)

Ollama Setup:
1. Install Ollama: https://ollama.ai/download
2. Pull a model: ollama pull llama2  (or mistral, codellama, etc.)
3. Start Ollama: ollama serve (runs on http://localhost:11434)

Recommended Ollama Models for Minecraft Agent:
- llama2: Good general-purpose model (default)
- mistral: Fast and capable, good for code generation
- codellama: Specialized for code, great for action generation
- llama2:13b: Larger model, better reasoning (requires more RAM)
"""

import os
from enum import Enum
from typing import Optional, List, Any
import warnings

# Suppress deprecation warnings from langchain
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"


# ============================================================================
# DEFAULT CONFIGURATION - Ollama (local, no API key required)
# ============================================================================
DEFAULT_PROVIDER = LLMProvider.OLLAMA
DEFAULT_MODEL = "llama2"  # Good general-purpose model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Free, local embeddings with Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI model defaults (used when provider="openai")
OPENAI_DEFAULT_MODEL = "gpt-4"
OPENAI_DEFAULT_EMBEDDING = "text-embedding-ada-002"


def get_provider_from_env() -> LLMProvider:
    """Get LLM provider from environment variable or use default"""
    provider_str = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER.value).lower()
    if provider_str == "openai":
        return LLMProvider.OPENAI
    return LLMProvider.OLLAMA


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0,
    request_timeout: int = 120,
    **kwargs
):
    """
    Get an LLM instance based on the provider.
    
    Args:
        provider: "ollama" or "openai" (default: from env or "ollama")
        model_name: Model name (default: depends on provider)
        temperature: Sampling temperature (default: 0)
        request_timeout: Request timeout in seconds (default: 120)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        A LangChain-compatible LLM instance
    
    Examples:
        # Use Ollama (default, no API key required)
        llm = get_llm()
        
        # Use Ollama with specific model
        llm = get_llm(provider="ollama", model_name="mistral")
        
        # Use OpenAI (requires OPENAI_API_KEY env var)
        llm = get_llm(provider="openai", model_name="gpt-4")
    """
    # Determine provider
    if provider is None:
        provider = get_provider_from_env()
    elif isinstance(provider, str):
        provider = LLMProvider(provider.lower())
    
    if provider == LLMProvider.OLLAMA:
        return _get_ollama_llm(
            model_name=model_name or DEFAULT_MODEL,
            temperature=temperature,
            request_timeout=request_timeout,
            **kwargs
        )
    elif provider == LLMProvider.OPENAI:
        return _get_openai_llm(
            model_name=model_name or OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            request_timeout=request_timeout,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_embeddings(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
):
    """
    Get an embeddings instance based on the provider.
    
    Args:
        provider: "ollama" or "openai" (default: from env or "ollama")
        model_name: Model name (default: depends on provider)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        A LangChain-compatible embeddings instance
    
    Examples:
        # Use Ollama embeddings (default, no API key required)
        embeddings = get_embeddings()
        
        # Use OpenAI embeddings (requires OPENAI_API_KEY)
        embeddings = get_embeddings(provider="openai")
    """
    # Determine provider
    if provider is None:
        provider = get_provider_from_env()
    elif isinstance(provider, str):
        provider = LLMProvider(provider.lower())
    
    if provider == LLMProvider.OLLAMA:
        return _get_ollama_embeddings(
            model_name=model_name or DEFAULT_EMBEDDING_MODEL,
            **kwargs
        )
    elif provider == LLMProvider.OPENAI:
        return _get_openai_embeddings(
            model_name=model_name or OPENAI_DEFAULT_EMBEDDING,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ============================================================================
# OLLAMA IMPLEMENTATION
# ============================================================================

def _get_ollama_llm(
    model_name: str,
    temperature: float,
    request_timeout: int,
    base_url: Optional[str] = None,
    **kwargs
):
    """
    Create an Ollama LLM instance.
    
    Ollama runs locally and requires no API key.
    Make sure Ollama is installed and running: https://ollama.ai
    """
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        try:
            from langchain.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community is required for Ollama support. "
                "Install it with: pip install langchain-community"
            )
    
    base_url = base_url or OLLAMA_BASE_URL
    
    print(f"\033[36m[LLM] Using Ollama: model={model_name}, base_url={base_url}\033[0m")
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        request_timeout=request_timeout,
        **kwargs
    )


def _get_ollama_embeddings(
    model_name: str,
    base_url: Optional[str] = None,
    **kwargs
):
    """
    Create Ollama embeddings instance.
    
    Recommended models for embeddings:
    - nomic-embed-text: Good quality, fast (default)
    - mxbai-embed-large: Higher quality, slower
    
    Pull the model first: ollama pull nomic-embed-text
    """
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-community is required for Ollama support. "
                "Install it with: pip install langchain-community"
            )
    
    base_url = base_url or OLLAMA_BASE_URL
    
    print(f"\033[36m[Embeddings] Using Ollama: model={model_name}\033[0m")
    
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
        **kwargs
    )


# ============================================================================
# OPENAI IMPLEMENTATION
# ============================================================================

def _get_openai_llm(
    model_name: str,
    temperature: float,
    request_timeout: int,
    **kwargs
):
    """
    Create an OpenAI LLM instance.
    
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI provider. "
            "Either set the API key or use Ollama (default) which requires no API key:\n"
            "  export LLM_PROVIDER=ollama"
        )
    
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        try:
            from langchain.chat_models import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI support. "
                "Install it with: pip install langchain-openai"
            )
    
    print(f"\033[36m[LLM] Using OpenAI: model={model_name}\033[0m")
    
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
        **kwargs
    )


def _get_openai_embeddings(
    model_name: str,
    **kwargs
):
    """
    Create OpenAI embeddings instance.
    
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI embeddings. "
            "Either set the API key or use Ollama (default) which requires no API key."
        )
    
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain.embeddings.openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI support. "
                "Install it with: pip install langchain-openai"
            )
    
    print(f"\033[36m[Embeddings] Using OpenAI: model={model_name}\033[0m")
    
    return OpenAIEmbeddings(
        model=model_name,
        **kwargs
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_ollama_connection(base_url: Optional[str] = None) -> bool:
    """
    Check if Ollama server is running and accessible.
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    import urllib.request
    import urllib.error
    
    base_url = base_url or OLLAMA_BASE_URL
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def list_ollama_models(base_url: Optional[str] = None) -> List[str]:
    """
    List available models on the Ollama server.
    
    Returns:
        List of model names available locally
    """
    import urllib.request
    import json
    
    base_url = base_url or OLLAMA_BASE_URL
    try:
        response = urllib.request.urlopen(f"{base_url}/api/tags", timeout=10)
        data = json.loads(response.read().decode())
        return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def ensure_ollama_model(model_name: str, base_url: Optional[str] = None) -> bool:
    """
    Check if a model is available locally. If not, print instructions to pull it.
    
    Returns:
        True if model is available, False otherwise
    """
    available_models = list_ollama_models(base_url)
    
    # Check if exact match or version match (e.g., "llama2" matches "llama2:latest")
    model_base = model_name.split(":")[0]
    for available in available_models:
        if available == model_name or available.startswith(f"{model_base}:"):
            return True
    
    print(f"\033[33mWarning: Model '{model_name}' not found locally.\033[0m")
    print(f"\033[33mPull it with: ollama pull {model_name}\033[0m")
    print(f"\033[33mAvailable models: {', '.join(available_models) if available_models else 'None'}\033[0m")
    return False
