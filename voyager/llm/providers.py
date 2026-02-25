"""
LLM Provider implementation for Voyager - Ollama Only

Ollama is the only supported LLM provider, offering:
- Free, local LLM - no API key required
- Privacy - all data stays on your machine
- No rate limits or usage costs

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
from typing import Optional, List
import warnings

# Suppress deprecation warnings from langchain
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


# ============================================================================
# DEFAULT CONFIGURATION - Ollama (local, no API key required)
# ============================================================================
DEFAULT_MODEL = "llama2"  # Good general-purpose model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Free, local embeddings with Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0,
    request_timeout: int = 120,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,  # Ignored, kept for backward compatibility
    **kwargs
):
    """
    Get an Ollama LLM instance.
    
    Args:
        model_name: Model name (default: llama2)
        temperature: Sampling temperature (default: 0)
        request_timeout: Request timeout in seconds (default: 120)
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional arguments passed to ChatOllama
    
    Returns:
        A LangChain ChatOllama instance
    
    Examples:
        # Use default model
        llm = get_llm()
        
        # Use specific model
        llm = get_llm(model_name="mistral")
        
        # With custom settings
        llm = get_llm(model_name="codellama", temperature=0.7)
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
    
    model = model_name or DEFAULT_MODEL
    url = base_url or OLLAMA_BASE_URL
    
    print(f"\033[36m[LLM] Using Ollama: model={model}, base_url={url}\033[0m")
    
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=url,
        request_timeout=request_timeout,
        **kwargs
    )


def get_embeddings(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,  # Ignored, kept for backward compatibility
    **kwargs
):
    """
    Get Ollama embeddings instance.
    
    Args:
        model_name: Model name (default: nomic-embed-text)
        base_url: Ollama server URL (default: http://localhost:11434)
        **kwargs: Additional arguments passed to OllamaEmbeddings
    
    Returns:
        A LangChain OllamaEmbeddings instance
    
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
    
    model = model_name or DEFAULT_EMBEDDING_MODEL
    url = base_url or OLLAMA_BASE_URL
    
    print(f"\033[36m[Embeddings] Using Ollama: model={model}\033[0m")
    
    return OllamaEmbeddings(
        model=model,
        base_url=url,
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
    
    url = base_url or OLLAMA_BASE_URL
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=5)
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
    
    url = base_url or OLLAMA_BASE_URL
    try:
        response = urllib.request.urlopen(f"{url}/api/tags", timeout=10)
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
