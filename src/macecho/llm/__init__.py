"""
MacEcho LLM Module

This module provides a unified interface for different Large Language Model providers,
including local MLX models and cloud-based services like OpenAI.

Key Components:
- BaseLLM: Abstract base class for all LLM implementations
- MLXQwenChat: MLX-based local model implementation
- OpenAILLM: OpenAI API-based implementation
- LLMFactory: Factory for creating LLM instances from configuration
- ConversationContextManager: Manages conversation history and context
"""

from .base import BaseLLM, LLMProvider, LLMResponse, LLMStreamChunk
from .context_manager import ConversationContextManager, ConversationTurn
from .mlx_qwen import MLXQwenChat
from .factory import LLMFactory, create_llm_from_config

# Conditional imports for optional dependencies
try:
    from .openai_llm import OpenAILLM
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAILLM = None
    OPENAI_AVAILABLE = False

__all__ = [
    # Base classes and interfaces
    "BaseLLM",
    "LLMProvider", 
    "LLMResponse",
    "LLMStreamChunk",
    
    # Context management
    "ConversationContextManager",
    "ConversationTurn",
    
    # LLM implementations
    "MLXQwenChat",
    "OpenAILLM",  # May be None if not available
    
    # Factory and utilities
    "LLMFactory",
    "create_llm_from_config",
    
    # Availability flags
    "OPENAI_AVAILABLE",
]


def get_available_providers():
    """Get list of available LLM providers based on installed dependencies"""
    providers = ["mlx"]  # MLX is always available as it's a core dependency
    
    if OPENAI_AVAILABLE:
        providers.append("openai")
    
    return providers


def create_llm(provider: str = None, **kwargs):
    """
    Convenience function to create an LLM instance
    
    Args:
        provider: LLM provider name ("mlx", "openai", etc.)
        **kwargs: Configuration parameters
        
    Returns:
        BaseLLM instance
        
    Example:
        >>> llm = create_llm("mlx", model_name="mlx-community/Qwen3-4B-8bit")
        >>> response = llm.chat_with_context("Hello!")
    """
    if provider is None:
        provider = "mlx"  # Default to MLX
    
    config = {"provider": provider, **kwargs}
    return LLMFactory.create_llm(config)


# Version info
__version__ = "1.0.0"