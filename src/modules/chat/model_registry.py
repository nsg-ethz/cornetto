"""
Registry for creating chat model instances based on configuration.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from typing import Dict, Any, Tuple

from src.modules.chat.base import BaseChat
from src.modules.chat.models.chat_hf import _ChatHuggingFace
from src.modules.chat.models.chat_anthropic import _ChatAnthropic
from src.modules.chat.models.chat_ollama import _ChatOllama
from src.modules.chat.models.chat_google import _ChatGoogle
from src.modules.chat.models.chat_mistral import _ChatMistral
from src.modules.chat.models.chat_openai import _ChatOpenAI, _ChatXAI, _ChatDeepSeek, _ChatZAI

VALID_MODEL_REGISTRY = {
    "hf": _ChatHuggingFace,
    "openai": _ChatOpenAI,
    "ollama": _ChatOllama,
    "google": _ChatGoogle,
    "xai": _ChatXAI,
    "deepseek": _ChatDeepSeek,
    "mistral": _ChatMistral,
    "anthropic": _ChatAnthropic,
    "zai": _ChatZAI
} 


# =========================================================================== #
#                             Chat Model Registry                             #
# =========================================================================== #
def create_chat_model(
    model_provider: str, 
    **kwargs: Any
) -> BaseChat:
    """
    Registry function to create a chat model instance.
    
    Args:
        model_provider (str): Type of model to create.
        
    Returns:
        An instance of BaseChat.        
    """
    try:
        return VALID_MODEL_REGISTRY[model_provider.lower()](**kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown model type: {model_provider}. \
            Currently available model families: \
            'hf', 'openai', 'ollama', 'google', 'xai', 'deepseek', 'mistral', 'anthropic', 'zai'"
        )