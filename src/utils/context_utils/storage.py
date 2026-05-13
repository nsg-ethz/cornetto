""" 
This script contains a general dictionary storage to map models to context window sizes. 
"""
from typing import Tuple


# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
model_ctxt_windows = {
    # OpenAI models
    "gpt-5.2-2025-12-11": 250_000,
    "gpt-5.1-2025-11-13": 250_000, # specific model knowledge version/cut-off
    "gpt-5.1": 250_000,
    "gpt-5": 250_000,
    "gpt-5-mini-2025-08-07": 250_000,
    "openai/gpt-oss-20b": 128_000,  # normally it is 128_000, we might reduce it to save from GPU memory use

    # xAI models
    "grok-4-1-fast-reasoning": 2_000_000,

    # Google models
    "gemini-3-pro-preview": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "google/gemma-3-12b-it": 128_000,
    "google/gemma-4-E4B-it": 128_000,

    # Anthropic models
    "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,

    # Qwen models
    "Qwen/Qwen3-4B-Thinking-2507": 250_000,
    "Qwen/Qwen3.5-9B": 260_000,

    # Z.AI models
    "glm-4.7": 200_000
}


# =========================================================================== #
#                           Context Window Retrieval                          #
# =========================================================================== #
def get_context_window(
    model: str,
    default: int = 200_000,
    safety_ratio: float = 0.6,
) -> Tuple[int, int]:
    """
    Get the effective context window size for a given model.

    Args:
        model (str): Model name or identifier.
        default (int): Default context window size if model not found.
            Defaults to 200_000.
        safety_ratio (float): Fraction of context window to use safely.
            Defaults to 0.6.

    Returns:
        Context window sizes (both limited and original) in tokens.
    """
    if model in model_ctxt_windows:
        return int(model_ctxt_windows[model] * safety_ratio), int(model_ctxt_windows[model])

    return int(default * safety_ratio), default
