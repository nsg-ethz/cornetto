"""
Registry for creating sampling methods for context window engineering.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from typing import Callable
from omegaconf import DictConfig

from src.utils.context_utils.sampling_methods import oracle_at_k, random_at_k, random_with_oracle_at_k

VALID_SAMPLING_REGISTRY = {
    "oracle": oracle_at_k,
    "random": random_at_k,
    "random_with_oracle": random_with_oracle_at_k,
} 

# =========================================================================== #
#                          Context Sampling Registry                          #
# =========================================================================== #
def create_sampling_method(
    model_config: DictConfig, 
) -> Callable:
    """
    Registry function to create a sampling method for model context window.
    
    Args:
        model_config (DictConfig): Configuration containing sampling method type.
        
    Returns:
        Sampling method function for context window engineering.
    """
    try:
        model_dict = dict(model_config)
        name = model_dict.pop("context_sampling")
        return VALID_SAMPLING_REGISTRY[name]
    except KeyError:
        valid_methods = list(VALID_SAMPLING_REGISTRY.keys())
        raise ValueError(
            f"Unknown sampling method: '{name}'. "
            f"Valid options are: {valid_methods}"
        )