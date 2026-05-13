# =========================================================================== #
#  REPLACE src/utils/parser_utils/parser_registry.py WITH THIS FILE           #
# =========================================================================== #

"""
Registry for creating parser instances based on network environment selection.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from typing import Callable, Any
import logging

from src.utils.parser_utils.parsers import (
    diff_patch_parser,
    full_config_parser,
    search_replace_parser,
    agent_response_parser,
)

logger = logging.getLogger(__name__)

VALID_PARSERS_REGISTRY = {
    "Batfish": diff_patch_parser,
    "Config2Spec": full_config_parser,
    # Explicit parser names
    "diff_patch": diff_patch_parser,
    "search_replace": search_replace_parser,
    "full_config": full_config_parser,
    "agent_response": agent_response_parser,
}


# =========================================================================== #
#                               Parser Registry                               #
# =========================================================================== #
def create_parser(net_env: Any, parser_name: str = None) -> Callable:
    """
    Registry function to get the appropriate parser based on network environment instance.
    
    Args:
        net_env (Any): Network environment instance (e.g., Batfish or Config2Spec object).
        parser_name (str): Optional explicit parser selection.
        
    Returns:
        Parser function for the specified environment.
    """
    if parser_name:
        if parser_name in VALID_PARSERS_REGISTRY:
            logger.info(f"Selected parser by name: {parser_name}")
            return VALID_PARSERS_REGISTRY[parser_name]
        else:
            valid = list(VALID_PARSERS_REGISTRY.keys())
            raise ValueError(f"Unknown parser: {parser_name}. Must be one of {valid}")

    if net_env is None:
        logger.error("No network environment provided!")
        raise
    
    # Get the class name of the environment instance
    env_class_name = net_env.__class__.__name__
    
    # Look up the appropriate parser
    if env_class_name in VALID_PARSERS_REGISTRY:
        logger.info(f"Selected parser for environment: {env_class_name}")
        return VALID_PARSERS_REGISTRY[env_class_name]
    else:
        valid_envs = [k for k in VALID_PARSERS_REGISTRY.keys() if k[0].isupper()]
        raise ValueError(
            f"Unknown network environment type: {env_class_name}. "
            f"Must be one of {valid_envs}"
        )