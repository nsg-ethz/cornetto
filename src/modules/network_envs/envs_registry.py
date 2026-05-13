"""
Registry for creating network environment instances based on configuration.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from typing import Dict, Any, Tuple
from omegaconf import DictConfig

from src.modules.network_envs.envs.batfish import Batfish
#from src.modules.network_envs.envs.config2spec import Config2Spec

VALID_ENVS_REGISTRY = {
    "Batfish": Batfish,
   # "Config2Spec": Config2Spec,
} 

# =========================================================================== #
#                         Network Environment Registry                        #
# =========================================================================== #
def create_network_env(
    env_config: DictConfig, 
):
    """
    Registry function to create a network environment instance.
    
    Args:
        env_config (DictConfig): Type of environment to create.
        
    Returns:
        An instance of external network environment.        
    """
    try:
        env_dict = dict(env_config)
        name = env_dict.pop("name")
        return VALID_ENVS_REGISTRY[name](**env_dict)
    except KeyError:
        raise ValueError(f"Unknown network environment type: {name}. Use 'Config2Spec' or 'Batfish'")