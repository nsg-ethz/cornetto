import logging
from omegaconf import DictConfig
from typing import Tuple, Dict, Any

from src.utils.data_utils.cornetto_generation import CornettoDataset
from src.modules.network_envs.envs_registry import create_network_env

logger = logging.getLogger(__name__)


def load_data(config: DictConfig) -> Tuple[Dict[str, Dict[str, Any]], Any]:
    """
    Load the Cornetto dataset and network environment using the same settings
    as the main zero-shot pipeline.

    Returns:
        Tuple of (data, net_env)
    """
    try:
        net_env = create_network_env(config.net_env)
    except Exception as e:
        logger.error(f"Failed to initialize network environment: {e}")
        raise

    try:
        additional_context = dict(getattr(config, "additional_context", {}) or {})
        data_pipe = CornettoDataset(
            **config.model,
            **additional_context,
            load_path=config.data.load_path,
            save_path=config.data.save_path,
            prompt_style=config.prompt_style,
            seed=config.data.seed,
        )
        data = data_pipe.generate_scenarios()
    except Exception as e:
        logger.error(f"Failed to load/generate dataset: {e}")
        raise

    return data, net_env
