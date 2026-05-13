"""
Script to implement benchmark zero-shot learning pipeline.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import hydra
import warnings
import logging
from omegaconf import DictConfig

from src.utils.data_utils.cornetto_generation import CornettoDataset
from src.modules.network_envs.envs_registry import create_network_env

from src.modules.ZeroShot import ZeroShot
from src.modules.ZeroShotRepair import ZeroShotRepair
from src.modules.ZeroShotRetrieval import ZeroShotRetrieval
from src.modules.MiniAgent import MiniAgent

# Pipeline mode registry
PIPELINES = {
    "naive": ZeroShot,
    "repair": ZeroShotRepair,
    "retrieval": ZeroShotRetrieval,
    "agentic": MiniAgent,
}

# Ignore mainly hydra based warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


# =========================================================================== #
#                           Zero-Shot Main Function                           #
# =========================================================================== #
@hydra.main(config_path="../../configs", config_name="zero_shot")
def main(config: DictConfig) -> None:
    """
    Main function for zero-shot benchmarking with Hydra configuration.

    Args:
        config (DictConfig): Hydra configuration object containing all parameters.
    """
    # Save directory is automatically managed by Hydra
    save_dir = f"{os.getcwd()}/"
    logger.info(f"Saving results to {save_dir}")

    try:
        # Load external network environment
        net_env = create_network_env(config.net_env)

        additional_context = dict(getattr(config, "additional_context", {}) or {})

        # Initialize data generation pipeline
        data_pipe = CornettoDataset(
            **config.model,
            **additional_context,
            load_path=config.data.load_path,
            save_path=config.data.save_path,
            prompt_style=config.prompt_style,
            seed=config.data.seed,
            scenario_filter=list(getattr(config.data, "scenario_filter", None) or []) or None,
        )

        # Generate IFT dataset
        data = data_pipe.generate_scenarios()
    
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return {"Error": str(e)}
        
    try:
        # Build pipeline kwargs
        pipeline_kwargs = {
            "system_prompt": config.system_message,
            "split_ratio": config.data.split_ratio,
            "seed": config.data.seed,
            "data": data,
            "net_env": net_env,
            "save_dir": save_dir,
            "prompt_style": config.prompt_style,
            "parser_name": getattr(config, "parser_name", None),
            "diagnosis_judge_config": getattr(config, "diagnosis_judge", None),
        }

        # Repair mode: access to prior naive run settings
        if config.pipeline_mode == "repair":
            pipeline_kwargs.update({
                "naive_results_dir": getattr(config, "naive_results_dir", None),
                "naive_success_threshold": getattr(
                    config, "naive_success_threshold", 1.0
                ),
                "naive_retry_parse_failures": getattr(
                    config, "naive_retry_parse_failures", True
                ),
                "naive_retry_errors": getattr(
                    config, "naive_retry_errors", True
                ),
                "naive_retry_low_scores": getattr(
                    config, "naive_retry_low_scores", True
                ),
            })

        # Agentic mode: agent-specific settings
        if config.pipeline_mode == "agentic":
            agentic_cfg = getattr(config, "agentic", {}) or {}
            pipeline_kwargs["max_agent_steps"] = getattr(
                agentic_cfg, "max_steps", 30
            )
            pipeline_kwargs["verification_mode"] = getattr(
                agentic_cfg, "verification_mode", "per_step"
            )
            pipeline_kwargs["context_prefill"] = getattr(
                agentic_cfg, "context_prefill", False
            )
        
        # Initialize pipeline
        model = PIPELINES[config.pipeline_mode](
            **config.model,
            **pipeline_kwargs,
        )
        
        # Run model inference
        results = model.inference_and_eval(
            feedback_regime=config.feedback.enabled,
            max_attempts=config.feedback.max_attempts,
            similarity_threshold=config.feedback.similarity_threshold,
        )

        return results
        
    except Exception as e:
        logger.error(f"Zero-shot benchmark experiment failed: {str(e)}")
        return {"Error": str(e)}
        
    finally:
        # Clear CUDA memory
        #torch.cuda.empty_cache()

        # Clear stale model cache 
        #model.clear_memory()

        # Clear any external tool session
        if net_env:
            net_env.cleanup()


if __name__ == "__main__":
    main()