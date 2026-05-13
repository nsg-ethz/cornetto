"""
This module generates misconfiguration dataset for the Cornetto benchmark.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import json
import pandas as pd
import logging
from typing import Dict, Union, Any, List, Tuple

from src.utils.prompt_utils.prompts import _init_prompt as _diff_prompt
from src.utils.prompt_utils.prompts_alt_patch import _init_prompt as _search_replace_prompt
from src.utils.context_utils.sampling_registry import create_sampling_method
from src.utils.reproducibility import set_all_seeds
from src.utils.token_counter import token_counter
from src.utils.data_utils.diff_formatters import (
    parse_route_diffs_log,
    parse_forwarding_diffs_log,
    format_route_diffs,
    format_forwarding_diffs,
)

# Set up logging
logger = logging.getLogger(__name__)


# =========================================================================== #
#                         Cornetto Dataset Generation                         #
# =========================================================================== #
class CornettoDataset:
    """
    Class instance to generate network misconfigurations use-cases from ready-to-use
    configuration data. 
    """

    def __init__(
        self,
        load_path: str = None,
        save_path: str = None,
        prompt_style: str = "diff_patch",
        path: str = None,
        seed: int = 42,
        scenario_filter: List[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the Cornetto dataset generator.

        Args:
            load_path (str): Path to the folder where scenarios are held.
                Defaults to None.
            save_path (str): Path where prompt scenarios are held.
                Defaults to None.
            model (str): Model that is used for tokenization.
                Defaults to None.
            prompt_style (str): Prompt formatting style.
                Defaults to 'diff_patch'.
            path (str): Backward-compatible alias; if provided, used for load_path (and save_path if unset).
                Defaults to None.
            seed (int): Random seed.
                Defaults to 42.
            scenario_filter (list[str]): If provided, only scenarios whose directory name
                appears in this list are loaded. Defaults to None (load all).
        """
        if load_path is None and path is not None:
            load_path = path
        if save_path is None:
            save_path = load_path if load_path is not None else path

        # Set the global data path
        self.load_path = load_path

        # Set the global save path
        self.save_path = save_path

        # Set the prompting style, used for prompt/output formatting
        self.prompt_style = prompt_style
    
        # Set object to store the final dataset 
        self.dataset = {}

        # Get extra variables (e.g. 'model_name')
        self.kwargs = kwargs

        # Optional allowlist of scenario directory names to load
        self.scenario_filter = set(scenario_filter) if scenario_filter else None

        # Set reproducibility seed
        self.seed = seed
        set_all_seeds(self.seed)

    def generate_scenarios(self) -> Dict[str, Union[str, Dict]]:
        """
        Generate misconfiguration prompts/scenarios from the common pool.

        Returns:
            Final prompts for each scenario with metadata.
        """
        try:
            logger.info(f"Directory exists at: {os.path.exists(self.load_path)}")

            if not os.path.exists(self.load_path):
                logger.warning(f"Error: Directory {self.load_path} does not exist!")
                return {}

            logger.info(f"Number of scenarios: {len(os.listdir(self.load_path)) - 1}")

            total_scenarios_found, total_scenarios_loaded = 0, 0
            all_scenarios = {}

            # Iterate over the root data directory
            for idx, scenario in enumerate(sorted(os.listdir(self.load_path))):
                artifact_path = os.path.join(self.load_path, scenario)
                
                # Skip if not a directory
                if not os.path.isdir(artifact_path):
                    continue

                # Skip scenarios not in the allowlist (if one is configured)
                if self.scenario_filter and scenario not in self.scenario_filter:
                    continue

                artifacts = sorted(os.listdir(artifact_path))

                # Check if scenario path is not empty
                if artifacts:
                    logger.info(f"{artifacts} found at: {artifact_path}")
                    total_scenarios_found += 1

                #-------------------#
                #   Load Topology   #
                #-------------------#
                # Check if topology folder can be found 
                topology_path = os.path.join(artifact_path, "data_and_metrics")
                if not os.path.exists(topology_path):
                    logger.warning(f"Could not find data_and_metrics in {artifact_path}")
                    continue

                # Load the topology file
                with open(os.path.join(topology_path, "topology.json")) as f:
                    topology = json.load(f)

                logger.info("Topology is loaded successfully")

                #--------------------#
                #    Load Configs    #
                #--------------------#
                # initial_configs are optional — not included in the public dataset.
                # oracle/random_with_oracle sampling requires them; random sampling does not.
                initial_config_path = os.path.join(artifact_path, "initial_configs/configs")
                if os.path.exists(initial_config_path):
                    try:
                        initial_configs = self.load_config(initial_config_path)
                    except Exception as e:
                        logger.error(f"Failed to load initial configs for {scenario}: {e}")
                        initial_configs = {}
                else:
                    logger.info(f"No initial_configs found in {artifact_path} — oracle sampling unavailable")
                    initial_configs = {}

                # Check if final configurations exist
                final_config_path = os.path.join(artifact_path, "final_configs/configs")
                if not os.path.exists(final_config_path):
                    logger.warning(f"Could not find final_config in {artifact_path}")
                    continue

                # Load all configuration files for the scenario
                try:
                    final_configs = self.load_config(final_config_path)
                except Exception as e:
                    logger.error(f"Failed to load configs for {scenario}: {e}")
                    continue

                if not final_configs:
                    logger.warning(f"Empty final configs for {scenario}, skipping")
                    continue

                #--------------------#
                #     Load Specs     #
                #--------------------#
                # Check if there is specifications folder
                predicates_path = os.path.join(artifact_path, "specifications")
                if not os.path.exists(predicates_path):
                    logger.warning(f"Could not find specifications folder in {artifact_path}")
                    continue
                
                # Load specifications/predicates - select largest file under token limit
                # Available percentages in descending order
                spec_percentages = ["100pct", "75pct", "50pct", "20pct", "10pct", "5pct", "2pct"]
                max_spec_tokens = 50_000  # Maximum tokens for specifications
                model_name = self.kwargs.get("model_name", "gpt-5-mini")
                
                spec_csv_path = None
                df_preds = None
                preds = None
                
                for pct in spec_percentages:
                    candidate_path = os.path.join(predicates_path, f"specifications_{pct}.csv")
                    if not os.path.exists(candidate_path):
                        continue
                    
                    # Load and process candidate specs
                    df_candidate = pd.read_csv(candidate_path)
                    # Drop any rows whose Status contains "intact"
                    df_candidate = df_candidate[~df_candidate["sources"].astype(str).str.contains("intact", case=False, na=False)]
                    candidate_preds = list(df_candidate.fillna("").astype(str).apply(lambda x: ','.join(x), axis=1))
                    
                    # Check token count
                    specs_text = "\n".join(candidate_preds)
                    specs_tokens = token_counter(specs_text, model_name)
                    
                    if specs_tokens <= max_spec_tokens:
                        spec_csv_path = candidate_path
                        df_preds = df_candidate
                        preds = candidate_preds
                        logger.info(f"Selected specifications_{pct}.csv ({specs_tokens} tokens)")
                        break
                    else:
                        logger.debug(f"specifications_{pct}.csv exceeds limit ({specs_tokens} > {max_spec_tokens} tokens)")
                
                if preds is None:
                    # Fallback to smallest available if all exceed limit
                    for pct in reversed(spec_percentages):
                        candidate_path = os.path.join(predicates_path, f"specifications_{pct}.csv")
                        if os.path.exists(candidate_path):
                            spec_csv_path = candidate_path
                            df_preds = pd.read_csv(candidate_path)
                            df_preds = df_preds[~df_preds["sources"].astype(str).str.contains("intact", case=False, na=False)]
                            preds = list(df_preds.fillna("").astype(str).apply(lambda x: ','.join(x), axis=1))
                            logger.warning(f"All specs exceed {max_spec_tokens} tokens, using smallest: specifications_{pct}.csv")
                            break
                
                if preds is None:
                    logger.warning(f"No specification files found in {predicates_path}")
                    continue

                logger.info("Specifications are loaded successfully")

                #---------------------------------#
                #   Load Route/Forwarding Diffs   #
                #---------------------------------#
                # Check if additional context options are enabled
                include_route_diffs = self.kwargs.get("include_route_diffs", False)
                include_forwarding_diffs = self.kwargs.get("include_forwarding_diffs", False)
                route_diffs_max_tokens = self.kwargs.get("route_diffs_max_tokens", 15000)
                forwarding_diffs_max_tokens = self.kwargs.get("forwarding_diffs_max_tokens", 15000)
                
                route_diffs_text = ""
                forwarding_diffs_text = ""
                
                if include_route_diffs:
                    route_diffs_path = os.path.join(topology_path, "route_diffs.log")
                    if os.path.exists(route_diffs_path):
                        route_diffs_data = parse_route_diffs_log(route_diffs_path)
                        route_diffs_text, route_tokens = format_route_diffs(
                            route_diffs_data,
                            max_tokens=route_diffs_max_tokens,
                            model_name=model_name
                        )
                        logger.info(f"Loaded route diffs ({route_tokens} tokens)")
                    else:
                        logger.warning(f"Route diffs not found at {route_diffs_path}")
                
                if include_forwarding_diffs:
                    forwarding_diffs_path = os.path.join(topology_path, "forwarding_diffs.log")
                    if os.path.exists(forwarding_diffs_path):
                        forwarding_diffs_data = parse_forwarding_diffs_log(forwarding_diffs_path)
                        forwarding_diffs_text, fwd_tokens = format_forwarding_diffs(
                            forwarding_diffs_data,
                            max_tokens=forwarding_diffs_max_tokens,
                            model_name=model_name
                        )
                        logger.info(f"Loaded forwarding diffs ({fwd_tokens} tokens)")
                    else:
                        logger.warning(f"Forwarding diffs not found at {forwarding_diffs_path}")

                #------------------------#
                #     Prepare Prompt     #
                #------------------------#
                # Store all scenarios with the selected artifacts
                scenario_id = f"Task-{idx}"
                prompt_fn = _search_replace_prompt if self.prompt_style == "search_replace" else _diff_prompt

                # After loading all network information (e.g. configs, preds), load and use selected context filling method
                sampling_fn = create_sampling_method(self.kwargs)
                selected_configs_only, included, excluded = sampling_fn(
                    original_configs=initial_configs,
                    final_configs=final_configs,
                    topology=topology,
                    preds=preds,
                    **self.kwargs
                )
                logger.info(f"Context window is filled with {sampling_fn.__name__} method")

                # Create the prompt via the template and network info
                # Pass additional context (route/forwarding diffs) if available
                all_scenarios[scenario_id] = {
                    "scenario_name": scenario,
                    "instruction": prompt_fn(
                        final_configs=selected_configs_only, 
                        topology=topology,
                        preds=preds,
                        route_diffs=route_diffs_text if route_diffs_text else None,
                        forwarding_diffs=forwarding_diffs_text if forwarding_diffs_text else None,
                    ),
                    "original_configs": initial_configs,
                    "faulty_configs": final_configs,
                    "selected_configs": selected_configs_only,
                    "included_routers": included,
                    "excluded_routers": excluded,
                    "original_specs": preds,
                    "specification_csv_path": spec_csv_path,
                    "topology": topology,
                    "route_diffs_included": bool(route_diffs_text),
                    "forwarding_diffs_included": bool(forwarding_diffs_text),
                }

                # Track scenarios
                total_scenarios_loaded += 1

            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            
            # Save the final dataset
            output_file = os.path.join(self.save_path, "cornetto_dataset.json")
            with open(output_file, "w") as f:
                json.dump(all_scenarios, f, indent=2)
            
            logger.info(f"Saved {len(all_scenarios)} scenarios to {output_file}")
            logger.info(f"Total found: {total_scenarios_found}, Total loaded: {total_scenarios_loaded}")

            return all_scenarios

        except Exception as e:
            logger.error(f"Error generating prompts for scenarios due to: {e}")
            return {}
        
    def load_config(self, config_dir: str) -> Dict[str, str]:
        """
        Load network configuration files from directory.
        
        Args:
            config_dir: Path to directory containing configuration files.
            
        Returns:
            Dictionary mapping router names to configurations.
        """
        config_storage = {}

        # Read each '.cfg' file and parse host name
        try:
            for filename in sorted(os.listdir(config_dir)):
                if filename.endswith(".cfg"):
                    with open(os.path.join(config_dir, filename), "r") as f:
                        config = f.read()
                        config_storage[filename] = config
            return config_storage
        except Exception as e:
            logger.error(f"Failed to load network configuration from {config_dir}: {str(e)}")
            raise

