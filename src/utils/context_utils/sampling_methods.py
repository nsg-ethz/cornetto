""" 
This script contains methods to perform context engineering with oracle and 
random sampling, such that final prompts would not exceed window sizes. 
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import json
import random
import logging
from typing import Dict, List, Tuple, Any

from src.utils.token_counter import token_counter
from src.utils.context_utils.storage import get_context_window

logger = logging.getLogger(__name__)

def _format_topology_text(topology: Any) -> str:
    """
    Function to transform unstructured topologies into strings.

    Args:
        topology (Any): Topology data.
    
    Returns:
        Structured string representing the topology.
    """
    if isinstance(topology, str):
        return topology
    try:
        return json.dumps(topology, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return str(topology)

def _resolve_context_token_budget(
    model: str | None,
    kwargs: dict,
) -> Tuple[int, int]:
    """
    Function to override the token budget, just in case.

    Args:
        model (str | None): Model name or identifier.
    
    Returns:
        If not overridden, safe context limit and original one. Overridden budget otherwise.
    """
    override = (
        kwargs.get("context_max_tokens")
        or kwargs.get("max_context_tokens")
        or kwargs.get("context_window_tokens")
    )
    if override is None or override == "full":
        return get_context_window(model)

    budget = int(override)
    if budget <= 0:
        raise ValueError(f"Context token budget must be > 0, got {budget}")
    
    # As in the 'full' case, reserve 60% of the window for config files only
    return 0.6 * budget, budget


# =========================================================================== #
#                           Oracle Context Filling                            #
# =========================================================================== #
def oracle_at_k(
    original_configs: Dict[str, str],
    final_configs: Dict[str, str],
    topology: Any,
    preds: Any,
    **kwargs,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Oracle context filling method to keep faulty configs only, within 'k' token limit.

    Args:
        original_configs (Dict[str, str]): Original configuration files.
        final_configs (Dict[str, str]): Faulty configuration files.
        topology (Any): Topology of the given network.
        preds (Any): Specifications of the given network.
        
    Returns:
        Tuple of selected configurations, names for included/excluded routers.
    """
    # Get model name and safe token limit for configs
    model = kwargs.get("model_name", None)
    max_context_tokens = kwargs.get("max_context_tokens", None)

    # Context filling conditioned on a user-specified context cap and full original model context window
    if max_context_tokens == "full":
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)

        # At the same time, get safe token limit for preds and topology
        topology_budget = token_counter(_format_topology_text(topology))
        preds_budget = token_counter("\n".join(preds) if isinstance(preds, list) else str(preds))
        network_info_budget = max(0, int((original_budget - (topology_budget + preds_budget)) * 0.9))

        # Our custom condition to determine the max safe context window limit
        max_tokens = min(config_budget, network_info_budget)

    else:
        # Otherwise, just assign max tokens to the user specified budget for the config files
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)
        max_tokens = config_budget

    # Get a list of all router names
    all_routers = list(original_configs.keys())
    
    faulty_routers = []
    
    # Check if routers are perturbed
    for router in all_routers:
        if original_configs[router] != final_configs[router]:
            faulty_routers.append(router)
    
    logger.info(f"Found {len(faulty_routers)} faulty routers out of {len(all_routers)} total routers")
    
    if not faulty_routers:
        logger.warning("No faulty routers found - configs are identical")
        return {}, [], []
    
    # Shuffle the faulty routers for random selection
    candidate_routers = faulty_routers.copy()
    random.shuffle(candidate_routers)
    
    # Track selected configs and token count
    selected_configs = {}
    total_tokens = 0
    included_routers = []
    excluded_routers = []
    
    # Try to include configs one by one until we hit the token limit
    for router in candidate_routers:
        config_text = final_configs[router]
        config_tokens = token_counter(config_text, model)
        
        # Warn if a single config exceeds the limit
        if config_tokens > max_tokens:
            logger.warning(f"{router} alone exceeds token limit ({config_tokens} > {max_tokens})")
        
        # Check if adding this config would exceed the limit
        if total_tokens + config_tokens <= max_tokens:
            selected_configs[router] = config_text
            included_routers.append(router)
            total_tokens += config_tokens
            logger.debug(f"Added {router}: {config_tokens} tokens (total: {total_tokens}/{max_tokens})")
        else:
            excluded_routers.append(router)
            logger.debug(f"Excluded {router}: would exceed token limit ({config_tokens} tokens)")
    
    # Log summary
    logger.info(f"Oracle context filling complete:")
    logger.info(f"  Included: {len(included_routers)} routers ({total_tokens} tokens)")
    logger.info(f"  Excluded: {len(excluded_routers)} routers")
    
    if excluded_routers:
        logger.warning(f"Could not include all faulty configs due to token limit. Excluded: {excluded_routers}")
        
    return selected_configs, included_routers, excluded_routers


# =========================================================================== #
#                           Random Context Filling                            #
# =========================================================================== #
def random_at_k(
    original_configs: Dict[str, str],
    final_configs: Dict[str, str],
    topology: Any,
    preds: Any,
    **kwargs,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Random context filling method to sample from all routers, within 'k' token limit.
    Note: This includes both faulty and non-faulty configs randomly.

    Args:
        original_configs (Dict[str, str]): Original configuration files.
        final_configs (Dict[str, str]): Faulty configuration files.
        topology (Any): Topology of the given network.
        preds (Any): Specifications of the given network.
        
    Returns:
        Tuple of selected configurations, names for included/excluded routers.
    """
    # Get model name and safe token limit for configs
    model = kwargs.get("model_name", None)
    max_context_tokens = kwargs.get("max_context_tokens", None)

    # Context filling conditioned on a user-specified context cap and full original model context window
    if max_context_tokens == "full":
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)

        # At the same time, get safe token limit for preds and topology
        topology_budget = token_counter(_format_topology_text(topology))
        preds_budget = token_counter("\n".join(preds) if isinstance(preds, list) else str(preds))
        network_info_budget = max(0, int((original_budget - (topology_budget + preds_budget)) * 0.9))

        # Our custom condition to determine the max safe context window limit
        max_tokens = min(config_budget, network_info_budget)

    else:
        # Otherwise, just assign max tokens to the user specified budget for the config files
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)
        max_tokens = config_budget

    # Get a list of all router names
    all_routers = list(final_configs.keys())

    logger.info(f"Sampling from all {len(all_routers)} routers")

    # Shuffle all routers for random selection
    candidate_routers = all_routers.copy()
    random.shuffle(candidate_routers)

    # Track selected configs and token count
    selected_configs = {}
    total_tokens = 0
    included_routers = []
    excluded_routers = []
   
    # Try to include configs one by one until we hit the token limit
    for router in candidate_routers:
        config_text = final_configs[router]
        config_tokens = token_counter(config_text, model)
        
        # Warn if a single config exceeds the limit
        if config_tokens > max_tokens:
            logger.warning(f"{router} alone exceeds token limit ({config_tokens} > {max_tokens})")
        
        # Check if adding this config would exceed the limit
        if total_tokens + config_tokens <= max_tokens:
            selected_configs[router] = config_text
            included_routers.append(router)
            total_tokens += config_tokens
            logger.debug(f"Added {router}: {config_tokens} tokens (total: {total_tokens}/{max_tokens})")
        else:
            excluded_routers.append(router)
            logger.debug(f"Excluded {router}: would exceed token limit ({config_tokens} tokens)")
    
    # Log summary
    logger.info(f"Random context filling complete:")
    logger.info(f"  Included: {len(included_routers)} routers ({total_tokens} tokens)")
    logger.info(f"  Excluded: {len(excluded_routers)} routers")
    
    if excluded_routers:
        logger.warning(f"Could not include all configs due to token limit. Excluded: {excluded_routers}")
        
    return selected_configs, included_routers, excluded_routers


# =========================================================================== #
#                     Random Context Filling with Oracle                      #
# =========================================================================== #
def random_with_oracle_at_k(
    original_configs: Dict[str, str],
    final_configs: Dict[str, str],
    topology: Any,
    preds: Any,
    **kwargs,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Random context filling method that guarantees oracle (faulty) configs are included,
    but shuffles them randomly among all selected configs.

    The method first ensures all faulty configs can fit within the token budget,
    then fills remaining space with random non-faulty configs. Finally, all selected
    configs are shuffled together so faulty configs appear at random positions.

    Args:
        original_configs (Dict[str, str]): Original configuration files.
        final_configs (Dict[str, str]): Faulty configuration files.
        topology (Any): Topology of the given network.
        preds (Any): Specifications of the given network.
                
    Returns:
        Tuple of selected configurations, names for included/excluded routers.
    """
    # Get model name and safe token limit for configs
    model = kwargs.get("model_name", None)
    max_context_tokens = kwargs.get("max_context_tokens", None)

    # Context filling conditioned on a user-specified context cap and full original model context window
    if max_context_tokens == "full":
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)

        # At the same time, get safe token limit for preds and topology
        topology_budget = token_counter(_format_topology_text(topology))
        preds_budget = token_counter("\n".join(preds) if isinstance(preds, list) else str(preds))
        network_info_budget = max(0, int((original_budget - (topology_budget + preds_budget)) * 0.85))

        # Our custom condition to determine the max safe context window limit
        max_tokens = min(config_budget, network_info_budget)

    else:
        # Otherwise, just assign max tokens to the user specified budget for the config files
        config_budget, original_budget = _resolve_context_token_budget(model, kwargs)
        max_tokens = config_budget

    # Get a list of all router names
    all_routers = list(final_configs.keys())
    
    # Identify faulty (oracle) and non-faulty routers
    faulty_routers = []
    non_faulty_routers = []
    
    for router in all_routers:
        if original_configs[router] != final_configs[router]:
            faulty_routers.append(router)
        else:
            non_faulty_routers.append(router)
    
    logger.info(f"Found {len(faulty_routers)} faulty routers and {len(non_faulty_routers)} non-faulty routers")

    # Track selected configs and token count
    selected_configs = {}
    total_tokens = 0
    included_routers = []
    excluded_routers = []
    
    # STEP 1: Reserve space for all oracle (faulty) configs first
    faulty_router_set = set(faulty_routers)
    
    for router in faulty_routers:
        config_text = final_configs[router]
        config_tokens = token_counter(config_text, model)
        
        # Warn if a single config exceeds the limit
        if config_tokens > max_tokens:
            logger.warning(f"Oracle config {router} alone exceeds token limit ({config_tokens} > {max_tokens})")
        
        # Check if adding this config would exceed the limit
        if total_tokens + config_tokens <= max_tokens:
            selected_configs[router] = config_text
            included_routers.append(router)
            total_tokens += config_tokens
            logger.debug(f"Reserved oracle {router}: {config_tokens} tokens (total: {total_tokens}/{max_tokens})")
        else:
            excluded_routers.append(router)
            logger.warning(f"Could not include oracle config {router}: would exceed token limit ({config_tokens} tokens)")
    
    oracle_count = len([r for r in included_routers if r in faulty_router_set])
    oracle_tokens = total_tokens
    logger.info(f"Oracle reservation complete: {oracle_count}/{len(faulty_routers)} faulty routers ({oracle_tokens} tokens)")
    
    # STEP 2: Fill remaining budget with random non-faulty configs
    # Shuffle non-faulty routers for random selection
    random.shuffle(non_faulty_routers)
    
    for router in non_faulty_routers:
        config_text = final_configs[router]
        config_tokens = token_counter(config_text, model)
        
        # Check if adding this config would exceed the limit
        if total_tokens + config_tokens <= max_tokens:
            selected_configs[router] = config_text
            included_routers.append(router)
            total_tokens += config_tokens
            logger.debug(f"Added random {router}: {config_tokens} tokens (total: {total_tokens}/{max_tokens})")
        else:
            excluded_routers.append(router)
            logger.debug(f"Excluded {router}: would exceed token limit ({config_tokens} tokens)")
    
    # STEP 3: Shuffle included_routers so faulty configs are at random positions
    random.shuffle(included_routers)
    
    # Rebuild selected_configs dict in the shuffled order
    shuffled_selected_configs = {router: selected_configs[router] for router in included_routers}
    
    # Log summary
    non_faulty_count = len(included_routers) - oracle_count
    logger.info(f"Random with oracle context filling complete:")
    logger.info(f"  Oracle configs included: {oracle_count}/{len(faulty_routers)}")
    logger.info(f"  Random configs included: {non_faulty_count}/{len(non_faulty_routers)}")
    logger.info(f"  Total included: {len(included_routers)} routers ({total_tokens} tokens)")
    logger.info(f"  Excluded: {len(excluded_routers)} routers")
    logger.info(f"  Configs shuffled: faulty configs now at random positions")
    
    if any(r in excluded_routers for r in faulty_routers):
        logger.warning(f"Could not include all oracle configs due to token limit!")
        
    return shuffled_selected_configs, included_routers, excluded_routers
