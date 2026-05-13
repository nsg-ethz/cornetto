"""
Utility module for formatting routing table and forwarding entry differences
for inclusion in prompts. Provides token-limited, human-readable summaries.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from src.utils.token_counter import token_counter

# Set up logging
logger = logging.getLogger(__name__)


# =========================================================================== #
#                     Routing Table Diff Formatter                            #
# =========================================================================== #
def format_route_diffs(
    route_diffs_data: Dict[str, Any],
    max_tokens: int = 15000,
    model_name: str = "gpt-5-mini"
) -> Tuple[str, int]:
    """
    Format routing table differences (RIB diffs) into a human-readable summary.
    
    The route_diffs.log file contains JSON objects per fault showing:
    - Added/removed routes with details (Node, Network, Protocol, Next_Hop, etc.)
    - Changed next-hop entries
    
    Note: Fault names/types are NOT included in output to avoid revealing the diagnosis.
    
    Args:
        route_diffs_data: Parsed route diffs data from route_diffs.log
        max_tokens: Maximum token budget for the formatted output
        model_name: Model name for tokenizer
        
    Returns:
        Tuple of (formatted_text, token_count)
    """
    if not route_diffs_data:
        return "", 0
    
    # Aggregate all route changes across faults (without revealing fault types)
    all_removed = []
    all_added = []
    all_changed_nh = []
    total_baseline = 0
    total_candidate = 0
    
    for fault_name, diff_info in route_diffs_data.items():
        if not isinstance(diff_info, dict):
            continue
        
        total_baseline = max(total_baseline, diff_info.get("baseline_route_count", 0))
        total_candidate = diff_info.get("candidate_route_count", total_candidate)
        
        all_removed.extend(diff_info.get("removed_samples", []))
        all_added.extend(diff_info.get("added_samples", []))
        all_changed_nh.extend(diff_info.get("changed_next_hop_samples", []))
    
    # Deduplicate based on (Node, Network) tuple
    def dedupe_routes(routes: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for route in routes:
            key = (route.get("Node", ""), route.get("Network", ""))
            if key not in seen:
                seen.add(key)
                unique.append(route)
        return unique
    
    all_removed = dedupe_routes(all_removed)
    all_added = dedupe_routes(all_added)
    
    header = """### Routing Table Differences (RIB Diffs)
These differences show how the routing information base changed due to the fault.
Each entry indicates routes that were added, removed, or had their next-hop changed.

**Column meanings:**
- Node: The router where this route exists
- Network: The destination prefix
- Protocol: Routing protocol (bgp, ibgp, ospf, static, etc.)
- Next_Hop_IP: The IP address of the next hop

**Summary:**
"""
    # Summary counts
    summary = f"  - Baseline routes: {total_baseline}, After fault: {total_candidate}\n"
    summary += f"  - Routes removed: {len(all_removed)}, Routes added: {len(all_added)}, Next-hop changed: {len(all_changed_nh)}\n"

    output_text = header + summary
    output_tokens = token_counter(output_text, model_name)

    def append_with_budget(text: str) -> bool:
        nonlocal output_text, output_tokens
        if not text:
            return True

        candidate_text = output_text + text
        candidate_tokens = token_counter(candidate_text, model_name, log_count=False)
        if candidate_tokens > max_tokens:
            truncation = f"\n[Output truncated due to {max_tokens} token limit]\n"
            trunc_candidate = output_text + truncation
            trunc_tokens = token_counter(trunc_candidate, model_name, log_count=True)
            if trunc_tokens <= max_tokens:
                output_text = trunc_candidate
                output_tokens = trunc_tokens
            return False

        output_text = candidate_text
        output_tokens = candidate_tokens
        return True

    # Format removed routes (most important for diagnosis)
    if all_removed:
        if not append_with_budget("\n**Removed Routes (should be present but missing):**\n"):
            return output_text, output_tokens

        for route in all_removed:
            node = route.get("Node", "?")
            network = route.get("Network", "?")
            protocol = route.get("Protocol", "?")
            next_hop = route.get("Next_Hop_IP", route.get("Next_Hop", "?"))
            if not append_with_budget(f"  - {node}: {network} via {next_hop} ({protocol})\n"):
                return output_text, output_tokens

    # Format added routes (unexpected routes)
    if all_added:
        if not append_with_budget("\n**Added Routes (present but should not be):**\n"):
            return output_text, output_tokens

        for route in all_added:
            node = route.get("Node", "?")
            network = route.get("Network", "?")
            protocol = route.get("Protocol", "?")
            next_hop = route.get("Next_Hop_IP", route.get("Next_Hop", "?"))
            if not append_with_budget(f"  - {node}: {network} via {next_hop} ({protocol})\n"):
                return output_text, output_tokens

    # Format changed next-hop entries
    if all_changed_nh:
        if not append_with_budget("\n**Changed Next-Hop (route exists but path changed):**\n"):
            return output_text, output_tokens

        for change in all_changed_nh:
            node = change.get("Node", "?")
            network = change.get("Network", "?")
            old_nh = change.get("old_next_hop", "?")
            new_nh = change.get("new_next_hop", "?")
            if not append_with_budget(f"  - {node}: {network} changed from {old_nh} to {new_nh}\n"):
                return output_text, output_tokens

    return output_text, output_tokens


# =========================================================================== #
#                   Forwarding Entry Diff Formatter                           #
# =========================================================================== #
def format_forwarding_diffs(
    forwarding_diffs_data: Dict[str, Any],
    max_tokens: int = 15000,
    model_name: str = "gpt-5-mini"
) -> Tuple[str, int]:
    """
    Format forwarding entry differences (FIB diffs) into a human-readable summary.
    
    The forwarding_diffs.log file contains forwarding graph groups showing:
    - Forwarding edges between routers
    - Actions taken (forwarded, accepted, dropped)
    - Interface and next-hop information per prefix
    
    Note: Fault names/types are NOT included in output to avoid revealing the diagnosis.
    
    Args:
        forwarding_diffs_data: Parsed forwarding diffs data from forwarding_diffs.log
        max_tokens: Maximum token budget for the formatted output
        model_name: Model name for tokenizer
        
    Returns:
        Tuple of (formatted_text, token_count)
    """
    if not forwarding_diffs_data:
        return "", 0
    
    # Aggregate all forwarding changes across faults (without revealing fault types)
    all_baseline_groups = []
    all_candidate_groups = []
    total_affected_groups = 0
    
    for fault_name, diff_info in forwarding_diffs_data.items():
        if not isinstance(diff_info, dict):
            continue
        
        total_affected_groups += diff_info.get("total_groups", 0)
        by_snapshot = diff_info.get("by_snapshot", {})
        all_baseline_groups.extend(by_snapshot.get("baseline", []))
        all_candidate_groups.extend(by_snapshot.get("candidate", []))
    
    # Deduplicate based on signature
    def dedupe_groups(groups: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for group in groups:
            sig = group.get("signature", str(group.get("prefixes", [])))
            if sig not in seen:
                seen.add(sig)
                unique.append(group)
        return unique
    
    all_baseline_groups = dedupe_groups(all_baseline_groups)
    all_candidate_groups = dedupe_groups(all_candidate_groups)

    def group_key(group: Dict[str, Any]) -> str:
        return group.get("signature", str(group.get("prefixes", [])))

    def normalize_group(group: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
        prefixes = group.get("prefixes", group.get("fec_prefixes", [])) or []
        norm_prefixes = tuple(sorted(str(prefix) for prefix in prefixes))

        edges = group.get("edges", []) or []
        norm_edges = tuple(sorted(
            (
                str(edge.get("src", "")),
                str(edge.get("dst", "")),
                str(edge.get("action", "")),
                str(edge.get("next_hop_interface", "")),
                str(edge.get("next_hop_ip", "")),
            )
            for edge in edges
        ))

        return norm_prefixes, norm_edges

    baseline_map = {group_key(group): group for group in all_baseline_groups}
    candidate_map = {group_key(group): group for group in all_candidate_groups}

    affected_keys = set()
    for key in set(baseline_map) | set(candidate_map):
        baseline_group = baseline_map.get(key)
        candidate_group = candidate_map.get(key)
        if baseline_group is None or candidate_group is None:
            affected_keys.add(key)
            continue
        if normalize_group(baseline_group) != normalize_group(candidate_group):
            affected_keys.add(key)

    # Hard filter: only include groups whose baseline/candidate differ.
    if affected_keys:
        all_baseline_groups = [group for group in all_baseline_groups if group_key(group) in affected_keys]
        all_candidate_groups = [group for group in all_candidate_groups if group_key(group) in affected_keys]
        total_affected_groups = len(affected_keys)
    else:
        all_baseline_groups = []
        all_candidate_groups = []
        total_affected_groups = 0
    
    header = """### Forwarding Entry Differences (FIB Diffs)
These differences show how actual packet forwarding behavior changed due to the fault.
Unlike routing tables (RIB), these represent the data plane - how packets actually flow.

**Terminology:**
- src: Source router initiating the forwarding
- dst: Destination (next router or final prefix)
- action: What happens to the packet (forwarded, accepted, dropped)
- next_hop_interface: Egress interface used
- next_hop_ip: IP address of next hop (if applicable)

**Reading the diffs:**
- "Expected" entries: Forwarding paths that should exist but are currently broken
- "Current" entries: Forwarding paths that exist after the fault was introduced
- Compare these to understand what forwarding behavior needs to be restored

**Summary:**
"""
    summary = f"  - Total forwarding equivalence classes affected: {total_affected_groups}\n"
    summary += f"  - Expected forwarding groups: {len(all_baseline_groups)}, Current forwarding groups: {len(all_candidate_groups)}\n"

    output_text = header + summary
    output_tokens = token_counter(output_text, model_name, log_count=True)

    def append_with_budget(text: str) -> bool:
        nonlocal output_text, output_tokens
        if not text:
            return True

        candidate_text = output_text + text
        candidate_tokens = token_counter(candidate_text, model_name, log_count=False)
        if candidate_tokens > max_tokens:
            truncation = f"\n[Output truncated due to {max_tokens} token limit]\n"
            trunc_candidate = output_text + truncation
            trunc_tokens = token_counter(trunc_candidate, model_name, log_count=True)
            if trunc_tokens <= max_tokens:
                output_text = trunc_candidate
                output_tokens = trunc_tokens
            return False

        output_text = candidate_text
        output_tokens = candidate_tokens
        return True

    # Process baseline (expected forwarding paths)
    if all_baseline_groups:
        if not append_with_budget("\n**Expected Forwarding Paths (should exist but may be broken):**\n"):
            return output_text, output_tokens

        for group in all_baseline_groups:
            prefixes = group.get("prefixes", group.get("fec_prefixes", []))
            prefix_str = ", ".join(prefixes)
            if not append_with_budget(f"  Prefixes: {prefix_str}\n"):
                return output_text, output_tokens

            edges = group.get("edges", [])
            for edge in edges:
                src = edge.get("src", "?")
                dst = edge.get("dst", "?")
                action = edge.get("action", "?")
                iface = edge.get("next_hop_interface", "")
                nh_ip = edge.get("next_hop_ip", "")

                path_info = f"{src} -> {dst} ({action})"
                if iface:
                    path_info += f" via {iface}"
                if nh_ip:
                    path_info += f" [{nh_ip}]"

                if not append_with_budget(f"    {path_info}\n"):
                    return output_text, output_tokens

            if not append_with_budget("\n"):
                return output_text, output_tokens

    # Process candidate (current faulty forwarding paths)
    if all_candidate_groups:
        if not append_with_budget("\n**Current Forwarding Paths (after fault):**\n"):
            return output_text, output_tokens

        for group in all_candidate_groups:
            prefixes = group.get("prefixes", group.get("fec_prefixes", []))
            prefix_str = ", ".join(prefixes)
            if not append_with_budget(f"  Prefixes: {prefix_str}\n"):
                return output_text, output_tokens

            edges = group.get("edges", [])
            for edge in edges:
                src = edge.get("src", "?")
                dst = edge.get("dst", "?")
                action = edge.get("action", "?")
                iface = edge.get("next_hop_interface", "")
                nh_ip = edge.get("next_hop_ip", "")

                path_info = f"{src} -> {dst} ({action})"
                if iface:
                    path_info += f" via {iface}"
                if nh_ip:
                    path_info += f" [{nh_ip}]"

                if not append_with_budget(f"    {path_info}\n"):
                    return output_text, output_tokens

            if not append_with_budget("\n"):
                return output_text, output_tokens

    return output_text, output_tokens


# =========================================================================== #
#                         Log File Parsers                                    #
# =========================================================================== #
_DIFF_HEADER_RE = re.compile(
    r"^===\s*(.*?)\s*===\s*$|^===\s*(.*?)\s*$",
    re.MULTILINE,
)


def _parse_diff_log(content: str) -> Dict[str, Any]:
    result = {}
    matches = list(_DIFF_HEADER_RE.finditer(content))
    for i, match in enumerate(matches):
        header_text = (match.group(1) or match.group(2) or "").strip()
        if "::" not in header_text:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start:end]

        json_start = block.find("{")
        json_end = block.rfind("}") + 1
        if json_start == -1 or json_end <= 0:
            continue

        json_str = block[json_start:json_end]
        try:
            data = json.loads(json_str)
            fault_name = header_text.rstrip("-").strip()
            result[fault_name] = data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for {header_text}: {e}")
            continue

    return result


def parse_route_diffs_log(log_path: str) -> Dict[str, Any]:
    """
    Parse the route_diffs.log file which contains JSON blocks per fault.
    
    The file format is:
    ```
    # Route diffs for scenario ...
    === fault 01 :: fault.type ===
    { JSON object }
    ---
    === fault 02 :: fault.type ===
    { JSON object }
    ---
    ```
    
    Args:
        log_path: Path to route_diffs.log file
        
    Returns:
        Dictionary mapping fault names to their diff data
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        return _parse_diff_log(content)
    except Exception as e:
        logger.error(f"Failed to parse route_diffs.log: {e}")
        
    return {}


def parse_forwarding_diffs_log(log_path: str) -> Dict[str, Any]:
    """
    Parse the forwarding_diffs.log file which contains JSON blocks per fault.
    
    The file format is similar to route_diffs.log:
    ```
    # Forwarding table diffs for scenario ...
    === fault 01 :: fault.type ===
    { JSON object }
    ---
    ```
    
    Args:
        log_path: Path to forwarding_diffs.log file
        
    Returns:
        Dictionary mapping fault names to their diff data
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        return _parse_diff_log(content)
    except Exception as e:
        logger.error(f"Failed to parse forwarding_diffs.log: {e}")
        
    return {}
