"""
Utility script to generate and manage prompts for network troubleshooting tasks
using search-and-replace instructions instead of diff patches.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import json
from typing import Dict, Any, List, Optional

import textwrap
from typing import Dict, Any, Optional


def _format_preds(preds: Any) -> str:
    """
    Formatter for predicate/specification data.

    Args:
        preds (Any): Specifications of the given network.

    Returns:
        Formatted predicate/specification data as a string.
    """

    if isinstance(preds, (list, tuple)):
      return "\n".join(str(p) for p in preds)
    return str(preds)


def _format_topology(topology: Any) -> str:
    """
    Formatter for topology data.

    Args:
        topology (Any): Topology of the given network.

    Returns:
        Formatted topology data as a string.
    """
    if isinstance(topology, str):
      return topology
    try:
      return json.dumps(topology, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
      return str(topology)


# =========================================================================== #
#                          Starter Prompt Generation                          #
# =========================================================================== #
def _init_prompt(
    final_configs: Dict[str, str],
    topology: Any,
    preds: Any,
    route_diffs: Optional[str] = None,
    forwarding_diffs: Optional[str] = None,
) -> str:
    """
    Build initial prompt for the LLM with all the relevant information.

    Args:
        final_configs (Dict[str, str]): Faulty configurations.
        topology (Any): Topology of the given network.
        preds (Any): Specifications of the given network.
        route_diffs (Optional[str]): Formatted routing table differences.
        forwarding_diffs (Optional[str]): Formatted forwarding entry differences.
        
    Returns:
        Complete initial prompt for model in 'search and replace' format.
    """
    
    # Use textwrap.dedent to remove the python indentation from the string
    preds_text = _format_preds(preds)
    topology_text = _format_topology(topology)

    prompt_template = textwrap.dedent(f"""\
    A network fault has occurred, possibly due to human error, causing discrepancies in at least one of the configuration files. 
    The specifications/predicates that govern the network behavior have changed as a result. These specifications in 
    question are automatically derived from the router configurations and they describe the resulting forwarding behavior.

    ### Task Requirements:
    1. Your task is to identify which routers must be modified to restore the correct forwarding behavior. 
    2. In this regard, you are expected to generate the necessary modifications for each router configuration. 
    3. Provide your solution as explicit search-and-replace instructions for each router.
    4. Each instruction should specify the exact text to find and the text to replace it with—no line numbers or diff markers.

    ### Expected Output Format:
    - Output must be valid YAML only, with the following **order** of top-level keys: `routers`, then `metadata`, then `replacements`.
    - `routers`: ordered list of router filenames (**only** the ones that require changes). Do not list routers that need no edits.
    - `metadata`: block scalars `problem_diagnosis` and `proposed_fix`.
    - `replacements`: mapping where each key is a router filename that appears in `routers`, and each value is an ordered list of search/replace operations.
    - Each search-and-replace operation must contain two keys:
      - `search`: block scalar (`|`) containing the exact text to find in the faulty config (no regex, no ellipses).
      - `replace`: block scalar (`|`) containing the corrected text that should replace the search block.
    - Make sure the filenames in `routers` and `replacements` **exactly match** the originals provided.

    ### Example Output:
    routers:
      - 0_as65005_0.cfg
      - 1_as65005_1.cfg
    metadata:
      problem_diagnosis: "Interface mask mismatch on Router 0 caused OSPF adjacency failure."
      proposed_fix: "Correct the subnet mask to /30 and enable OSPF on the interface."
    replacements:
      0_as65005_0.cfg:
        - search: |
            interface GigabitEthernet0/0
             ip address 10.0.0.1 255.255.255.0
          replace: |
            interface GigabitEthernet0/0
             ip address 10.0.0.1 255.255.255.252
             ip ospf 1 area 0
      1_as65005_1.cfg:
        - search: |
            hostname OldName
          replace: |
            hostname NewName

    ### Search-and-Replace Rules:
    - Provide "search" blocks that appear **exactly once** in the faulty config. Include enough surrounding lines to ensure uniqueness.
    - Keep indentation, spacing, punctuation, and capitalization **identical** to the faulty configuration in the search block.
    - Do not include line numbers, diff headers, or leading +/- markers—only the raw text blocks.
    - Order operations from top to bottom as they should be applied; avoid overlapping replacements.
    - The YAML must keep the top-level ordering: routers -> metadata -> replacements.
    - Start the YAML with `routers:` and return YAML only (no markdown fences).
    
    ### Specification Semantics and Column Definitions
    The specifications provided below describe the difference between the *intended* network behavior (the Spec) and the *current* faulty behavior. They are formatted as CSV lines with the following columns:
    `Type, Source_Node, Destination_Prefix, waypoint_node, num_routes (for load_balancing), Status`

    **1. The "Status" Column (Critical):**
    - **broken_removed**: This behavior IS required but is CURRENTLY MISSING. You must **restore** this behavior.
    - **broken_added**: This behavior IS NOT required but is CURRENTLY PRESENT. You must **remove** this behavior.

    **2. The Invariant Types:**
    - **reachability**: `Source_Node` must be able to reach `Destination_Prefix`.
    - *Example:* `reachability,NodeA,10.0.0.1/32,,,broken_removed` → NodeA cannot reach 10.0.0.1, but it should. Fix connectivity.
    - **waypointing**: Traffic from `Source_Node` to `Destination_Prefix` must pass through the firewall/middlebox specified in `waypoint_node`.
    - *Example:* `waypointing,NodeA,10.0.0.1/32,FirewallB,,broken_removed` → Traffic is bypassing FirewallB. Force traffic through FirewallB.
    - **isolation**: `Source_Node` must NOT be able to reach `Destination_Prefix`.
    - *Example:* `isolation,NodeA,10.0.0.1/32,,,broken_removed` → NodeA *can* reach 10.0.0.1 (violation). Block this traffic. *(Note: isolation can be represented as negative reachability).*
    - **load_balancing**: Traffic from `Source_Node` to `Destination_Prefix` must be split across multiple paths (ECMP).
    - *Example:* `load_balancing,NodeA,10.0.0.1/32,,2,broken_removed` → Traffic is not being load-balanced as required. Restore Load Balancing across 2 paths.
    
    - **Network specifications/predicates**, after the fault was introduced:
    {preds_text}

    - **Current network topology**:
    {topology_text}
    
    - **Router configurations** to analyze:
    """)

    # Append configs
    config_section = ""
    for filename, config in final_configs.items():
        config_section += f"\n--- START OF {filename} ---\n{config}\n--- END OF {filename} ---\n"

    # Add optional additional context sections
    additional_context_section = ""
    
    if route_diffs:
        additional_context_section += f"""
    
### Additional Diagnostic Information

The following sections provide supplementary information about how the network state changed due to the fault.
Use this information to help diagnose the root cause and identify which configurations need to be modified.

{route_diffs}
"""

    if forwarding_diffs:
        if not route_diffs:
            additional_context_section += """
    
### Additional Diagnostic Information

The following sections provide supplementary information about how the network state changed due to the fault.
Use this information to help diagnose the root cause and identify which configurations need to be modified.

"""
        additional_context_section += f"""
{forwarding_diffs}
"""

    final_prompt = prompt_template + config_section + additional_context_section + "\nGenerate the solution now:\n\n"

    return final_prompt


# =========================================================================== #
#                          Feedback Prompt Generation                         #
# =========================================================================== #
def _feedback_prompt(
    attempt: int,
    fixed_specs: Dict[str, Dict[str, str]],
    fix_evaluation: Dict[str, Any],
    fix_metadata: Dict[str, Any],
) -> str:
    """
    Build feedback prompt for the LLM with all the relevant information.
    
    Args:
        attempt (int): Number of attempts in recursive chat.
        fixed_specs (Dict[str, Dict[str, str]]): Fixed specifications with nested structure.
        fix_evaluation (Dict[str, Any]): Evaluation report of previous attempt.
        fix_metadata (Dict[str, Any]): Fix metadata of previous attempt.
        
    Returns:
        Complete feedback prompt for model.
    """
    # Extract relevant data from specifications
    fixed_policies = fixed_specs.get("cleaned_policies.csv", {}) \
    .get("content", "Faulty policy information not available.")

    # Extract relevant info from evaluation report
    similarity_score = fix_evaluation.get("similarity_score", 0.0)
    missing_policies = fix_evaluation.get("missing_policies", [])
    unexpected_policies = fix_evaluation.get("unexpected_policies", [])

    # Build feedback prompt with sections
    feedback_prompt = f""" \
    Your previous attempt (#{attempt}) to fix the network configuration achieved a similarity score of {similarity_score:.4f},
    but there are still certain differences between the specifications derived from
    original configurations and the fixed configurations that you provided in your last response.

    Consider this constructive feedback to recall our previous messages from this discussion while repairing the incorrect components.
    Also once again recall the problem description from your memory.

    Specification policies derived from almost-fixed configurations that you provided earlier in the ongoing discussion:
    {fixed_policies}

    Part of evaluation report identifying potential discrepancies between specification policies derived
    from original configurations and the fixed configurations that you provided in your last response:
      1. Policies that are still missing (should be present):
      {missing_policies if missing_policies else "None - good job on restoring all required policies!"}
    
      2. Unexpected policies that should not be present:
      {unexpected_policies if unexpected_policies else "None - good job on removing all unexpected policies!"}

    Metadata of your one previous attempt at fixing the faulty configurations:
    {fix_metadata} 

    ### REVISED TASK:
    Based on this feedback, please provide an improved set of search-and-replace instructions.
    Focus specifically on addressing the missing and unexpected policies identified above.
    
    ### IMPORTANT REMINDERS:
    1. Strictly follow the same YAML format requirements and ordering: start with ``routers`` (list of routers needing changes), then ``metadata`` (problem_diagnosis/proposed_fix), then ``replacements`` (router keys mapped to ordered lists of ``search``/``replace`` operations).
    2. Use verbatim search blocks that exist in the faulty configuration; do not use regex, placeholders, or ellipses.
    3. Keep indentation and command ordering intact when providing replacement blocks.
    4. Your ``replacements`` routers must align with the ``routers`` list; use ``[]`` for routers that need no changes if already referenced.
    5. Include detailed metadata with your updated diagnosis and fix approach.
    
    Your response should start directly with 'routers:' and contain only valid YAML.
    """

    return feedback_prompt


# =========================================================================== #
#                         Format Reinforcement Prompt                         #
# =========================================================================== #
def _formatter_prompt(
    model_name: str,
) -> str:
    """
    Build format reinforcement prompt to help model reformat its response.
    
    Args:
        model_name (str): Name of selected model.
        
    Returns:
        Format reinforcement prompt for model.
    """
    formatter_prompt = """
    Your previous response contained the correct analysis and solution, but it was not formatted properly according to the YAML requirements. 

    Please reformat your previous response using the exact YAML structure required for search-and-replace instructions:

    YAML FORMAT REQUIREMENTS (ORDER MATTERS):
    1. Start with 'routers:' listing only the router filenames that require changes (or [] if none).
    2. Then 'metadata:' with block scalars 'problem_diagnosis' and 'proposed_fix'.
    3. Then 'replacements:' mapping each router filename to an ordered list of mappings with 'search' and 'replace' keys.
       - If a router needs no changes, set its value to an empty list [].
       - Use a pipe symbol followed by a SPACE: "| " for multi-line search/replace blocks.
       - Maintain consistent indentation for every line in each block scalar and preserve the relative indentation of commands.
    4. The YAML must be syntactically valid and properly structured with consistent indentation.
    5. Do NOT include markdown code fences, diff markers, or leading '+'/'-' signs.

    Example structure:
      routers:
        - Router1.cfg
        - Router2.cfg
      metadata:
        problem_diagnosis: |
          [Your diagnosis from previous response]
        proposed_fix: |
          [Your proposed fix from previous response]
      replacements:
        Router1.cfg:
          - search: |
              !
              hostname Router1
            replace: |
              !
              hostname FixedRouter1
        Router2.cfg: []

    IMPORTANT:
    1. Keep the same technical analysis and solution from your previous response
    2. Only reformat it according to the YAML structure above
    3. Do NOT change the actual configuration content or diagnosis
    4. Return only the YAML output with no markdown code blocks
    5. Your response should start directly with 'routers:' and contain only valid YAML

    Please reformat your previous analysis into this exact YAML structure.
    """
    
    return formatter_prompt


# =========================================================================== #
#                            Agent Tool Definitions                           #
# =========================================================================== #
AGENT_TOOL_DEFINITIONS = [
    {
        "name": "list_routers",
        "description": (
            "List all router configuration filenames available in the network."
        ),
        "parameters": {},
        "returns": "A list of router filenames.",
    },
    {
        "name": "inspect_config",
        "description": (
            "Retrieve the full configuration text of a specific router. "
            "Use this to examine a router before deciding what to fix."
        ),
        "parameters": {
            "router_name": "Filename of the router config (e.g. '0_as65003_0.cfg').",
        },
        "returns": "The full configuration text of the specified router.",
    },
    {
        "name": "get_violated_specs",
        "description": (
            "Get the list of currently violated network specifications. "
            "These describe the gap between intended and actual behavior."
        ),
        "parameters": {},
        "returns": "Violated specifications with type, source, destination, and status.",
    },
    {
        "name": "get_topology",
        "description": "Get the network topology (nodes and links).",
        "parameters": {},
        "returns": "The network topology as a JSON structure.",
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a search-and-replace edit to a router configuration. "
            "The search block must appear exactly once in the config."
        ),
        "parameters": {
            "router_name": "Filename of the router config to edit.",
            "search": "Exact text block to find (must be unique).",
            "replace": "Replacement text block.",
        },
        "returns": "Success or failure message with details.",
    },
    {
        "name": "verify",
        "description": (
            "Run formal data-plane verification on the current network state. "
            "Returns which specs are fixed, which remain broken, and any regressions."
        ),
        "parameters": {},
        "returns": (
            "Verification report with fix_score, regression_rate, "
            "and per-predicate classification."
        ),
    },
    {
        "name": "rollback",
        "description": (
            "Undo ALL patches applied since the last successful verification. "
            "Restores configs to the last verified-safe state."
        ),
        "parameters": {},
        "returns": "Confirmation that configs have been rolled back.",
    },
    {
        "name": "submit",
        "description": (
            "Submit the current configuration as the final solution. "
            "Call when satisfied with verification results."
        ),
        "parameters": {},
        "returns": "Confirmation that the solution has been submitted.",
    },
]


def _format_agent_tool_descriptions() -> str:
    """
    Format tool definitions into a human-readable block for the agent system prompt.

    Returns:
        Formatted string describing all available tools.
    """
    lines = []
    for tool in AGENT_TOOL_DEFINITIONS:
        lines.append(f"### {tool['name']}")
        lines.append(f"  Description: {tool['description']}")
        if tool["parameters"]:
            lines.append("  Parameters:")
            for pname, pdesc in tool["parameters"].items():
                lines.append(f"    - {pname}: {pdesc}")
        else:
            lines.append("  Parameters: None")
        lines.append(f"  Returns: {tool['returns']}")
        lines.append("")
    return "\n".join(lines)


# =========================================================================== #
#                            Agent System Prompt                              #
# =========================================================================== #
_AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert network engineer agent. Your task is to diagnose and repair \
network misconfigurations by iteratively inspecting configurations, applying \
targeted fixes, and verifying correctness using formal data-plane analysis.

## Available Tools

You interact with the network environment through tool calls. Each turn, you \
should output your reasoning (Thought), then a single tool call (Action).

{tool_descriptions}

## Tool Call Format

These tools are NOT native function calls. You must invoke them using
plain text in the EXACT format shown below. Do not use any other
tool-calling syntax, function-calling format, or internal protocol.
Simply output the following three lines as plain text:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with parameters, or {{}} for no-parameter tools>

Examples:
  Thought: I need to see which routers are in the network.
  Action: list_routers
  Action Input: {{}}

  Thought: Let me inspect the BGP configuration of router 0_as65003_0.cfg.
  Action: inspect_config
  Action Input: {{"router_name": "0_as65003_0.cfg"}}

  Thought: I found the misconfigured subnet mask. Let me fix it.
  Action: apply_patch
  Action Input: {{"router_name": "0_as65003_0.cfg", "search": " ip address 10.0.0.1 255.255.255.0", "replace": " ip address 10.0.0.1 255.255.255.252"}}

  Thought: I've applied my patches. Let me verify the network state.
  Action: verify
  Action Input: {{}}

  Thought: All specs are fixed with no regressions. I'll submit.
  Action: submit
  Action Input: {{}}

## Strategy

1. Start by examining the violated specifications to understand what is broken.
2. Inspect relevant router configurations to diagnose root causes.
3. Apply targeted patches to fix the identified issues.
4. Run verification to check if specs are restored and no regressions occurred.
5. If issues remain, analyze the verification feedback and iterate.
6. If a patch introduces regressions, rollback and try a different approach.
7. Submit when satisfied or when you've exhausted reasonable repair attempts.

## Important Rules

- Be precise with search blocks: they must match the config EXACTLY \
(whitespace, indentation, etc.).
- Prefer small, targeted patches over large rewrites.
- Always verify after applying patches before submitting.
- If verification shows regressions, consider rolling back.
- You have a limited budget of {max_steps} tool calls. Use them wisely.
"""


def _agent_system_prompt(max_steps: int = 30) -> str:
    """
    Build the full agent system prompt with tool descriptions.

    Args:
        max_steps (int): Maximum tool-call steps allowed.
            Defaults to 30.

    Returns:
        Formatted system prompt string.
    """
    return _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=_format_agent_tool_descriptions(),
        max_steps=max_steps,
    )


# =========================================================================== #
#                             Agent Task Messages                             #
# =========================================================================== #
AGENT_INITIAL_TASK_MESSAGE = (
    "A network fault has been detected. Please diagnose and repair "
    "the misconfiguration.\n\n"
    "Start by examining the violated specifications to understand what "
    "is broken, then inspect the relevant router configurations, apply "
    "fixes, and verify.\n\n"
    "You have a budget of {max_steps} tool calls for this task.\n\n"
    "Begin now."
)

AGENT_NUDGE_FORMAT_MESSAGE = (
    "I couldn't parse a valid tool call from your response. "
    "Please respond with the exact format:\n"
    "Thought: <reasoning>\n"
    "Action: <tool_name>\n"
    "Action Input: <json params>"
)

AGENT_CTX_ABLATION_MESSAGE = (
    "All network context has been provided above. "
    "The retrieval tools (list_routers, inspect_config, "
    "get_violated_specs, get_topology) are disabled — "
    "use the information above instead.\n\n"
    "Proceed directly to diagnosing the issue and applying "
    "patches using the Thought/Action/Action Input format.\n\n"
    "You have a budget of {max_steps} tool calls for this task.\n\n"
    "Begin now."
)