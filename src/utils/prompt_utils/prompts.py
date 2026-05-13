"""
Utility script to generate and manage prompts for network troubleshooting tasks.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import json
from typing import Dict, Any, List, Optional
        

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
        Complete initial prompt for model.
    """
    
    def _format_preds(preds_value: Any) -> str:
      if isinstance(preds_value, (list, tuple)):
        return "\n".join(str(p) for p in preds_value)
      return str(preds_value)

    def _format_topology(topology_value: Any) -> str:
      if isinstance(topology_value, str):
        return topology_value
      try:
        return json.dumps(topology_value, separators=(",", ":"), ensure_ascii=False)
      except TypeError:
        return str(topology_value)

    preds_text = _format_preds(preds)
    topology_text = _format_topology(topology)

    # Build initial prompt with sections
    prompt = f""" \
    A network fault has occurred, possibly due to the human error, causing discrepancies in at least one of the configuration files. 
    The specifications/predicates that govern the network behavior have changed as a result. These specifications in 
    question are automatically derived from the router configurations and they describe the resulting forwarding behavior.

    ### Task Requirements:
    1. Your task is to identify which routers must be modified to restore the correct forwarding behavior. 
    2. In this regard, you are expected to generate the necessary modifications for each router configuration. 
    3. Provide your solution in the **unified diff (``unidiff'') format for each **misconfigured router** only.
    4. Think of your output as a result of the standard ``diff'' shell command, which can be used as input to the ``patch'' program.

    ### Expected Output Format:
    - For easy parsing, return the full solution in **YAML only** template, with no markdown or code fences.
    - The top-level key must be ``patches:''.
    - Under ``patches:'', list one **key** entry per router (by the corresponding filename) that needs modifications.
    - Each entry value must be a block scalar (``|'') containing the required **diff patch**.
    - Also return **another top-level key ``metadata:''** section with short text:
      - ``problem_diagnosis'': What root issue caused the fault.
      - ``proposed_fix'': The intent behind the patch.
    - Preserve indentation and spacing exactly; no reformatting or cosmetic edits.
    - Make sure the filenames contained in your final output **always exactly match** with the original ones **with a proper ``.cfg'' extension**.
    - Example (return exactly this YAML envelope, no markdown fences):
      patches:
        0_as65005_0.cfg: |
          --- 0_as65005_0.cfg
          +++ 0_as65005_0.cfg
          @@ -5,3 +5,4 @@
           interface GigabitEthernet0/0
          - ip address 10.0.0.1 255.255.255.0
          + ip address 10.0.0.1 255.255.255.252
          + ip ospf 1 area 0
        1_as65005_1.cfg: |
          --- 1_as65005_1.cfg
          +++ 1_as65005_1.cfg
          @@ -1,3 +1,3 @@
          -hostname OldName
          +hostname NewName
           !
      metadata:
        problem_diagnosis: concise root cause
        proposed_fix: concise intent

    ### Unidiff Patch Rules:
    - Include minimal context (1-3 unchanged lines before/after changes), as in output of standard ``diff -u'' command.
    - Start with ``--- <filename>'' header, then ``+++ <filename>'' header.
    - Then each change hunk must continue with "@@ -start,count +start,count @@" header.
    - Follow with one or more change hunks showing line differences:
      - Deletion lines: preceded by minus sign ``-'', as diff marker, not YAML syntax.
      - Addition lines: preceded by plus sign ``+'', as diff marker, not YAML syntax.
      - Unchanged context lines: preceded by space character `` ''.

    ### Network-wide Details:
    - **Network specifications/predicates**, after the fault was introduced:
      {preds_text}

    - **Current network topology**:
      {topology_text}
      
    - **Router configurations** to analyze:
    """

    # Add each configuration file to the prompt
    for filename, config in final_configs.items():
        prompt += f"\n--- {filename} ---\n{config}\n"

    # Add optional additional context sections
    additional_context = ""
    
    if route_diffs:
        additional_context += f"""
    
    ### Additional Diagnostic Information
    
    The following sections provide supplementary information about how the network state changed due to the fault.
    Use this information to help diagnose the root cause and identify which configurations need to be modified.
    
    {route_diffs}
    """

    if forwarding_diffs:
        if not route_diffs:
            additional_context += """
    
    ### Additional Diagnostic Information
    
    The following sections provide supplementary information about how the network state changed due to the fault.
    Use this information to help diagnose the root cause and identify which configurations need to be modified.
    
    """
        additional_context += f"""
    {forwarding_diffs}
    """

    prompt += additional_context + "\nGenerate the solution now:"

    return prompt


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
    Based on this feedback, please provide an improved solution.
    Focus specifically on addressing the missing and unexpected policies identified above.
    
    ### IMPORTANT REMINDERS:
    1. Strictly follow the same YAML format requirements. YAML output must be syntactically correct with no markdown code blocks.
    2. Strictly follow the **example structure** shown in your initial instructions when returning your answer in YAML format 
    3. For each router that requires changes, provide the COMPLETE configuration (not just the changed parts).
    4. For routers that do not need changes, use "No change needed".
    5. You must validate your output contains all routers you have already seen.
    6. Include detailed metadata with your updated diagnosis and fix approach.
    
    Your response should start directly with 'configs:' and contain only valid YAML.
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

    Please reformat your previous response using the exact YAML structure required:

    YAML FORMAT REQUIREMENTS:
    1. Start your response with 'configs:' on the first line
    2. For router configurations, use YAML block scalar syntax with consistent indentation:
      - Use a pipe symbol followed by a SPACE: "| " 
      - Choose ONE consistent level of indentation for ALL lines in the configuration
      - Maintain the same indentation for every line, including lines with "!"
      - Preserve the relative indentation of commands within the configuration
    3. The YAML must be syntactically valid and properly structured
    4. For routers that don't need changes, write: "RouterName.cfg: No change needed"

    Example structure:
    ```yaml
    configs:
      Router1.cfg: |
        !
        hostname Router1
        !
        interface FastEthernet0/0
        ip address 192.168.1.1 255.255.255.0
        no shutdown
        !
        router ospf 1
        network 192.168.1.0 0.0.0.255 area 0
        !
        end
      Router2.cfg: No change needed
    metadata:
      problem_diagnosis: |
        [Your diagnosis from previous response]
      proposed_fix: |
        [Your proposed fix from previous response]
    ```

    IMPORTANT:
    1. Keep the same technical analysis and solution from your previous response
    2. Only reformat it according to the YAML structure above
    3. Do NOT change the actual configuration content or diagnosis
    4. Return only the YAML output with no markdown code blocks
    5. Your response should start directly with 'configs:' and contain only valid YAML

    Please reformat your previous analysis into this exact YAML structure.
    """
    
    return formatter_prompt
