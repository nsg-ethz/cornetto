"""
Verification-guided agentic pipeline for network configuration repair.

Implements a ReAct-style agent that iteratively diagnoses and repairs
network misconfigurations using formal data-plane verification (Batfish)
as its feedback oracle. The agent has access to tools for inspecting
configurations, applying patches, running verification, and rolling
back unsafe changes.

All evaluation metrics (fix score, regression rate, router F1, diagnosis
judge) are preserved for direct comparison with baseline runs produced
by the ZeroShot pipeline.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.modules.ZeroShot import ZeroShot as BaseZeroShot
from src.utils.parser_utils.parsers import agent_response_parser
from src.utils.prompt_utils.prompts_alt_patch import (
    _agent_system_prompt,
    AGENT_INITIAL_TASK_MESSAGE,
    AGENT_NUDGE_FORMAT_MESSAGE,
    AGENT_TOOL_DEFINITIONS,
    AGENT_CTX_ABLATION_MESSAGE
)
from src.utils.token_counter import token_counter

logger = logging.getLogger(__name__)


# =========================================================================== #
#                               Tool Environment                              #
# =========================================================================== #
class ToolEnvironment:
    """
    Mutable network state that the agent interacts with via tool calls.

    Holds current configuration files, applies patches, runs Batfish
    verification, and tracks history for rollback and analysis.
    """

    # Sentinel returned by the submit tool so the agent loop can detect it
    SUBMIT_SIGNAL = "SUBMIT_SIGNAL"

    def __init__(
        self,
        faulty_configs: Dict[str, str],
        original_configs: Dict[str, str],
        full_faulty_configs: Dict[str, str],
        topology: Any,
        violated_specs: List[str],
        specification_csv_path: Optional[str],
        original_specs: Optional[List[str]],
        net_env: Optional[Callable],
        evaluator: Callable,
        evaluator_kwargs: Dict[str, Any],
        context_mode: str = "random",
        verification_mode: str = "per_step",
        context_prefill: bool = False,
        prefilled_instruction: Optional[str] = None, 
    ):
        """
        Initialize the tool environment.

        Args:
            faulty_configs (Dict[str, str]): Broken router configs the agent may inspect/edit.
            original_configs (Dict[str, str]): Golden configs (for evaluation).
            full_faulty_configs (Dict[str, str]): Complete faulty snapshot
                (always full; used when saving to Batfish).
            topology (Any): Network topology data.
            violated_specs (List[str]): Violated specification strings.
            specification_csv_path (Optional[str]): Path to full spec CSV.
            original_specs (Optional[List[str]]): Spec strings from golden state.
            net_env (Optional[Callable]): Batfish network environment callable.
            evaluator (Callable): Specification evaluation function.
            evaluator_kwargs (Dict[str, Any]): Base kwargs for the evaluator.
            context_mode (str): One of 'oracle', 'full', or 'random'.
                Defaults to 'full'.
            verification_mode (str): One of 'per_step' or 'post_hoc'.
                Defaults to 'per_step'.
            context_prefill (bool): Flag to include the entire context in prompt.
            prefilled_instruction (str): Original input prompt.
        """
        # Immutable references
        self.original_configs = original_configs
        self.full_faulty_configs = full_faulty_configs
        self.topology = topology
        self.violated_specs = violated_specs
        self.specification_csv_path = specification_csv_path
        self.original_specs = original_specs
        self.net_env = net_env
        self.evaluator = evaluator
        self.evaluator_kwargs = evaluator_kwargs
        self.context_mode = context_mode
        self.verification_mode = verification_mode
        self.context_prefill = context_prefill
        self.prefilled_instruction = prefilled_instruction

        # Mutable state: configs the agent can see and edit
        self.current_configs: Dict[str, str] = copy.deepcopy(faulty_configs)

        # Checkpoint: last verified-safe state
        self.checkpoint_configs: Dict[str, str] = copy.deepcopy(faulty_configs)

        # Tracking
        self.inspected_routers: Set[str] = set()
        self.pending_patches: List[Dict[str, Any]] = []
        self.verification_history: List[Dict[str, Any]] = []
        self.tool_call_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #                           Tool Dispatcher                          #
    # ------------------------------------------------------------------ #
    def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> str:
        """
        Dispatch a tool call and return the observation string.

        Args:
            tool_name (str): Name of the tool to execute.
            params (Dict[str, Any]): Parameters for the tool.

        Returns:
            Observation string to feed back to the agent.
        """
        start_time = time.time()

        handler = {
            "list_routers": lambda: self._tool_list_routers(),
            "inspect_config": lambda: self._tool_inspect_config(
                params.get("router_name", "")
            ),
            "get_violated_specs": lambda: self._tool_get_violated_specs(),
            "get_topology": lambda: self._tool_get_topology(),
            "apply_patch": lambda: self._tool_apply_patch(
                params.get("router_name", ""),
                params.get("search", ""),
                params.get("replace", ""),
            ),
            "verify": lambda: self._tool_verify(),
            "rollback": lambda: self._tool_rollback(),
            "submit": lambda: self._tool_submit(),
        }.get(tool_name)

        try:
            if handler is None:
                names = [t["name"] for t in AGENT_TOOL_DEFINITIONS]
                result = (
                    f"Error: Unknown tool '{tool_name}'. Available: {names}"
                )
            else:
                result = handler()
        except Exception as e:
            result = f"Error executing {tool_name}: {str(e)}"

        elapsed = time.time() - start_time
        self.tool_call_log.append({
            "tool": tool_name,
            "params": params,
            "result_length": len(result),
            "elapsed": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
        return result

    # ------------------------------------------------------------------ #
    #                      Individual Tool Handlers                      #
    # ------------------------------------------------------------------ #
    def _tool_list_routers(self) -> str:
        """
        List router filenames visible to the agent under the active context mode.
 
        Returns:
            Formatted string enumerating available router config files.
        """
        if self.context_prefill:
            return "This information was already provided in the task description above."

        routers = sorted(self.current_configs.keys())
        return (
            f"Available routers ({len(routers)} total):\n"
            + "\n".join(f"  - {r}" for r in routers)
        )
 
    def _tool_inspect_config(self, router_name: str) -> str:
        """
        Return the full configuration text for a single router.
 
        Args:
            router_name (str): Filename of the router config to retrieve.
 
        Returns:
            Config text wrapped in delimiters, or an error message.
        """
        if self.context_prefill:
            return "All configurations were already provided in the task description above."

        if router_name not in self.current_configs:
            available = sorted(self.current_configs.keys())[:10]
            suffix = "..." if len(self.current_configs) > 10 else ""
            return (
                f"Error: Router '{router_name}' not found. "
                f"Available: {available}{suffix}"
            )
        self.inspected_routers.add(router_name)
        config = self.current_configs[router_name]
        return (
            f"--- Configuration of {router_name} ---\n"
            f"{config}\n"
            f"--- End of {router_name} ---"
        )
 
    def _tool_get_violated_specs(self) -> str:
        """
        Return the violated specification list with semantic guidance.
 
        Returns:
            Formatted string with CSV-style spec lines and status legend.
        """
        if self.context_prefill:
            return "Violated specifications were already provided in the task description above."

        if not self.violated_specs:
            return "No violated specifications found."
        header = (
            "Violated specifications "
            "(CSV: type,source_node,dest_prefix,waypoint,num_routes,status):\n"
            "  - 'broken_removed': Required but MISSING. Restore it.\n"
            "  - 'broken_added': Not required but PRESENT. Remove it.\n\n"
        )
        specs_text = "\n".join(f"  {s}" for s in self.violated_specs)
        return header + specs_text
 
    def _tool_get_topology(self) -> str:
        """
        Return the network topology data as a string.
 
        Returns:
            JSON-serialised topology or its string representation.
        """
        if self.context_prefill:
            return "Network topology was already provided in the task description above."

        if isinstance(self.topology, dict):
            return json.dumps(
                self.topology, separators=(",", ":"), ensure_ascii=False
            )
        return str(self.topology)
 
    def _tool_apply_patch(
        self,
        router_name: str,
        search: str,
        replace: str,
    ) -> str:
        """
        Apply a single search-and-replace edit to a router config.
        Tries exact match first, then whitespace-normalised fallback.
        """
        if router_name not in self.current_configs:
            return f"Error: Router '{router_name}' not found."
        if not search.strip():
            return "Error: Search block is empty."
 
        config = self.current_configs[router_name]
 
        # Exact match
        count = config.count(search)
        if count == 1:
            self.current_configs[router_name] = config.replace(
                search, replace, 1
            )
            self.pending_patches.append({
                "router": router_name,
                "search": search,
                "replace": replace,
                "match_strategy": "exact",
            })
            return (
                f"Patch applied to {router_name}. "
                f"{len(self.pending_patches)} pending patch(es) "
                f"since last checkpoint."
            )
        if count > 1:
            return (
                f"Error: Search block found {count} times in {router_name}. "
                f"Include more surrounding context to make it unique."
            )
 
        # Whitespace-normalised fallback
        def _normalise(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.split("\n"))
 
        norm_config = _normalise(config)
        norm_search = _normalise(search)
        if norm_config.count(norm_search) == 1:
            self.current_configs[router_name] = norm_config.replace(
                norm_search, replace, 1
            )
            self.pending_patches.append({
                "router": router_name,
                "search": search,
                "replace": replace,
                "match_strategy": "whitespace_normalised",
            })
            return (
                f"Patch applied to {router_name} "
                f"(whitespace-normalised match). "
                f"{len(self.pending_patches)} pending patch(es) "
                f"since last checkpoint."
            )
 
        return (
            f"Error: Search block not found in {router_name}. "
            f"Ensure the text matches EXACTLY (including whitespace). "
            f"Use inspect_config to view the current state."
        )
 
    def _tool_verify(self) -> str:
        """
        Run formal data-plane verification via Batfish on the current state.
 
        In ``per_step`` mode: runs Batfish, evaluates against ground-truth
        specs, and returns structured feedback (fixed / unfixed / regressed).
        Updates the checkpoint if no regressions are detected.
 
        In ``post_hoc`` mode: returns a deferred message; the agent must
        commit without intermediate verification feedback.
 
        Returns:
            Structured verification observation or deferral message.
        """
        if not self.net_env:
            return "Error: No network environment available for verification."
 
        # In post_hoc mode the agent does not get intermediate feedback
        if self.verification_mode == "post_hoc":
            return (
                "Verification is deferred to submission in this mode. "
                "Continue diagnosing and patching, then call submit "
                "when you are confident in your repairs."
            )
 
        # Save merged configs to temp dir and run Batfish
        try:
            tmpdir = self._save_configs_to_tempdir(self.current_configs)
            processed_results = self.net_env(local_scenario_path=tmpdir.name)
        except Exception as e:
            return f"Error running verification: {str(e)}"
 
        if processed_results is None:
            return (
                "Error: Verification returned no results "
                "(Batfish may have failed)."
            )
 
        # Evaluate against ground-truth specifications
        try:
            eval_kwargs = self.evaluator_kwargs.copy()
            eval_kwargs["specification_csv_path"] = self.specification_csv_path
            eval_kwargs["reference_spec"] = (
                None if self.specification_csv_path else self.original_specs
            )
            eval_kwargs["compared_spec"] = processed_results
            evaluation = self.evaluator(**eval_kwargs)
        except Exception as e:
            return f"Error during evaluation: {str(e)}"
 
        summary = evaluation.get("summary", {})
        results = evaluation.get("results", {})
 
        fixed_count = summary.get("fixed", 0)
        not_fixed_count = summary.get("not_fixed", 0)
        side_effects_count = summary.get(
            "side_effects", summary.get("broken", 0)
        )
        fix_rate = summary.get("fix_rate", 0.0)
        total = fixed_count + not_fixed_count + side_effects_count
        regression_rate = side_effects_count / total if total > 0 else 0.0
 
        # Store verification result
        entry = {
            "step": len(self.verification_history) + 1,
            "fixed": fixed_count,
            "not_fixed": not_fixed_count,
            "side_effects": side_effects_count,
            "fix_rate": fix_rate,
            "regression_rate": regression_rate,
            "patches_since_checkpoint": len(self.pending_patches),
            "evaluation": evaluation,
            "processed_results": processed_results,
        }
        self.verification_history.append(entry)
 
        # Build structured observation
        obs = self._format_verification_observation(entry, results)
 
        # Update checkpoint if safe
        if side_effects_count == 0:
            self.checkpoint_configs = copy.deepcopy(self.current_configs)
            self.pending_patches = []
            obs += "\nCheckpoint updated (no regressions detected)."
 
        if not_fixed_count == 0 and side_effects_count == 0:
            obs += (
                "\n*** All specifications satisfied with zero regressions! "
                "You may submit. ***"
            )
 
        return obs
 
    def _tool_rollback(self) -> str:
        """
        Undo all patches applied since the last checkpoint.
 
        Restores ``current_configs`` to the last verified-safe state
        and clears the pending patch list.
 
        Returns:
            Confirmation message with the number of rolled-back patches.
        """
        if not self.pending_patches:
            return (
                "Nothing to rollback; no patches pending since last checkpoint."
            )
        rolled_back = len(self.pending_patches)
        self.current_configs = copy.deepcopy(self.checkpoint_configs)
        self.pending_patches = []
        return (
            f"Rolled back {rolled_back} patch(es). "
            f"Configs restored to last checkpoint state."
        )
 
    def _tool_submit(self) -> str:
        """
        Signal the agent loop to terminate with the current configs.
 
        Returns:
            The ``SUBMIT_SIGNAL`` sentinel string.
        """
        return self.SUBMIT_SIGNAL

    # ------------------------------------------------------------------ #
    #                          Public Accessors                          #
    # ------------------------------------------------------------------ #
    def get_final_configs(self) -> Dict[str, str]:
        """
        Return the current state of all configurations.

        Returns:
            Deep copy of the agent's working configuration set.
        """
        return copy.deepcopy(self.current_configs)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the agent's interaction with the environment.

        Returns:
            Dictionary with tool-call counts, inspected routers, and
            the verification trajectory.
        """
        return {
            "total_tool_calls": len(self.tool_call_log),
            "routers_inspected": sorted(self.inspected_routers),
            "total_patches_attempted": sum(
                1 for log in self.tool_call_log if log["tool"] == "apply_patch"
            ),
            "total_verifications": len(self.verification_history),
            "total_rollbacks": sum(
                1 for log in self.tool_call_log if log["tool"] == "rollback"
            ),
            "verification_trajectory": [
                {
                    "step": v["step"],
                    "fix_rate": v["fix_rate"],
                    "regression_rate": v["regression_rate"],
                    "fixed": v["fixed"],
                    "not_fixed": v["not_fixed"],
                    "side_effects": v["side_effects"],
                }
                for v in self.verification_history
            ],
            "tool_call_log": self.tool_call_log,
        }

    # ------------------------------------------------------------------ #
    #                           Private Helpers                          #
    # ------------------------------------------------------------------ #
    def _save_configs_to_tempdir(
        self,
        configs: Dict[str, str],
    ):
        """
        Merge edited configs into the full network and save to a temp dir.

        Always starts from the complete faulty snapshot so Batfish
        analyses the entire topology, then overlays the agent's edits.

        Args:
            configs (Dict[str, str]): Agent's current working configs.

        Returns:
            Temporary directory with merged config files.
        """
        import tempfile

        merged = copy.deepcopy(self.full_faulty_configs)
        merged.update(configs)

        tmpdir = tempfile.TemporaryDirectory()
        config_dir = os.path.join(tmpdir.name, "configs")
        os.makedirs(config_dir, exist_ok=True)

        for filename, content in merged.items():
            if not filename.endswith(".cfg"):
                filename = f"{filename}.cfg"
            with open(os.path.join(config_dir, filename), "w") as f:
                f.write(content)

        return tmpdir

    @staticmethod
    def _format_verification_observation(
        entry: Dict[str, Any],
        results: Dict[str, Any],
        max_unfixed: int = 20,
        max_regressions: int = 10,
    ) -> str:
        """
        Build the observation string the agent receives after verification.

        Args:
            entry (Dict[str, Any]): Single verification history entry.
            results (Dict[str, Any]): Classification from the evaluator.
            max_unfixed (int): Max unfixed specs to show.
                Defaults to 20.
            max_regressions (int): Max regressions to show.
                Defaults to 10.

        Returns:
            Human-readable verification observation.
        """
        lines = [
            f"=== Verification Results (Round {entry['step']}) ===",
            (
                f"Fix Score: {entry['fix_rate']:.4f} "
                f"({entry['fixed']} fixed out of "
                f"{entry['fixed'] + entry['not_fixed']} violations)"
            ),
            (
                f"Regressions: {entry['side_effects']} new violations "
                f"(regression rate: {entry['regression_rate']:.4f})"
            ),
            (
                f"Patches applied since last checkpoint: "
                f"{entry['patches_since_checkpoint']}"
            ),
        ]

        not_fixed_list = results.get("not_fixed", [])
        if not_fixed_list:
            lines.append(
                f"\nRemaining unfixed violations "
                f"({len(not_fixed_list)} total):"
            )
            for item in not_fixed_list[:max_unfixed]:
                pred = item.get("predicate", {})
                lines.append(
                    f"  - {pred.get('type', '?')}: "
                    f"node={pred.get('node', '?')}, "
                    f"prefix={pred.get('prefix', '?')}, "
                    f"status={item.get('initial_status', '?')}"
                )
            if len(not_fixed_list) > max_unfixed:
                lines.append(
                    f"  ... and {len(not_fixed_list) - max_unfixed} more"
                )

        side_effects_list = results.get("side_effects", [])
        if side_effects_list:
            lines.append(
                f"\nNew regressions ({len(side_effects_list)} total):"
            )
            for item in side_effects_list[:max_regressions]:
                pred = item.get("predicate", {})
                lines.append(
                    f"  - {pred.get('type', '?')}: "
                    f"node={pred.get('node', '?')}, "
                    f"prefix={pred.get('prefix', '?')}"
                )
            if len(side_effects_list) > max_regressions:
                lines.append(
                    f"  ... and "
                    f"{len(side_effects_list) - max_regressions} more"
                )

        return "\n".join(lines)


# =========================================================================== #
#                              Agent Loop Runner                              #
# =========================================================================== #
def _run_agent_loop(
    model: Any,
    environment: ToolEnvironment,
    system_prompt: str,
    max_steps: int = 30,
    timeout: int = 1200,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Execute the ReAct tool-calling loop.

    The agent receives the system prompt and an initial task message,
    then alternates between LLM generation and tool execution until it
    calls ``submit``, hits the step budget, or times out.

    Args:
        model (Any): LLM chat model (BaseChat instance).
        environment (ToolEnvironment): Tool execution environment.
        system_prompt (str): Formatted agent system prompt.
        max_steps (int): Maximum number of tool-call steps.
            Defaults to 30.
        timeout (int): Total wall-clock timeout in seconds.
            Defaults to 1200.

    Returns:
        Tuple of (final_configs, agent_metadata).
    """
    # Shoul we ablate on context or not
    if environment.context_prefill:
        initial_message = (
            environment.prefilled_instruction
            + AGENT_CTX_ABLATION_MESSAGE.format(max_steps=max_steps)
        )
    else:
        initial_message = AGENT_INITIAL_TASK_MESSAGE.format(max_steps=max_steps)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_message),
    ]    

    agent_metadata: Dict[str, Any] = {
        "steps": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "submitted": False,
        "terminated_reason": None,
    }

    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Timeout check
        if time.time() - start_time > timeout:
            agent_metadata["terminated_reason"] = "timeout"
            logger.warning(
                "Agent timed out after %ds at step %d", timeout, step
            )
            break

        # Truncate old observations to manage context window as in SWE-Agent
        query_messages = _truncate_old_observations(
            messages, keep_last_n=5
        )

        # Count input tokens (full conversation sent to the model)
        step_input_tokens = sum(
            token_counter(msg.content) for msg in query_messages
        )
        agent_metadata["total_prompt_tokens"] += step_input_tokens

        # Query the LLM
        response_text = _query_model(model, query_messages, step, agent_metadata)
        if response_text is None:
            # Record the failed call so we know what context size caused it
            agent_metadata["steps"].append({
                "step": step,
                "thought": "",
                "action": "LLM_ERROR",
                "params": {},
                "observation": agent_metadata.get("terminated_reason", "unknown error"),
                "step_input_tokens": step_input_tokens,
                "step_output_tokens": 0,
            })
            break

        step_output_tokens = token_counter(response_text)
        agent_metadata["total_completion_tokens"] += step_output_tokens

        # Parse thought / action / params
        thought, action, params = agent_response_parser(response_text)

        # Handle unparseable response
        if not action:
            messages.append(AIMessage(content=response_text))
            messages.append(HumanMessage(content=AGENT_NUDGE_FORMAT_MESSAGE))
            agent_metadata["steps"].append({
                "step": step,
                "thought": thought,
                "action": "PARSE_ERROR",
                "params": {},
                "observation": "Invalid format",
                "raw_response": response_text,
                "step_input_tokens": step_input_tokens,
                "step_output_tokens": step_output_tokens,
            })
            continue

        # Execute the tool
        observation = environment.execute_tool(action, params)

        # Check for submit signal
        if observation == ToolEnvironment.SUBMIT_SIGNAL:
            agent_metadata["submitted"] = True
            agent_metadata["terminated_reason"] = "submitted"
            agent_metadata["steps"].append({
                "step": step,
                "thought": thought,
                "action": action,
                "params": params,
                "observation": "Solution submitted.",
                "raw_response": response_text,
                "step_input_tokens": step_input_tokens,
                "step_output_tokens": step_output_tokens,
            })
            logger.info("Agent submitted solution at step %d", step)
            break

        # Record step
        agent_metadata["steps"].append({
            "step": step,
            "thought": thought,
            "action": action,
            "params": params,
            "observation": observation[:2000],
            "raw_response": response_text,
            "full_observation": observation,
            "step_input_tokens": step_input_tokens,
            "step_output_tokens": step_output_tokens,
        })

        logger.info(
            "Step %d: Action=%s, Thought=%s...",
            step, action, thought[:500],
        )

        # Append to conversation (truncate very long observations)
        messages.append(AIMessage(content=response_text))
        obs_msg = f"Observation: {observation}"
        if len(obs_msg) > 50_000:
            obs_msg = obs_msg[:50_000] + "\n... [truncated]"
        messages.append(HumanMessage(content=obs_msg))

    else:
        agent_metadata["terminated_reason"] = "max_steps_reached"
        logger.warning(
            "Agent exhausted %d steps without submitting", max_steps
        )

    # Finalise metadata
    agent_metadata["total_steps"] = len(agent_metadata["steps"])
    agent_metadata["total_time"] = time.time() - start_time
    agent_metadata["environment_summary"] = environment.get_summary()

    return environment.get_final_configs(), agent_metadata


def _query_model(
    model: Any,
    messages: list,
    step: int,
    agent_metadata: Dict[str, Any],
) -> Optional[str]:
    """
    Send conversation to the LLM and return the response text.

    On failure, records the error in *agent_metadata* and returns None.

    Args:
        model: LLM chat model.
        messages: Full conversation history.
        step: Current step number.
        agent_metadata: Mutable metadata dict to update on error.

    Returns:
        Response text, or None if the call failed.
    """
    try:
        if hasattr(model, "memory") and hasattr(model.memory, "chat_memory"):
            model.memory.chat_memory.clear()
            for msg in messages:
                model.memory.chat_memory.add_message(msg)

        response = model.invoke([messages[-1]])
        return (
            response.content if hasattr(response, "content")
            else str(response)
        )
    except Exception as e:
        logger.error("LLM call failed at step %d: %s", step, e)
        agent_metadata["terminated_reason"] = f"llm_error: {str(e)}"
        return None

def _truncate_old_observations(
    messages: List,
    keep_last_n: int = 5,
) -> List:
    """
    Replace old tool-output observations with a short placeholder,
    keeping the last *keep_last_n* observations intact.

    All Thought/Action messages (AIMessage) are preserved in full
    so the model retains its reasoning history.  Only the bulky
    HumanMessage observations (tool outputs) are truncated.

    Args:
        messages (List): Full conversation message list.
        keep_last_n (int): Number of recent observations to keep.
            Defaults to 5.

    Returns:
        New message list with old observations truncated.
    """
    # Find indices of all observation messages
    obs_indices = [
        i for i, msg in enumerate(messages)
        if isinstance(msg, HumanMessage)
        and msg.content.startswith("Observation:")
    ]

    # Nothing to truncate
    if len(obs_indices) <= keep_last_n:
        return messages

    # Indices to truncate (all but the last keep_last_n)
    truncate_set = set(obs_indices[:-keep_last_n])

    truncated = []
    for i, msg in enumerate(messages):
        if i in truncate_set:
            original_lines = msg.content.count("\n") + 1
            truncated.append(HumanMessage(
                content=f"Observation: [previous output truncated — {original_lines} lines]"
            ))
        else:
            truncated.append(msg)

    return truncated


# =========================================================================== #
#                              MiniAgent Pipeline                             #
# =========================================================================== #
class MiniAgent(BaseZeroShot):
    """
    Verification-guided agentic pipeline for network configuration repair.

    Replaces the monolithic single-shot prompt with a ReAct agent loop
    where the LLM can inspect configs, apply patches, run Batfish
    verification, and rollback unsafe changes iteratively.

    Inherits the full evaluation and metric infrastructure from
    :class:`BaseZeroShot` so that results are directly comparable
    with baseline runs.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Any]],
        system_prompt: str,
        net_env=None,
        split_ratio: float = 0.08,
        seed: int = 42,
        save_dir: str = None,
        timeout: int = 1200,
        prompt_style: str = "search_replace",
        parser_name: str = None,
        diagnosis_judge_config: Optional[Dict[str, Any]] = None,
        max_agent_steps: int = 30,
        verification_mode: str = "per_step",
        **kwargs,
    ):
        """
        Initialize the agentic pipeline.

        Args:
            data (Dict[str, Dict[str, Any]]): Network dataset.
            system_prompt (str): Fallback system prompt (agent uses its own).
            net_env: Batfish network environment callable.
                Defaults to None.
            split_ratio (float): Evaluation split ratio.
                Defaults to 0.08.
            seed (int): Random seed.
                Defaults to 42.
            save_dir (str): Base path for saving results.
                Defaults to None.
            timeout (int): Per-scenario timeout in seconds.
                Defaults to 1200.
            prompt_style (str): Prompt formatting style.
                Defaults to 'search_replace'.
            parser_name (str): Parser override.
                Defaults to None.
            diagnosis_judge_config (Optional[Dict[str, Any]]): LLM-judge
                settings for diagnosis scoring.
                Defaults to None.
            max_agent_steps (int): Maximum tool-call steps per scenario.
                Defaults to 30.
            verification_mode (str): One of 'per_step' or 'post_hoc'.
                Controls whether the verify tool returns real Batfish
                feedback or a deferred message.
                Defaults to 'per_step'.
        """
        kwargs["batch_api"] = False

        super().__init__(
            data=data,
            system_prompt=system_prompt,
            net_env=net_env,
            split_ratio=split_ratio,
            seed=seed,
            save_dir=save_dir,
            timeout=timeout,
            prompt_style=prompt_style,
            parser_name=parser_name,
            diagnosis_judge_config=diagnosis_judge_config,
            **kwargs,
        )
        self.max_agent_steps = max_agent_steps
        self.agent_context_mode = kwargs.get("context_sampling", "random")
        self.verification_mode = verification_mode
        self.context_prefill = kwargs.get("context_prefill", False)

    # ------------------------------------------------------------------ #
    #                         Public Entry Point                         #
    # ------------------------------------------------------------------ #
    def inference_and_eval(
        self,
        feedback_regime=None,
        max_attempts: int = 1,
        similarity_threshold: float = 1.0,
    ):
        """
        Run the agentic pipeline on all evaluation scenarios.

        Args:
            feedback_regime: Unused (kept for interface compatibility).
            max_attempts (int): Unused (agent handles iteration).
                Defaults to 1.
            similarity_threshold (float): Score threshold for success.
                Defaults to 1.0.

        Returns:
            Aggregate statistics dictionary.
        """
        if self.eval_dataset is not None:
            base_save_dir = self.save_dir
            return self._agentic_eval(
                similarity_threshold=similarity_threshold,
                base_save_dir=base_save_dir,
            )

    # ------------------------------------------------------------------ #
    #                        Core Evaluation Loop                        #
    # ------------------------------------------------------------------ #
    def _agentic_eval(
        self,
        similarity_threshold: float,
        base_save_dir: str,
    ):
        """
        Run agent loop for each scenario, evaluate, and save results.

        Args:
            similarity_threshold (float): Score threshold for success.
            base_save_dir (str): Root directory for result artefacts.

        Returns:
            Aggregate statistics dictionary.
        """
        results: List[Dict[str, Any]] = []
        agent_system_prompt = _agent_system_prompt(self.max_agent_steps)

        for sample_key, sample in tqdm(
            self.eval_dataset.items(), desc="Agentic evaluation"
        ):
            task_id = int(sample_key.split("-")[-1])
            current_task_dir = os.path.join(base_save_dir, f"Task_{task_id}")

            # Resume support
            if os.path.exists(
                os.path.join(current_task_dir, "evaluation_results.json")
            ):
                logger.info("Skipping task %d (already completed)", task_id)
                continue

            time.sleep(5)

            try:
                result = self._process_single_task(
                    task_id=task_id,
                    sample=sample,
                    current_task_dir=current_task_dir,
                    agent_system_prompt=agent_system_prompt,
                    similarity_threshold=similarity_threshold,
                )
                results.append(result)
                self._save_incremental_result(
                    result, base_save_dir, None, 1, similarity_threshold
                )
                self.clear_memory()

            except Exception as e:
                logger.error("Error processing task %d: %s", task_id, str(e))
                import traceback
                traceback.print_exc()

                context = self._summarize_task_context(sample)
                result = {
                    "sample_id": task_id,
                    "error": str(e),
                    "failure_mode": f"exception:{e.__class__.__name__}",
                    **context,
                }
                results.append(result)
                self._save_incremental_result(
                    result, base_save_dir, None, 1, similarity_threshold
                )

        self.save_dir = base_save_dir
        return self._save_metrics(
            results, base_save_dir, None, 1, similarity_threshold
        )

    # ------------------------------------------------------------------ #
    #                       Single-Task Processing                       #
    # ------------------------------------------------------------------ #
    def _process_single_task(
        self,
        task_id: int,
        sample: Dict[str, Any],
        current_task_dir: str,
        agent_system_prompt: str,
        similarity_threshold: float,
    ) -> Dict[str, Any]:
        """
        Run the agent loop on a single scenario and compute all metrics.

        Args:
            task_id (int): Numeric task identifier.
            sample (Dict[str, Any]): Dataset sample.
            current_task_dir (str): Directory for this task's artefacts.
            agent_system_prompt (str): Formatted system prompt.
            similarity_threshold (float): Success threshold.

        Returns:
            Result dictionary compatible with the base evaluation pipeline.
        """
        logger.info("Processing task %d (agentic mode)", task_id)

        # Extract sample data
        self.instruction = sample["instruction"]
        self.original_configs = sample["original_configs"]
        self.faulty_configs = sample["faulty_configs"]
        self.full_faulty_configs = sample.get(
            "full_faulty_configs", self.faulty_configs
        )
        self.full_original_configs = sample.get(
            "full_original_configs", self.original_configs
        )
        self.original_specs = sample.get("original_specs", [])
        self.specification_csv_path = sample.get("specification_csv_path")

        # Setup directories
        self.save_dir = current_task_dir
        self.fault_dir = os.path.join(current_task_dir, "fault")
        self.fix_dir = os.path.join(current_task_dir, "fix")
        os.makedirs(current_task_dir, exist_ok=True)
        os.makedirs(self.fault_dir, exist_ok=True)
        os.makedirs(self.fix_dir, exist_ok=True)

        # Persist faulty configs
        for router, content in self.faulty_configs.items():
            with open(os.path.join(self.fault_dir, router), "w") as f:
                f.write(content)

        self.clear_memory()

        # Determine which configs the agent can see
        agent_visible_configs = self._resolve_agent_configs(sample)

        # Build tool environment
        violated_specs = [
            p for p in self.original_specs if "broken" in p.lower()
        ]
        env = ToolEnvironment(
            faulty_configs=agent_visible_configs,
            original_configs=self.original_configs,
            full_faulty_configs=self.full_faulty_configs,
            topology=sample.get("topology", {}),
            violated_specs=violated_specs,
            specification_csv_path=self.specification_csv_path,
            original_specs=self.original_specs,
            net_env=self.net_env,
            evaluator=self.evaluator,
            evaluator_kwargs=self.evaluator_kwargs.copy(),
            context_mode=self.agent_context_mode,
            verification_mode=self.verification_mode,
            context_prefill=self.context_prefill,
            prefilled_instruction=sample["instruction"] if self.context_prefill else None,
        )

        # Run the agent loop
        start_time = time.time()
        fixed_results, agent_metadata = _run_agent_loop(
            model=self.model,
            environment=env,
            system_prompt=agent_system_prompt,
            max_steps=self.max_agent_steps,
            timeout=self.timeout,
        )
        inference_time = time.time() - start_time

        # Save fixed configs
        for filename, config_text in fixed_results.items():
            with open(os.path.join(self.fix_dir, filename), "w") as f:
                f.write(config_text)

        # Save agent metadata
        self._save_agent_metadata(agent_metadata)

        # Final evaluation
        fix_evaluation = self._final_evaluation(env, fixed_results)
        fix_score = (
            fix_evaluation
            .get("fix_evaluation", {})
            .get("summary", {})
            .get("fix_rate", 0.0)
        )
        fixed_cnt, unfixed_cnt, broken_cnt, fix_ratio, regression_rate = (
            self._extract_fix_stats(fix_evaluation)
        )

        # Diagnosis judge (concatenate agent reasoning as diagnosis)
        diagnosis_text = self._extract_diagnosis_from_agent(agent_metadata)
        diagnosis_eval = self._diagnosis_judge_eval(diagnosis_text)
        fix_evaluation["diagnosis_evaluation"] = diagnosis_eval
        diagnosis_score = (
            diagnosis_eval.get("mean_score")
            if not diagnosis_eval.get("skipped") else None
        )
        diagnosis_completeness = diagnosis_eval.get("mean_completeness")
        diagnosis_soundness = diagnosis_eval.get("mean_soundness")

        self._save_results(fix_evaluation, "evaluation_results")

        # Router identification metrics
        (
            gt_changed, pred_changed, tp, fp, fn,
            precision, recall, f1
        ) = self._router_identification_metrics(
            self.faulty_configs, self.original_configs, fixed_results
        )

        # Edit stats
        routers_changed, loc_changed = self._proposed_edit_stats(
            self.faulty_configs, fixed_results
        )

        # Context info
        context = self._summarize_task_context(sample)

        # Verification trajectory
        env_summary = env.get_summary()
        verification_trajectory = [
            {
                "step": v["step"],
                "fix_rate": v["fix_rate"],
                "regression_rate": v["regression_rate"],
            }
            for v in env.verification_history
        ]

        result = {
            "sample_id": task_id,
            "best_score": fix_score,
            "success": fix_score >= similarity_threshold,
            "results_dir": str(current_task_dir),
            "task_inference_time": inference_time,
            "task_input_tokens": agent_metadata.get(
                "total_prompt_tokens", 0
            ),
            "task_output_tokens": agent_metadata.get(
                "total_completion_tokens", 0
            ),
            "task_token_count": (
                agent_metadata.get("total_prompt_tokens", 0)
                + agent_metadata.get("total_completion_tokens", 0)
            ),
            "parse_failures": 0,
            "total_attempts": 1,
            "failure_mode": (
                "none" if fix_score > 0
                else (
                    agent_metadata.get("terminated_reason") or "no_fix"
                )
            ),
            "regression_rate": regression_rate,
            "router_tp": tp,
            "router_fp": fp,
            "router_fn": fn,
            "router_precision": precision,
            "router_recall": recall,
            "router_f1": f1,
            "routers_changed_proposed": routers_changed,
            "loc_changed_proposed": loc_changed,
            "diagnosis_score": diagnosis_score,
            "diagnosis_completeness": diagnosis_completeness,
            "diagnosis_soundness": diagnosis_soundness,
            # Agentic-specific metrics
            "agent_total_steps": agent_metadata.get("total_steps", 0),
            "agent_submitted": agent_metadata.get("submitted", False),
            "agent_terminated_reason": agent_metadata.get(
                "terminated_reason"
            ),
            "agent_verifications": len(env.verification_history),
            "agent_verification_trajectory": verification_trajectory,
            "agent_routers_inspected": len(
                env_summary.get("routers_inspected", [])
            ),
            "agent_patches_attempted": env_summary.get(
                "total_patches_attempted", 0
            ),
            "agent_rollbacks": env_summary.get("total_rollbacks", 0),
            "agent_context_mode": self.agent_context_mode,
            "agent_verification_mode": self.verification_mode,
            **context,
        }

        logger.info(
            "Task %d completed: fix_score=%.4f, steps=%d, "
            "verifications=%d, reason=%s",
            task_id,
            fix_score,
            agent_metadata.get("total_steps", 0),
            len(env.verification_history),
            agent_metadata.get("terminated_reason"),
        )
        return result

    # ------------------------------------------------------------------ #
    #                           Private Helpers                          #
    # ------------------------------------------------------------------ #
    def _resolve_agent_configs(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Determine selected configs the agent can see, regardless of context mode.

        Args:
            sample (Dict[str, Any]): Dataset sample.

        Returns:
            Dictionary of config filenames to config text.
        """
        return sample.get("selected_configs", sample.get("faulty_configs", {}))

    def _final_evaluation(
        self,
        env: ToolEnvironment,
        fixed_results: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Produce the final evaluation dict.  Reuses the last verification
        result when available; otherwise runs a fresh Batfish evaluation.

        Args:
            env (ToolEnvironment): Tool environment with verification history.
            fixed_results (Dict[str, str]): Agent's final config state.

        Returns:
            Evaluation dict with 'fix_evaluation' key.
        """
        if env.verification_history:
            return {
                "fix_evaluation": env.verification_history[-1]["evaluation"]
            }

        # Agent never verified — run evaluation now
        no_changes = all(
            fixed_results.get(name, content) == content
            for name, content in self.faulty_configs.items()
        )
        if no_changes:
            return {
                "fix_evaluation": {
                    "summary": {"fix_rate": 0.0},
                    "evaluation_skipped": True,
                    "reason": "No config changes applied",
                }
            }

        tmpdir = env._save_configs_to_tempdir(fixed_results)
        processed = (
            self.net_env(local_scenario_path=tmpdir.name)
            if self.net_env else None
        )
        if processed:
            eval_kwargs = self.evaluator_kwargs.copy()
            eval_kwargs["specification_csv_path"] = (
                self.specification_csv_path
            )
            eval_kwargs["reference_spec"] = (
                None if self.specification_csv_path else self.original_specs
            )
            eval_kwargs["compared_spec"] = processed
            return {"fix_evaluation": self.evaluator(**eval_kwargs)}

        return {
            "fix_evaluation": {
                "summary": {"fix_rate": 0.0},
                "evaluation_failed": True,
            }
        }

    @staticmethod
    def _extract_diagnosis_from_agent(
        agent_metadata: Dict[str, Any],
    ) -> str:
        """
        Concatenate the agent's reasoning (thought fields) into a single
        diagnosis string suitable for the LLM-as-judge scorer.

        Args:
            agent_metadata (Dict[str, Any]): Agent run metadata.

        Returns:
            Diagnosis text, or a fallback message.
        """
        thoughts = []
        for step in agent_metadata.get("steps", []):
            thought = step.get("thought", "")
            if thought:
                thoughts.append(
                    f"Step {step.get('step', '?')}: {thought}"
                )
        return (
            "\n".join(thoughts) if thoughts
            else "No diagnosis available."
        )

    @staticmethod
    def _router_identification_metrics(
        faulty_configs: Dict[str, str],
        original_configs: Dict[str, str],
        fixed_results: Dict[str, str],
    ) -> Tuple[set, set, int, int, int, float, float, float]:
        """
        Compute precision, recall, F1 for router-level fault identification.

        Args:
            faulty_configs (Dict[str, str]): Broken configs.
            original_configs (Dict[str, str]): Golden configs.
            fixed_results (Dict[str, str]): Agent's output configs.

        Returns:
            Tuple of (gt_changed, pred_changed, tp, fp, fn, precision,
            recall, f1).
        """
        gt_changed = {
            name for name, cfg in faulty_configs.items()
            if original_configs.get(name) != cfg
        }
        pred_changed = {
            name for name, cfg in faulty_configs.items()
            if fixed_results.get(name, cfg) != cfg
        }
        tp = len(gt_changed & pred_changed)
        fp = len(pred_changed - gt_changed)
        fn = len(gt_changed - pred_changed)
        precision = (
            tp / len(pred_changed) if pred_changed
            else (1.0 if not gt_changed else 0.0)
        )
        recall = (
            tp / len(gt_changed) if gt_changed
            else (1.0 if not pred_changed else 0.0)
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return gt_changed, pred_changed, tp, fp, fn, precision, recall, f1

    def _save_agent_metadata(
        self,
        agent_metadata: Dict[str, Any],
    ) -> None:
        """
        Persist agent metadata and full trajectory to disk.

        Saves a lean summary to ``agent_metadata.json`` and the complete
        step-by-step trajectory (including full model responses and
        observations) to ``agent_trajectory.json``.

        Args:
            agent_metadata (Dict[str, Any]): Raw agent metadata.
        """
        # Full trajectory with complete responses and observations
        trajectory_path = os.path.join(self.fix_dir, "agent_trajectory.json")
        try:
            with open(trajectory_path, "w") as f:
                json.dump(agent_metadata["steps"], f, indent=2, default=str)
        except Exception as e:
            logger.warning("Could not save agent trajectory: %s", e)

        # Lean summary (truncated observations, no raw responses)
        meta_path = os.path.join(self.fix_dir, "agent_metadata.json")
        serialisable = {
            k: v for k, v in agent_metadata.items()
            if k not in ("environment_summary", "steps")
        }
        # Keep truncated steps in summary
        serialisable["steps"] = [
            {k: v for k, v in step.items()
             if k not in ("raw_response", "full_observation")}
            for step in agent_metadata.get("steps", [])
        ]
        env_summary = agent_metadata.get("environment_summary", {})
        serialisable["environment_summary"] = {
            k: v for k, v in env_summary.items()
            if k != "tool_call_log"
        }
        try:
            with open(meta_path, "w") as f:
                json.dump(serialisable, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Could not save agent metadata: %s", e)