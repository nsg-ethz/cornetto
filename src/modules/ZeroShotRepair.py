"""
Repair-focused zero-shot pipeline for search-and-replace outputs.

This variant retries failed search/replace generations with a much smaller
context that only contains the touched configs plus the failing operations.

It also supports loading results from a prior naive run and only re-running
failed tasks while preserving successful results.
"""

import copy
import json
import logging
import os
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.token_counter import token_counter
from src.utils.parser_utils.errors import ParserError, classify_error, FailureCategory

from src.modules.ZeroShot import ZeroShot as BaseZeroShot

logger = logging.getLogger(__name__)


class ZeroShotRepair(BaseZeroShot):
    """
    Zero-shot pipeline that automatically issues a correction call when a
    search/replace response cannot be applied.
    
    Supports loading results from a prior naive run via `naive_results_dir` parameter.
    When provided, only failed tasks from the naive run will be re-executed while
    successful results are preserved and merged into the final output.
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
        naive_results_dir: str = None,
        naive_success_threshold: float = 1.0,
        naive_retry_parse_failures: bool = True,
        naive_retry_errors: bool = True,
        naive_retry_low_scores: bool = True,
        **kwargs,
    ):
        """
        Initialize with search/replace as the default prompt style for repair mode.
        
        Args:
            naive_results_dir: Path to a prior naive run's results directory.
                If provided, successful tasks from that run will be skipped.
            naive_success_threshold: Score threshold above which naive results
                are considered successful (default: 1.0 = only perfect scores).
            naive_retry_parse_failures: Whether to retry tasks with parse failures.
            naive_retry_errors: Whether to retry tasks with API/other errors.
            naive_retry_low_scores: Whether to retry tasks with scores below threshold.
        """
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
            **kwargs,
        )

        if self.prompt_style != "search_replace":
            logger.warning(
                "ZeroShotRepair is optimized for search_replace prompts; current style: %s",
                self.prompt_style,
            )

        # Initialize naive results handling
        self.naive_results_dir = naive_results_dir
        self.naive_success_threshold = naive_success_threshold
        self.naive_retry_parse_failures = naive_retry_parse_failures
        self.naive_retry_errors = naive_retry_errors
        self.naive_retry_low_scores = naive_retry_low_scores
        
        # Load naive results if provided
        self.naive_data = None
        self.naive_failed_ids: Set[int] = set()
        self.naive_successful_ids: Set[int] = set()
        self.naive_successful_results: List[Dict[str, Any]] = []
        
        if self.naive_results_dir:
            self._load_naive_results()

    def _load_naive_results(self) -> None:
        """
        Load and analyze results from a prior naive run to identify failed tasks.
        """
        results_path = Path(self.naive_results_dir)
        results_file = results_path / "zero_shot_benchmark_results.json"
        
        if not results_file.exists():
            logger.warning(f"Naive results file not found: {results_file}")
            return
        
        with open(results_file) as f:
            self.naive_data = json.load(f)
        
        results = self.naive_data.get("results", [])
        
        parse_failure_ids = []
        error_ids = []
        low_score_ids = []
        
        for result in results:
            sample_id = result.get("sample_id")
            if sample_id is None:
                continue
            
            is_failed = False
            
            # Check for parse failures
            if self.naive_retry_parse_failures and result.get("parse_failures", 0) > 0:
                parse_failure_ids.append(sample_id)
                is_failed = True
            
            # Check for error states
            if self.naive_retry_errors:
                if result.get("error") or result.get("failure_mode") in [
                    "api_error", "timeout", "unknown_error"
                ]:
                    if sample_id not in error_ids:
                        error_ids.append(sample_id)
                    is_failed = True
                
                # Check for responses that start with "Error"
                raw_response = result.get("raw_response", "")
                if raw_response and isinstance(raw_response, str) and raw_response.startswith("Error"):
                    if sample_id not in error_ids:
                        error_ids.append(sample_id)
                    is_failed = True
            
            # Check for low scores
            if self.naive_retry_low_scores and not is_failed:
                best_score = result.get("best_score", 0.0)
                if best_score < self.naive_success_threshold:
                    low_score_ids.append(sample_id)
                    is_failed = True
            
            if is_failed:
                self.naive_failed_ids.add(sample_id)
            else:
                self.naive_successful_ids.add(sample_id)
                # Copy and mark successful result
                result_copy = copy.deepcopy(result)
                result_copy["source_pipeline"] = "naive"
                self.naive_successful_results.append(result_copy)
        
        logger.info(f"Loaded naive results from: {self.naive_results_dir}")
        logger.info(f"  Total tasks: {len(results)}")
        logger.info(f"  Successful (score >= {self.naive_success_threshold}): {len(self.naive_successful_ids)}")
        logger.info(f"  Failed (will retry): {len(self.naive_failed_ids)}")
        logger.info(f"    - Parse failures: {len(parse_failure_ids)}")
        logger.info(f"    - API/other errors: {len(error_ids)}")
        logger.info(f"    - Low scores: {len(low_score_ids)}")

    def _should_skip_task(self, task_id: int) -> bool:
        """
        Check if a task should be skipped because it succeeded in the naive run.
        
        Args:
            task_id: The task ID to check.
            
        Returns:
            True if the task should be skipped (was successful in naive run).
        """
        if not self.naive_results_dir:
            return False
        return task_id in self.naive_successful_ids

    def _get_naive_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the naive run result for a task ID.
        
        Args:
            task_id: The task ID to look up.
            
        Returns:
            The result dictionary from the naive run, or None.
        """
        for result in self.naive_successful_results:
            if result.get("sample_id") == task_id:
                return result
        return None

    # ------------------------------------------------------------------ #
    #                          Public Interface                          #
    # ------------------------------------------------------------------ #
    def inference(
        self,
        user_prompt: str,
        step: int = 1,
    ):
        """
        Run inference and, on search/replace parse failures, trigger a
        follow-up correction call with a reduced context.
        """

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Inference timed out after {self.timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            if self.batch_api:
                # Batch path stays identical to the base implementation.
                return super().inference(user_prompt, step)

            return self._single_inference_with_repair(user_prompt, step)
        except TimeoutError as e:
            logger.error(f"Model inference timed out: {str(e)}")
            raise RuntimeError(f"Model inference failed: {e}")
        finally:
            signal.alarm(0)

    # ------------------------------------------------------------------ #
    #                         Internal Routines                          #
    # ------------------------------------------------------------------ #
    def _single_inference_with_repair(
        self,
        user_prompt: str,
        step: int,
    ):
        """
        Mirror the base single-sample inference but add a repair round on
        search/replace parser failures.
        """
        try:
            prompt_tokens = token_counter(user_prompt)

            if step == 1:
                system_message = SystemMessage(content=self.system_prompt)
                self.model.memory.chat_memory.add_message(system_message)
                system_tokens = token_counter(self.system_prompt)
            else:
                system_tokens = 0

            human_message = HumanMessage(content=user_prompt)

            response = self.model.invoke([human_message])
            logger.info(f"Model response received (length: {len(response.content)})")
            self.model.memory.chat_memory.add_ai_message(response.content)

            completion_tokens = token_counter(response.content)

            try:
                results, metadata = self._parse_single_response(
                    response,
                    self.faulty_configs,
                    self.fault_dir,
                    self.fix_dir,
                    self.parser,
                )
                metadata["token_count"] = {
                    "prompt_tokens": prompt_tokens,
                    "system_tokens": system_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + system_tokens + completion_tokens,
                }
                metadata.setdefault("raw_response", response.content)
                return results, metadata

            except Exception as parse_exc:
                raw_response = response.content if response else None
                # If the model signaled context overflow, skip repair attempts.
                if self._is_context_overflow(raw_response or str(parse_exc)):
                    logger.warning(
                        "Detected context/length overflow in model response; skipping repair retry."
                    )
                    raise self._runtime_error(parse_exc, raw_response)
                logger.warning(
                    "Search/replace parsing failed; attempting targeted repair: %s",
                    str(parse_exc),
                )

                initial_tokens = {
                    "prompt_tokens": prompt_tokens,
                    "system_tokens": system_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + system_tokens + completion_tokens,
                }

                if self.prompt_style != "search_replace":
                    raise self._runtime_error(parse_exc, raw_response)

                repair = self._attempt_search_replace_repair(
                    parse_error=parse_exc,
                    raw_response=raw_response,
                    initial_token_count=initial_tokens,
                )
                if repair is not None:
                    return repair

                raise self._runtime_error(parse_exc, raw_response)

        except Exception as e:
            raw_response = getattr(e, "response_content", None)
            if raw_response is None and "response" in locals():
                raw_response = response.content if hasattr(response, "content") else None
            err = RuntimeError(f"Model inference failed: {e}")
            setattr(err, "response_content", raw_response)
            raise err

    def _attempt_search_replace_repair(
        self,
        parse_error: Exception,
        raw_response: Optional[str],
        initial_token_count: Dict[str, int],
    ) -> Optional[Tuple[Dict[str, str], Dict[str, Any]]]:
        """
        Issue a correction call with only the touched configs and failing blocks.
        """
        replacements, prev_metadata = self._extract_replacements(raw_response)
        repair_prompt = self._build_repair_prompt(
            replacements=replacements,
            prev_metadata=prev_metadata,
            parse_error=str(parse_error),
            raw_response=raw_response,
        )

        memory_snapshot = self._snapshot_memory()
        try:
            if hasattr(self.model, "memory") and hasattr(self.model.memory, "chat_memory"):
                self.model.memory.chat_memory.clear()

            system_message = SystemMessage(content=self.system_prompt)
            user_message = HumanMessage(content=repair_prompt)

            response = self.model.invoke([system_message, user_message])
            repair_completion_tokens = token_counter(response.content)
            repair_prompt_tokens = token_counter(repair_prompt)
            repair_system_tokens = token_counter(self.system_prompt)

            repaired_configs, repair_metadata = self._parse_single_response(
                response,
                self.faulty_configs,
                self.fault_dir,
                self.fix_dir,
                self.parser,
            )

            combined_tokens = {
                "prompt_tokens": initial_token_count.get("prompt_tokens", 0)
                + repair_prompt_tokens,
                "system_tokens": initial_token_count.get("system_tokens", 0)
                + repair_system_tokens,
                "completion_tokens": initial_token_count.get("completion_tokens", 0)
                + repair_completion_tokens,
            }
            combined_tokens["total_tokens"] = sum(combined_tokens.values())
            combined_tokens["repair_prompt_tokens"] = repair_prompt_tokens
            combined_tokens["repair_completion_tokens"] = repair_completion_tokens
            combined_tokens["initial_completion_tokens"] = initial_token_count.get(
                "completion_tokens", 0
            )

            repair_metadata.setdefault("raw_response", response.content)
            repair_metadata["initial_raw_response"] = raw_response
            repair_metadata["repair_attempted"] = True
            repair_metadata["repair_succeeded"] = True
            # Classify and store the original failure that was repaired
            if isinstance(parse_error, ParserError):
                original_failure_category = parse_error.category.value
            else:
                original_failure_category = classify_error(parse_error).value
            repair_metadata["original_failure_mode"] = original_failure_category
            repair_metadata["repair_context"] = {
                "parse_error": str(parse_error),
                "original_failure_category": original_failure_category,
                "touched_configs": list(replacements.keys())
                if isinstance(replacements, dict)
                else [],
            }
            repair_metadata["token_count"] = combined_tokens

            return repaired_configs, repair_metadata

        except Exception as repair_exc:
            logger.error("Repair attempt failed: %s", str(repair_exc))
            repair_raw = response.content if "response" in locals() else None
            err = RuntimeError(f"Repair inference failed: {repair_exc}")
            setattr(err, "response_content", repair_raw)
            raise err
        finally:
            self._restore_memory(memory_snapshot)

    # ------------------------------------------------------------------ #
    #                        Prompt Construction                         #
    # ------------------------------------------------------------------ #
    def _build_repair_prompt(
        self,
        replacements: Dict[str, Any],
        prev_metadata: Dict[str, Any],
        parse_error: str,
        raw_response: Optional[str] = None,
    ) -> str:
        """
        Build a condensed correction prompt with only the failing files and
        operations.
        """
        replacements = replacements or {}
        prev_metadata = prev_metadata or {}

        touched_files = [
            name for name in replacements.keys() if name in (self.faulty_configs or {})
        ]
        if not touched_files:
            touched_files = list((self.faulty_configs or {}).keys())

        faulty_sections: List[str] = []
        for filename in touched_files:
            config_body = (self.faulty_configs or {}).get(filename, "").strip()
            faulty_sections.append(f"# Faulty config: {filename}\n{config_body}")
        faulty_blob = "\n\n".join(faulty_sections)

        try:
            replacements_yaml = yaml.safe_dump(
                {"replacements": replacements}, sort_keys=False
            )
        except Exception:
            replacements_yaml = json.dumps({"replacements": replacements}, indent=2)

        try:
            metadata_yaml = yaml.safe_dump(prev_metadata, sort_keys=False)
        except Exception:
            metadata_yaml = json.dumps(prev_metadata, indent=2)

        prompt_parts = [
            "Your previous search-and-replace YAML could not be applied.",
            "Return corrected YAML with the same top-level keys: `replacements` and `metadata`.",
            "Do not wrap the YAML in markdown.",
            f"Failure reason: {parse_error}",
            "",
            "Previous replacements:",
            replacements_yaml,
            "Previous metadata:",
            metadata_yaml if metadata_yaml else "None provided",
            "",
            "Previous malformed YAML (for reference, fix formatting and content as needed):",
            raw_response or "None provided",
            "",
            "Faulty configs you should edit (limit your changes to these routers):",
            faulty_blob,
            "",
            "Requirements:",
            "- Use the provided configs to choose exact search blocks; avoid fuzzy matches that do not exist.",
            "- Keep or improve the diagnosis and fix description in the metadata section.",
            "- If a router does not need changes, set it to `No change needed` or omit it.",
            "- Return valid YAML only with `replacements` and `metadata`.",
        ]

        return "\n".join(prompt_parts)

    def _extract_replacements(
        self,
        raw_response: Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Best-effort extraction of replacements and metadata from a failed reply.
        """
        if not raw_response:
            return {}, {}

        cleaned = re.sub(r"^```\s*yaml\s*\n", "", raw_response)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)

        try:
            parsed = yaml.safe_load(cleaned) or {}
            if not isinstance(parsed, dict):
                return {}, {}
            replacements = parsed.get("replacements") or {}
            metadata = parsed.get("metadata") or {}

            if not isinstance(replacements, dict):
                replacements = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("raw_response", raw_response)
            return replacements, metadata
        except Exception as e:
            logger.warning("Failed to extract replacements from raw response: %s", str(e))
            return {}, {"raw_response": raw_response, "parse_error": str(e)}

    # ------------------------------------------------------------------ #
    #                         Memory Management                          #
    # ------------------------------------------------------------------ #
    def _snapshot_memory(self):
        """
        Capture current chat history so we can temporarily clear it.
        """
        try:
            if hasattr(self.model, "memory") and hasattr(self.model.memory, "chat_memory"):
                return list(self.model.memory.chat_memory.messages)
        except Exception:
            pass
        return None

    def _restore_memory(self, snapshot):
        """
        Restore chat history after a temporary clear.
        """
        if snapshot is None:
            return
        try:
            if hasattr(self.model, "memory") and hasattr(self.model.memory, "chat_memory"):
                self.model.memory.chat_memory.messages = snapshot
        except Exception as e:
            logger.warning("Could not restore chat memory: %s", str(e))

    # ------------------------------------------------------------------ #
    #               Naive Results Integration for _single_eval           #
    # ------------------------------------------------------------------ #
    def _single_eval(
        self,
        feedback_regime: bool,
        max_attempts: int,
        similarity_threshold: float,
        base_save_dir: str,
        continuous_save: bool = True
    ):
        """
        Handle one inference and evaluation at a time.
        
        Overrides base class to skip successful naive tasks and merge results.
        """
        from tqdm import tqdm
        import time
        from datetime import datetime
        
        # If we have naive results, we'll merge them
        if self.naive_results_dir:
            logger.info(f"Repair mode with naive results from: {self.naive_results_dir}")
            logger.info(f"  Skipping {len(self.naive_successful_ids)} successful tasks")
            logger.info(f"  Retrying {len(self.naive_failed_ids)} failed tasks")
        
        # Store results - start with successful naive results
        results = []
        if self.naive_results_dir:
            results.extend(self.naive_successful_results)
            logger.info(f"Pre-loaded {len(self.naive_successful_results)} successful naive results")
        
        # Track statistics for repair
        repair_stats = {
            "tasks_skipped": 0,
            "tasks_retried": 0,
            "repair_successes": 0,
            "repair_failures": 0,
        }
        
        # Iterate over all instruction prompt and output pair samples
        for sample_key, sample in tqdm(self.eval_dataset.items(), desc="Processing examples"):
            # Extract task ID
            task_id = int(sample_key.split("-")[-1])
            
            # Check if this task should be skipped (successful in naive run)
            if self._should_skip_task(task_id):
                logger.info(f"Skipping task {task_id} (successful in naive run)")
                repair_stats["tasks_skipped"] += 1
                continue
            
            # Setup save path for this task using base directory
            current_task_dir = os.path.join(base_save_dir, f"Task_{task_id}")
            
            # Skip if task already completed (resume from checkpoint)
            if os.path.exists(os.path.join(current_task_dir, "evaluation_results.json")):
                logger.info(f"Skipping task {task_id} (already completed in current run)")
                continue
            
            # Make a stop
            time.sleep(15)
            
            repair_stats["tasks_retried"] += 1

            try:
                # Task ID info
                logger.info(f"Processing task {task_id} (repair mode)")

                # Call the parent class's single task processing logic
                # We need to replicate the core logic here to get the result
                result = self._process_single_task_for_repair(
                    task_id, sample, current_task_dir, base_save_dir,
                    feedback_regime, max_attempts, similarity_threshold
                )
                
                # Mark as from repair pipeline
                result["source_pipeline"] = "repair"
                results.append(result)
                
                if result.get("success", False):
                    repair_stats["repair_successes"] += 1
                else:
                    repair_stats["repair_failures"] += 1
                
                # Continuous save
                if continuous_save:
                    self._save_incremental_result(
                        result, base_save_dir, feedback_regime,
                        max_attempts, similarity_threshold
                    )

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                context = self._summarize_task_context(sample)
                failure_mode = self._failure_mode_tag(False, None, e)
                raw_response = getattr(e, "response_content", None)
                
                result = {
                    "sample_id": task_id,
                    "error": str(e),
                    "failure_mode": failure_mode,
                    "raw_response": raw_response,
                    "source_pipeline": "repair",
                    **context,
                }
                results.append(result)
                repair_stats["repair_failures"] += 1
                
                if continuous_save:
                    self._save_incremental_result(
                        result, base_save_dir, feedback_regime,
                        max_attempts, similarity_threshold
                    )
        
        # Reset save directory
        self.save_dir = base_save_dir
        
        # Log repair statistics
        if self.naive_results_dir:
            logger.info(f"\nRepair statistics:")
            logger.info(f"  Tasks skipped (naive success): {repair_stats['tasks_skipped']}")
            logger.info(f"  Tasks retried: {repair_stats['tasks_retried']}")
            logger.info(f"  Repair successes: {repair_stats['repair_successes']}")
            logger.info(f"  Repair failures: {repair_stats['repair_failures']}")
        
        # Calculate aggregate statistics and save final results
        return self._save_metrics_with_naive_info(
            results, base_save_dir, feedback_regime, max_attempts, similarity_threshold
        )

    def _process_single_task_for_repair(
        self,
        task_id: int,
        sample: Dict[str, Any],
        current_task_dir: str,
        base_save_dir: str,
        feedback_regime: bool,
        max_attempts: int,
        similarity_threshold: float
    ) -> Dict[str, Any]:
        """
        Process a single task through the repair pipeline.
        
        This replicates the core logic from the parent _single_eval for one task.
        """
        import time
        from src.utils.prompt_utils.prompts import _feedback_prompt
        
        # Extract sample data ingredients
        self.instruction = sample["instruction"]
        self.original_configs = sample["original_configs"]
        self.faulty_configs = sample["faulty_configs"]
        self.full_faulty_configs = sample.get("full_faulty_configs", self.faulty_configs)
        self.full_original_configs = sample.get("full_original_configs", self.original_configs)
        self.original_specs = sample.get("original_specs", [])
        self.specification_csv_path = sample.get("specification_csv_path")

        self.save_dir = current_task_dir
        self.fault_dir = os.path.join(current_task_dir, "fault")
        self.fix_dir = os.path.join(current_task_dir, "fix")
        
        os.makedirs(current_task_dir, exist_ok=True)
        os.makedirs(self.fault_dir, exist_ok=True)
        os.makedirs(self.fix_dir, exist_ok=True)

        # Clear memory
        self.clear_memory()

        # Track best results
        best_fixed_results = None
        best_processed_results = None
        best_evaluation = None
        best_score = 0.0
        best_metadata = None
        attempt_logs = []

        # Save instruction (wrap in dict to avoid copy() error)
        self._save_results({"instruction": self.instruction}, "instruction_prompt")
        current_prompt = self.instruction

        for attempt in range(1, max_attempts + 1):
            if max_attempts > 1:
                attempt_dir = os.path.join(self.fix_dir, f"attempt_{attempt}")
                os.makedirs(attempt_dir, exist_ok=True)
                original_fix_dir = self.fix_dir
                self.fix_dir = attempt_dir

            start_time = time.time()
            parse_error_encountered = None
            raw_response_error = None
            
            try:
                fixed_results, metadata = self.inference(current_prompt, attempt)
                parse_failed = False
            except Exception as e:
                parse_error_encountered = str(e)
                raw_response_error = getattr(e, "response_content", None)
                fixed_results = {}
                metadata = {
                    "problem_diagnosis": "Parsing failed",
                    "proposed_fix": "Unable to parse model response",
                    "parse_error": parse_error_encountered,
                    "raw_response": raw_response_error,
                }
                parse_failed = True
            
            inference_time = time.time() - start_time

            if max_attempts > 1:
                self.fix_dir = original_fix_dir

            # Handle parse failure or evaluate
            if parse_failed:
                processed_results = None
                evaluation = {
                    "fault_evaluation": {},
                    "fix_evaluation": {
                        "summary": {"fix_rate": 0.0},
                        "parse_failed": True,
                    }
                }
                fix_score = 0.0
                self._save_parse_failure(metadata)
                suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                self._save_results(evaluation, "evaluation_results", suffix)
            else:
                no_changes = all(
                    fixed_results.get(name, content) == content
                    for name, content in self.faulty_configs.items()
                )

                if no_changes:
                    evaluation = {
                        "fault_evaluation": {},
                        "fix_evaluation": {
                            "summary": {"fix_rate": 0.0},
                            "evaluation_skipped": True,
                            "reason": "No config changes applied"
                        }
                    }
                    fix_score = 0.0
                    suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                    self._save_results(evaluation, "evaluation_results", suffix)
                    processed_results = None
                else:
                    fixed_tmp = self._save_tempdir(fixed_results)
                    if self.net_env is not None:
                        processed_results = self._run_net_env(fixed_tmp.name)
                    else:
                        processed_results = None

                    if self.evaluator is not None and processed_results is not None:
                        evaluation = {}
                        temp_eval_kwargs = self.evaluator_kwargs.copy()
                        temp_eval_kwargs["specification_csv_path"] = self.specification_csv_path
                        temp_eval_kwargs["reference_spec"] = None if self.specification_csv_path else self.original_specs
                        temp_eval_kwargs["compared_spec"] = processed_results
                        evaluation["fix_evaluation"] = self.evaluator(**temp_eval_kwargs)
                        
                        suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                        self._save_results(evaluation, "evaluation_results", suffix)
                        fix_score = evaluation.get("fix_evaluation", {}).get("summary", {}).get("fix_rate", 0.0)
                    else:
                        evaluation = {
                            "fault_evaluation": {},
                            "fix_evaluation": {
                                "summary": {"fix_rate": 0.0},
                                "evaluation_failed": True,
                                "reason": "Evaluator or processed results not available"
                            }
                        }
                        fix_score = 0.0
                        suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                        self._save_results(evaluation, "evaluation_results", suffix)

            # Store attempt info
            parse_error_value = metadata.get("parse_error") or metadata.get("format_error") if parse_failed else parse_error_encountered
            attempt_info = {
                "attempt": attempt,
                "fixed_results": fixed_results,
                "processed_results": processed_results,
                "fix_evaluation": evaluation["fix_evaluation"],
                "metadata": metadata,
                "parse_error": parse_error_value,
                "raw_response_error": raw_response_error,
                "token_count": metadata.get("token_count", {}),
                "inference_time": inference_time,
                "parse_failed": parse_failed,
            }
            attempt_logs.append(attempt_info)
            
            if fix_score >= best_score:
                best_score = fix_score
                best_fixed_results = fixed_results
                best_processed_results = processed_results
                best_evaluation = evaluation
                best_metadata = metadata

                if max_attempts > 1 and not parse_failed:
                    for filename, content in fixed_results.items():
                        with open(os.path.join(self.fix_dir, filename), 'w') as f:
                            f.write(content)
        
            if fix_score >= similarity_threshold:
                break

            if attempt < max_attempts and feedback_regime and "fix_evaluation" in evaluation and not parse_failed:
                current_prompt = _feedback_prompt(
                    model_name=self.model_name,
                    attempt=attempt_info["attempt"],
                    fixed_specs=attempt_info["processed_results"],
                    fix_evaluation=attempt_info["fix_evaluation"],
                    fix_metadata=attempt_info["metadata"],
                )

        # Build result
        parse_failures = sum(1 for log in attempt_logs if log.get("parse_failed", False))
        total_attempts = len(attempt_logs)
        context = self._summarize_task_context(sample)
        parse_failure_for_mode = parse_failures == total_attempts and total_attempts > 0
        failure_mode = self._failure_mode_tag(parse_failure_for_mode, best_evaluation)
        
        # Diagnosis evaluation
        parse_error_detail = next(
            (log.get("metadata", {}).get("parse_error") for log in reversed(attempt_logs) if log.get("metadata", {}).get("parse_error")),
            None
        )
        diagnosis_eval = self._diagnosis_judge_eval(
            (best_metadata or {}).get("problem_diagnosis"),
            fallback_text=(best_metadata or {}).get("raw_response") or parse_error_detail,
        )
        if best_evaluation is not None:
            best_evaluation["diagnosis_evaluation"] = diagnosis_eval
        diagnosis_score = diagnosis_eval.get("mean_score") if not diagnosis_eval.get("skipped") else None
        diagnosis_completeness = diagnosis_eval.get("mean_completeness")
        diagnosis_soundness = diagnosis_eval.get("mean_soundness")
        
        raw_response = next(
            (log.get("metadata", {}).get("raw_response") for log in reversed(attempt_logs) if log.get("metadata", {}).get("raw_response")),
            None
        )
        if not parse_error_detail:
            parse_error_detail = next(
                (log.get("parse_error") for log in reversed(attempt_logs) if log.get("parse_error")),
                None
            )
        if failure_mode == "none" and parse_error_detail:
            failure_mode = "parse_retry"

        patch_fail_count = best_metadata.get("patch_fail_count", 0) if best_metadata else 0
        routers_changed_proposed, loc_changed_proposed = self._proposed_edit_stats(self.faulty_configs, best_fixed_results)
        
        # Router metrics
        gt_changed = {name for name, cfg in self.faulty_configs.items() if self.original_configs.get(name) != cfg}
        pred_changed = set()
        if best_fixed_results:
            for name, cfg in self.faulty_configs.items():
                if best_fixed_results.get(name, cfg) != cfg:
                    pred_changed.add(name)
        tp = len(gt_changed & pred_changed)
        fp = len(pred_changed - gt_changed)
        fn = len(gt_changed - pred_changed)
        if best_fixed_results:
            precision = tp / len(pred_changed) if len(pred_changed) > 0 else (1.0 if len(gt_changed) == 0 else 0.0)
            recall = tp / len(gt_changed) if len(gt_changed) > 0 else (1.0 if len(pred_changed) == 0 else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = None

        fixed_cnt, unfixed_cnt, broken_cnt, fix_ratio, regression_rate = self._extract_fix_stats(best_evaluation if best_evaluation else {})
        
        # Token stats
        retrieval_meta = sample.get("retrieval_meta", {})
        retrieval_tokens = retrieval_meta.get("token_count", {}).get("total_tokens", 0)
        retrieval_prompt_tokens = retrieval_meta.get("token_count", {}).get("prompt_tokens", 0) + retrieval_meta.get("token_count", {}).get("system_tokens", 0)
        retrieval_completion_tokens = retrieval_meta.get("token_count", {}).get("completion_tokens", 0)
        retrieval_time = retrieval_meta.get("inference_time", 0)
        input_tokens_fix = sum((log.get("token_count", {}).get("prompt_tokens", 0) + log.get("token_count", {}).get("system_tokens", 0)) for log in attempt_logs)
        output_tokens_fix = sum(log.get("token_count", {}).get("completion_tokens", 0) for log in attempt_logs)
        task_input_tokens = input_tokens_fix + retrieval_prompt_tokens
        task_output_tokens = output_tokens_fix + retrieval_completion_tokens
        task_prompt_tokens = sum(log.get("token_count", {}).get("prompt_tokens", 0) for log in attempt_logs)

        result = {
            "sample_id": task_id,
            "best_score": best_score,
            "success": best_score >= similarity_threshold,
            "results_dir": str(current_task_dir),
            "task_token_count": sum(log.get("token_count", {}).get("total_tokens", 0) for log in attempt_logs) + retrieval_tokens,
            "task_prompt_tokens": task_prompt_tokens + retrieval_prompt_tokens,
            "task_input_tokens": task_input_tokens,
            "task_output_tokens": task_output_tokens,
            "task_inference_time": sum(log.get("inference_time", 0) for log in attempt_logs) + retrieval_time,
            "parse_failures": parse_failures,
            "total_attempts": total_attempts,
            "failure_mode": failure_mode,
            "raw_response": raw_response,
            "parse_error": parse_error_detail,
            "routers_changed_proposed": routers_changed_proposed,
            "loc_changed_proposed": loc_changed_proposed,
            "patch_fail_count": patch_fail_count,
            "regression_rate": regression_rate,
            "router_tp": tp,
            "router_fp": fp,
            "router_fn": fn,
            "router_precision": precision,
            "router_recall": recall,
            "router_f1": f1,
            "diagnosis_score": diagnosis_score,
            "diagnosis_completeness": diagnosis_completeness,
            "diagnosis_soundness": diagnosis_soundness,
            **context,
        }
        
        logger.info(
            f"Task {task_id} completed with score: {best_score:.4f} "
            f"(fixed={fixed_cnt}, unfixed={unfixed_cnt}, "
            f"broken={broken_cnt}, fix_ratio={fix_ratio:.4f})"
        )
        
        return result

    def _save_metrics_with_naive_info(
        self,
        results: List[Dict[str, Any]],
        base_save_dir: str,
        feedback_regime: bool,
        max_attempts: int,
        similarity_threshold: float
    ):
        """
        Save metrics with additional information about naive results integration.
        """
        # Sort results by sample_id for consistency
        results.sort(key=lambda x: x.get("sample_id", 0))
        
        # Compute source breakdown
        naive_results = [r for r in results if r.get("source_pipeline") == "naive"]
        repair_results = [r for r in results if r.get("source_pipeline") == "repair"]
        
        # Use parent's save_metrics for the core functionality
        final_output = self._save_metrics(
            results, base_save_dir, feedback_regime, max_attempts, similarity_threshold
        )
        
        # Add naive integration info to the saved file
        results_file = os.path.join(base_save_dir, "zero_shot_benchmark_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
            
            # Add source breakdown
            data["source_breakdown"] = {
                "from_naive": len(naive_results),
                "from_repair": len(repair_results),
            }
            
            # Add naive run info if available
            if self.naive_results_dir:
                data["naive_run_info"] = {
                    "naive_results_dir": self.naive_results_dir,
                    "naive_success_threshold": self.naive_success_threshold,
                    "naive_successful_count": len(self.naive_successful_ids),
                    "naive_failed_count": len(self.naive_failed_ids),
                }
                
                # Compute separate stats for naive and repair
                naive_successful = [r for r in naive_results if r.get("success", False)]
                repair_successful = [r for r in repair_results if r.get("success", False)]
                
                data["stats"]["naive_success_rate"] = len(naive_successful) / len(naive_results) if naive_results else 0.0
                data["stats"]["repair_success_rate"] = len(repair_successful) / len(repair_results) if repair_results else 0.0
            
            with open(results_file, "w") as f:
                json.dump(data, f, indent=2)
        
        return final_output

    # ------------------------------------------------------------------ #
    #                        Utility Helpers                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_context_overflow(text: Optional[str]) -> bool:
        """
        Heuristic detection for context/window overflow messages to avoid futile repairs.
        """
        if not text:
            return False
        lowered = str(text).lower()
        patterns = [
            "maximum context length",
            "max context length",
            "context length exceeded",
            "context length is",
            "token limit",
            "too many tokens",
            "context window",
            "reduce the length",
            "request too large",
            "input tokens exceed",
            "reduce the length"
            "tokens exceed",
        ]
        return any(pat in lowered for pat in patterns)

    @staticmethod
    def _runtime_error(exc: Exception, raw_response: Optional[str]) -> RuntimeError:
        err = RuntimeError(f"Model inference failed: {exc}")
        setattr(err, "response_content", raw_response)
        return err
