""" 
This module contains the implementation of end-to-end communication
with selected model under zero-shot setting. 
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import re
import json
import time
import yaml
import math
import logging
import random
import tempfile
import tiktoken
import signal
import csv
import difflib

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from tqdm import tqdm
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.utils.prompt_utils.prompts import _feedback_prompt, _formatter_prompt
from src.utils.reproducibility import set_all_seeds
from src.modules.chat.model_registry import create_chat_model
from src.utils.evaluation_utils.newer_spec_evaluator import generate_spec_fix_report
from src.utils.evaluation_utils.diagnosis_judge_evaluation import (
    evaluate_diagnosis_with_llm_judges,
    infer_fault_metrics_path,
)
from src.utils.parser_utils.parser_registry import create_parser
from src.utils.parser_utils.errors import (
    ParserError,
    FailureCategory,
    classify_error,
    format_failure_mode,
)
from src.utils.token_counter import token_counter

logger = logging.getLogger(__name__)


def _get_failure_group(category: FailureCategory) -> str:
    """
    Get the high-level group for a failure category.
    """
    yaml_categories = {
        FailureCategory.YAML_SYNTAX,
        FailureCategory.YAML_EMPTY,
        FailureCategory.YAML_STRUCTURE,
    }
    search_replace_categories = {
        FailureCategory.SEARCH_NOT_FOUND,
        FailureCategory.SEARCH_MULTIPLE,
        FailureCategory.SEARCH_EMPTY,
        FailureCategory.REPLACEMENT_INVALID,
    }
    config_categories = {
        FailureCategory.MISSING_CONFIG,
        FailureCategory.EMPTY_RESULT,
    }
    api_categories = {
        FailureCategory.API_ERROR,
        FailureCategory.API_QUOTA,
        FailureCategory.API_TIMEOUT,
        FailureCategory.CONTEXT_OVERFLOW,
    }
    
    if category in yaml_categories:
        return "yaml_parsing"
    if category in search_replace_categories:
        return "search_replace"
    if category in config_categories:
        return "config_data"
    if category in api_categories:
        return "api_network"
    if category == FailureCategory.NONE:
        return "success"
    if category == FailureCategory.PARSE_RETRY:
        return "recovered"
    return "other"


# =========================================================================== #
#                          Zero-Shot Learning Class                           #
# =========================================================================== #
class ZeroShot:
    """
    Zero-shot communication pipeline consisting of model prompting, output parsing
    and evaluation with feedback regime availability.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Any]],
        system_prompt: str,
        net_env = None,
        split_ratio: float = 0.08,
        seed: int = 42,         
        save_dir: str = None,
        timeout: int = 1200,
        prompt_style: str = "diff_patch",
        parser_name: str = None,
        diagnosis_judge_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the zero-shot learning with dataset support.

        Args:
            data (Dict[str, Dict[str, Any]]): Network dataset.
            system_prompt (str): System prompt content.
            net_env: Network environment.
                Defaults to None.
            split_ratio (float): Ratio of data to use for evaluation.
                Defaults to 0.2.
            seed (int): Random seed.
                Defaults to 42.
            save_dir (str): Base path for saving results.
                Defaults to None.
            timeout (int): Timeout in seconds for inference calls.
                Defaults to 1200.
            diagnosis_judge_config (Optional[Dict[str, Any]]): Settings for
                the diagnosis LLM-judge metric (enable flag, models, limits).
        """
        # Initialize model specifications
        self.model_kwargs = kwargs
        self.model_provider = self.model_kwargs.get("provider", "")
        self.model_name = self.model_kwargs.get("model_name", "")
        self.batch_api = self.model_kwargs.get("batch_api", False)

        # Check if few-shot Chain-of-Thought enabled
        self.few_shot_mode = self.model_kwargs.get("few_shot_mode", False)

        # Set reproducibility seed
        self.seed = seed
        set_all_seeds(self.seed)

        # Setup directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(self.save_dir)
        
        # Store network environment and prompt style
        self.net_env = net_env
        self.prompt_style = prompt_style
        self.diagnosis_judge_config = diagnosis_judge_config or {}

        # Get appropriate parser based on network environment instance or explicit selection
        selected_parser = parser_name
        if not selected_parser and prompt_style == "search_replace":
            selected_parser = "search_replace"
        self.parser = create_parser(net_env, selected_parser)

        # Store evaluator
        self.evaluator = generate_spec_fix_report
        self.evaluator_kwargs = {
            "reference_spec": None,
            "compared_spec": None,
            "specification_csv_path": None,
            "broken_only": False,
        }
        
        # Create model instance
        self.model = create_chat_model(self.model_provider, **self.model_kwargs)

        # Split full dataset
        self.full_dataset = data
        _, self.eval_dataset = self._split_dataset(data, split_ratio)

        # Augment eval set with two-shot CoT examples
        # if self.few_shot_mode:
        #     self.eval_dataset = add_few_shot_cot(self.eval_dataset)

        # Create system prompt
        self.system_prompt = system_prompt

        # Set inference timeout
        self.timeout = timeout

    def clear_memory(self):
        """
        Clear memory cache of specified language model.
        """
        if (hasattr(self.model, 'memory') and 
            hasattr(self.model.memory, 'chat_memory')):
            self.model.memory.chat_memory.clear()

    def _split_dataset(
            self, 
            dataset: Dict[str, Dict[str, Any]], 
            split_ratio: float
        ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Split dataset into training and evaluation sets.
        
        Args:
            dataset (Dict[str, Dict[str, Any]]): Full dataset.
            split_ratio (float): Ratio of data to use for evaluation.
            
        Returns:
            Tuple of train and evaluation datasets.
        """
        # Get all sample IDs and shuffle them deterministically
        all_sample_ids = list(dataset.keys())
        
        # Use the already set random seed
        random.shuffle(all_sample_ids)
        
        # Determine split indices
        split_idx = int(len(all_sample_ids) * (1 - split_ratio))
        train_ids = all_sample_ids[:split_idx]
        eval_ids = all_sample_ids[split_idx:]
        
        # Split into train and evaluation datasets
        train_dataset = {id: dataset[id] for id in train_ids}
        eval_dataset = {id: dataset[id] for id in eval_ids}
        
        logger.info(f"Split dataset into {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
        
        # Save the split for reproducibility
        split_info = {
            "train_ids": train_ids,
            "eval_ids": eval_ids,
            "split_ratio": split_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save a separate split metadata
        split_file = os.path.join(self.save_dir, "dataset_split_metadata.json")
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Dataset split information saved to {split_file}")
        
        return train_dataset, eval_dataset

    def _prepare_batch_prompts(
        self, 
        dataset: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Prepare batch prompts for batch API processing.
        
        Args:
            dataset (Dict[str, Dict]): Dataset to process.
            
        Returns:
            Dictionary of formatted prompts for batch processing.
        """
        batch_prompts = {}
        
        for sample_id, sample in dataset.items():
            # Create full prompt with system and user messages            
            batch_prompts[sample_id] = {
                "system_prompt": self.system_prompt,
                "user_prompt": sample["instruction"],
                "original_data": sample
            }
        
        return batch_prompts

    def inference(
        self, 
        user_prompt: Union[str, Dict[str, Dict]],
        step: int = 1
    ) -> Union[Tuple[Dict, Dict], Dict]:
        """
        Run inference with the given model using user provided prompt.

        Args:
            user_prompt (Union[str, Dict[str, Dict]]): Task and instruction prompted by user. 
            step (int): Number of rounds of dialogue.
                Defaults to 1.

        Returns:
            Tuple of model generation and metadata.
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Inference timed out after {self.timeout} seconds")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            if self.batch_api:
                try:
                    # Calculate total prompt tokens for batch
                    total_prompt_tokens = 0
                    for prompt_data in user_prompt.values():
                        total_prompt_tokens += token_counter(prompt_data["user_prompt"])

                    # Run batch inference
                    response = self.model.invoke(user_prompt)

                    # Parse batch responses
                    batch_results = {}
                    batch_metadata = {}
                    
                    for sample_id, response_content in response.items():
                        try:
                            # Get faulty configs from original data for this sample
                            original_faulty_configs = getattr(self, 'faulty_configs', None)
                            if sample_id in user_prompt:
                                original_data = user_prompt[sample_id].get("original_data", {})
                                self.faulty_configs = original_data.get("faulty_configs", {})
                            
                            # Parse individual response using batch parser
                            parsed_result, metadata = self._parse_batch_response(
                                response_content,
                                self.faulty_configs,
                                self.parser
                            )                            
                            
                            batch_results[sample_id] = parsed_result
                            batch_metadata[sample_id] = metadata
                            
                            # Restore original fault directory
                            if original_faulty_configs is not None:
                                self.faulty_configs = original_faulty_configs
                            elif hasattr(self, 'faulty_configs'):
                                delattr(self, 'faulty_configs')
                            
                        except Exception as e:
                            logger.error(f"Failed to parse response for sample {sample_id}: {str(e)}")
                            batch_results[sample_id] = {}
                            batch_metadata[sample_id] = {
                                "problem_diagnosis": "Parsing failed",
                                "proposed_fix": "Unable to parse model response",
                                "parse_error": str(e),
                                "raw_response": response_content
                            }
                            
                            # Restore original faulty_configs even if parsing failed
                            if original_faulty_configs is not None:
                                self.faulty_configs = original_faulty_configs
                            elif hasattr(self, 'faulty_configs'):
                                delattr(self, 'faulty_configs')

                    # Calculate completion tokens
                    total_completion_tokens = sum(
                        token_counter(response_content) 
                        for response_content in response.values()
                    )

                    # Add aggregate token counts
                    aggregate_metadata = {
                        "token_count": {
                            "prompt_tokens": total_prompt_tokens,
                            "system_tokens": token_counter(self.system_prompt) * len(user_prompt),
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": \
                                total_prompt_tokens + \
                                token_counter(self.system_prompt) * len(user_prompt) + \
                                total_completion_tokens
                        },
                        "batch_size": len(user_prompt),
                        "individual_metadata": batch_metadata
                    }

                    return batch_results, aggregate_metadata
                except Exception as e:
                    logger.error(f"Batched model inference failed: {str(e)}")
                    raise RuntimeError(f"Batched model inference failed: {e}")
            else:
                # Single mode inference
                try:
                    # Count prompt tokens
                    prompt_tokens = token_counter(user_prompt)

                    # Create system message and add to memory
                    if step == 1:
                        system_message = SystemMessage(content=self.system_prompt)
                        self.model.memory.chat_memory.add_message(system_message)
                        # Count system message tokens
                        system_tokens = token_counter(self.system_prompt)
                    else:
                        system_tokens = 0

                    # Create human message (memory addition is handled by _prepend_buffer_memory in invoke)
                    human_message = HumanMessage(content=user_prompt)

                    # Get model response
                    response = self.model.invoke([human_message])
                    self.model.memory.chat_memory.add_ai_message(response.content)

                    # Count completion tokens
                    completion_tokens = token_counter(response.content)

                    logger.info(f"Model response received (length: {len(response.content)})")

                    # Parse and save model outputs
                    try:
                        results, metadata = self._parse_single_response(
                            response, 
                            self.faulty_configs, 
                            self.fault_dir, 
                            self.fix_dir,
                            self.parser
                        )
                    except Exception as e:
                        # Attach raw response for downstream diagnostics
                        e.response_content = response.content if response else None
                        raise

                    # Add token counts to metadata
                    metadata["token_count"] = {
                        "prompt_tokens": prompt_tokens,
                        "system_tokens": system_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + system_tokens + completion_tokens
                    }
                    metadata.setdefault("raw_response", response.content)

                    # Save metadata with token counts
                    metadata_path = os.path.join(self.fix_dir, "fix_metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    return results, metadata 

                except Exception as e:
                    logger.error(f"Model inference failed: {str(e)}")
                    raw_response = getattr(e, "response_content", None)
                    if raw_response is None and "response" in locals():
                        raw_response = response.content if hasattr(response, "content") else None
                    err = RuntimeError(f"Model inference failed: {e}")
                    setattr(err, "response_content", raw_response)
                    raise err     
                           
        except TimeoutError as e:
            logger.error(f"Model inference timed out: {str(e)}")
            raise RuntimeError(f"Model inference failed: {e}")
        finally:
            # Cancel the alarm
            signal.alarm(0)

    def inference_and_eval(
        self,
        feedback_regime = None,
        max_attempts: int = 3,
        similarity_threshold: float = 1.0
    ):
        """
        Run inference with feedback loop integration.
        
        Args:
            feedback_regime (bool): Feedback control flag.
                Defaults to None.
            max_attempts (int): Number of maximum retries.
                Defaults to 3.
            similarity_threshold (float): Target similarity score.
                Defaults to 1.0.
                
        Returns:
            Tuple of best configuration, environment processing, evaluation results, and score
        """
        if self.eval_dataset is not None:
            # Store the base save directory to avoid nesting
            base_save_dir = self.save_dir

            # Handle batch or one single task processing
            if self.batch_api:
                return self._batch_eval(
                    feedback_regime, max_attempts, similarity_threshold, base_save_dir
                )
            else:
                return self._single_eval(
                    feedback_regime, max_attempts, similarity_threshold, base_save_dir
                )

    def _batch_eval(
        self,
        feedback_regime: bool,
        max_attempts: int,
        similarity_threshold: float,
        base_save_dir: str
    ):
        """
        Handle full batch evaluation with automatic splitting.
        
        Args:
            feedback_regime (bool): Feedback control flag.
                Defaults to None.
            max_attempts (int): Number of maximum retries.
                Defaults to 3.
            similarity_threshold (float): Target similarity score.
                Defaults to 1.0.
            base_save_dir (str): Base directory for saving results.
                
        Returns:
            Dictionary of computed statistics and results.
        """
        # Store results
        results = []
        
        # Split into batches of 50 samples each
        all_samples = list(self.eval_dataset.items())
        batch_size = 50
        
        logger.info(f"Splitting {len(all_samples)} samples into batches of {batch_size}")
        
        # Process in batches
        for batch_idx in range(0, len(all_samples), batch_size):
            batch_samples = all_samples[batch_idx:batch_idx + batch_size]
            batch_dataset = dict(batch_samples)
            
            batch_num = (batch_idx // batch_size) + 1
            total_batches = math.ceil(len(all_samples) / batch_size)
            
            logger.info(f"Processing batch {batch_num}/{total_batches}...")
            logger.info(f"Batch {batch_num}: {len(batch_dataset)} samples")
            
            # Prepare batch prompts
            batch_prompts = self._prepare_batch_prompts(batch_dataset)
            
            try:
                # Run batch model inference
                start_time = time.time()
                batch_results, batch_metadata = self.inference(batch_prompts)
                inference_time = time.time() - start_time
                
                logger.info(f"Batch {batch_num} inference completed in {inference_time:.2f} seconds")
                
                # Process each result individually
                for sample_id in batch_results:
                    # Extract task ID
                    id = int(sample_id.split("-")[-1])

                    try:
                        # Task ID info                    
                        logger.info(f"Processing task {id}")
                        
                        # Get original sample data
                        sample = batch_dataset[sample_id]
                        
                        # Extract sample data ingredients
                        self.instruction = sample["instruction"]
                        self.original_configs = sample["original_configs"]
                        self.faulty_configs = sample["faulty_configs"]
                        # Optional full-config snapshots (e.g., retrieval pipelines)
                        self.full_faulty_configs = sample.get("full_faulty_configs", self.faulty_configs)
                        self.full_original_configs = sample.get("full_original_configs", self.original_configs)
                        self.original_specs = sample.get("original_specs", [])
                        self.faulty_specs = sample.get("faulty_specs")
                        self.specification_csv_path = sample.get("specification_csv_path")
                        
                        # Setup save path for this task
                        current_task_dir = os.path.join(base_save_dir, f"Task_{id}")
                        self.save_dir = current_task_dir
                        
                        # Setup complementary paths for fix and fault operations
                        self.fault_dir = os.path.join(current_task_dir, "fault")
                        self.fix_dir = os.path.join(current_task_dir, "fix")
                        
                        os.makedirs(current_task_dir, exist_ok=True)
                        os.makedirs(self.fault_dir, exist_ok=True)
                        os.makedirs(self.fix_dir, exist_ok=True)
                        
                        # Get parsed results from model inference
                        fixed_results = batch_results[sample_id]
                        individual_metadata = batch_metadata["individual_metadata"][sample_id]
                        
                        # Save instruction prompt
                        self._save_results(self.instruction, "instruction_prompt")
                        
                        # Check if parsing failed
                        parse_failed = "parse_error" in individual_metadata
                        
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
                            self._save_parse_failure(individual_metadata)
                            
                            logger.info(f"Task {id} failed to parse... Assigning fix similarity score: 0.0")
                            
                        else:
                            # Save all faulty configs to fault directory
                            for router, content in self.faulty_configs.items():
                                fault_config_path = os.path.join(self.fault_dir, router)
                                with open(fault_config_path, "w") as f:
                                    f.write(content)
                            
                            # Then process each config file from model output
                            for filename, config_text in fixed_results.items():
                                config_path = os.path.join(self.fix_dir, filename)

                                if config_text is None:
                                    logger.warning(f"Config content is None for {filename}, skipping write")
                                    continue

                                if config_text != "No change needed":
                                    # Save the modified config
                                    with open(config_path, "w") as f:
                                        f.write(config_text)
                                else:
                                    # Copy from fault directory
                                    try:
                                        fault_config_path = os.path.join(self.fault_dir, filename)
                                        if os.path.exists(fault_config_path):
                                            with open(fault_config_path, "r") as src:
                                                duplicate_config = src.read()
                                            with open(config_path, "w") as dest:
                                                dest.write(duplicate_config)
                                        else:
                                            logger.warning(f"Original config file not found: {fault_config_path}")
                                    except Exception as e:
                                        logger.warning(f"Error copying config file {filename}: {str(e)}")
                            
                        # Save metadata
                        metadata_path = os.path.join(self.fix_dir, "fix_metadata.json")
                        with open(metadata_path, "w") as f:
                            json.dump(individual_metadata, f, indent=2)
                            
                            # Detect if no changes were applied; skip Batfish to save time
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
                                processed_results = None
                                logger.info("No config changes detected; skipping network evaluation")
                            else:
                                # Save fixed configs to temporary directory to run network environment
                                fixed_tmp = self._save_tempdir(fixed_results)
                                
                                # Process with network environment to get fixed specs
                                if self.net_env is not None:
                                    processed_results = self._run_net_env(fixed_tmp.name)
                                else:
                                    processed_results = None
                                
                                # Evaluate results if possible
                                if (self.evaluator is not None and processed_results is not None):
                                    # Initialize evaluation dictionary
                                    evaluation = {}
                                    
                                    # Create a copy of kwargs for evaluation
                                    temp_eval_kwargs = self.evaluator_kwargs.copy()
                                    temp_eval_kwargs["specification_csv_path"] = self.specification_csv_path
                                    temp_eval_kwargs["reference_spec"] = None if self.specification_csv_path else self.original_specs
                                    
                                    # Optionally evaluate how different the fault is
                                    if self.faulty_specs is not None:
                                        temp_eval_kwargs_fault = temp_eval_kwargs.copy()
                                        temp_eval_kwargs_fault["compared_spec"] = self.faulty_specs
                                        evaluation["fault_evaluation"] = self.evaluator(**temp_eval_kwargs_fault)
                                    
                                    # Then update kwargs and evaluate the fix
                                    temp_eval_kwargs["compared_spec"] = processed_results
                                    evaluation["fix_evaluation"] = self.evaluator(**temp_eval_kwargs)
                                    
                                    # Log results
                                    fault_score = evaluation.get("fault_evaluation", {}).get("summary", {}).get("fix_rate", 0.0)
                                    fix_score = evaluation.get("fix_evaluation", {}).get("summary", {}).get("fix_rate", 0.0)
                                    
                                    logger.info(f"Task {id} fix similarity score: {fix_score:.4f}, "
                                                f"fault similarity score: {fault_score:.4f}")
                                    
                                else:
                                    logger.warning("Evaluator or processed results not available... \
                                                   Assigning fix similarity score: 0.0")
                                    evaluation = {
                                        "fault_evaluation": {},
                                        "fix_evaluation": {
                                            "summary": {"fix_rate": 0.0},
                                            "evaluation_failed": True,
                                            "reason": "Evaluator or processed results not available"
                                        }
                                    }
                                    fix_score = 0.0
                        
                        diagnosis_eval = self._diagnosis_judge_eval(
                            individual_metadata.get("problem_diagnosis"),
                            fallback_text=individual_metadata.get("raw_response")
                            or individual_metadata.get("parse_error"),
                        )
                        evaluation["diagnosis_evaluation"] = diagnosis_eval
                        diagnosis_score = diagnosis_eval.get("mean_score") \
                            if not diagnosis_eval.get("skipped") else None
                        diagnosis_completeness = diagnosis_eval.get("mean_completeness")
                        diagnosis_soundness = diagnosis_eval.get("mean_soundness")

                        # Save evaluation results
                        self._save_results(evaluation, "evaluation_results")

                        context = self._summarize_task_context(sample)
                        failure_mode = self._failure_mode_tag(parse_failed, evaluation, metadata=individual_metadata)
                        # Extract detailed failure information including fuzzy matching stats
                        failure_details = self._get_failure_details(metadata=individual_metadata)
                        raw_response = individual_metadata.get("raw_response")
                        parse_error_detail = individual_metadata.get("parse_error") or \
                            individual_metadata.get("format_error")
                        patch_fail_count = individual_metadata.get("patch_fail_count", 0)
                        routers_changed_proposed, loc_changed_proposed = \
                            self._proposed_edit_stats(self.faulty_configs, fixed_results if not parse_failed else None)
                        # Router identification accuracy (precision/recall/F1)
                        gt_changed = {
                            name for name, cfg in self.faulty_configs.items()
                            if self.original_configs.get(name) != cfg
                        }
                        pred_changed = set()
                        if not parse_failed:
                            for name, cfg in self.faulty_configs.items():
                                if fixed_results.get(name, cfg) != cfg:
                                    pred_changed.add(name)
                        tp = len(gt_changed & pred_changed)
                        fp = len(pred_changed - gt_changed)
                        fn = len(gt_changed - pred_changed)
                        precision = (
                            tp / len(pred_changed)
                            if len(pred_changed) > 0
                            else (1.0 if len(gt_changed) == 0 else 0.0)
                        )
                        recall = (
                            tp / len(gt_changed)
                            if len(gt_changed) > 0
                            else (1.0 if len(pred_changed) == 0 else 0.0)
                        )
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0.0
                        )
                        fixed_cnt, unfixed_cnt, broken_cnt, fix_ratio, regression_rate = \
                            self._extract_fix_stats(evaluation)

                        # Store result for this example
                        retrieval_meta = sample.get("retrieval_meta", {})
                        retrieval_tokens = retrieval_meta.get("token_count", {}).get("total_tokens", 0)
                        retrieval_prompt_tokens = retrieval_meta.get("token_count", {}).get("prompt_tokens", 0) + \
                            retrieval_meta.get("token_count", {}).get("system_tokens", 0)
                        retrieval_completion_tokens = retrieval_meta.get("token_count", {}).get("completion_tokens", 0)
                        retrieval_time = retrieval_meta.get("inference_time", 0)
                        fix_token_count = individual_metadata.get("token_count", {})
                        input_tokens_fix = fix_token_count.get("prompt_tokens", 0) + fix_token_count.get("system_tokens", 0)
                        output_tokens_fix = fix_token_count.get("completion_tokens", 0)
                        task_input_tokens = input_tokens_fix + retrieval_prompt_tokens
                        task_output_tokens = output_tokens_fix + retrieval_completion_tokens
                        result = {
                            "sample_id": id,
                            "batch_id": batch_num,
                            "best_score": fix_score,
                            "success": fix_score >= similarity_threshold,
                            "results_dir": str(current_task_dir),
                            "task_token_count": individual_metadata.get("token_count", {}).get("total_tokens", 0) + retrieval_tokens,
                            "task_prompt_tokens": individual_metadata.get("token_count", {}).get("prompt_tokens", 0) + retrieval_prompt_tokens,
                            "task_input_tokens": task_input_tokens,
                            "task_output_tokens": task_output_tokens,
                            "task_inference_time": inference_time / len(batch_dataset) + retrieval_time,
                            "parse_failures": 1 if parse_failed else 0,
                            "total_attempts": 1,
                            "failure_mode": failure_mode,
                            "failure_category_group": failure_details.get("failure_category_group", "unknown"),
                            "fuzzy_match_used": failure_details.get("fuzzy_match_used", False),
                            "fuzzy_match_count": failure_details.get("fuzzy_match_count", 0),
                            "match_strategies_used": failure_details.get("match_strategies_used", []),
                            "repair_attempted": failure_details.get("repair_attempted", False),
                            "repair_succeeded": failure_details.get("repair_succeeded", False),
                            "original_failure_mode": failure_details.get("original_failure_mode"),
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
                        
                        results.append(result)
                        logger.info(
                            f"Task {id} completed with score: {fix_score:.4f} "
                            f"(fixed={fixed_cnt}, unfixed={unfixed_cnt}, "
                            f"broken={broken_cnt}, fix_ratio={fix_ratio:.4f}, "
                            f"regression_rate={regression_rate if regression_rate is not None else 'n/a'})"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing task {sample_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        context = self._summarize_task_context(sample)
                        failure_mode = self._failure_mode_tag(False, None, e)
                        failure_details = self._get_failure_details(error=e)
                        raw_response = getattr(e, "response_content", None)
                        routers_changed_proposed, loc_changed_proposed = None, None

                        # Add failed result
                        result = {
                            "sample_id": sample_id,
                            "batch_id": batch_num,
                            "error": str(e),
                            "failure_mode": failure_mode,
                            "failure_category_group": failure_details.get("failure_category_group", "other"),
                            "fuzzy_match_used": False,
                            "fuzzy_match_count": 0,
                            "match_strategies_used": [],
                            "raw_response": raw_response,
                            "routers_changed_proposed": routers_changed_proposed,
                            "loc_changed_proposed": loc_changed_proposed,
                            **context,
                        }
                        results.append(result)
            
            except Exception as e:
                logger.error(f"Batch {batch_num} processing failed: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add failed results for all samples in this batch
                for sample_id, _ in batch_samples:
                    results.append({
                        "sample_id": sample_id,
                        "batch_id": batch_num,
                        "error": f"Batch processing failed: {str(e)}"
                    })
        
        # Reset save directory to base directory after processing all tasks
        self.save_dir = base_save_dir
        
        # Calculate aggregate statistics and save results
        return \
            self._save_metrics(
                results, 
                base_save_dir, 
                feedback_regime, 
                max_attempts, 
                similarity_threshold
            )

    def _save_incremental_result(
        self,
        result: Dict,
        base_save_dir: str,
        feedback_regime: bool,
        max_attempts: int,
        similarity_threshold: float
    ):
        """
        Save a single result incrementally to the results JSON file.
        
        This allows continuous updates during benchmark runs.
        
        Args:
            result (Dict): Single task result to save.
            base_save_dir (str): Base directory for saving results.
            feedback_regime (bool): Feedback control flag.
            max_attempts (int): Number of maximum retries.
            similarity_threshold (float): Target similarity score.
        """
        results_file = os.path.join(base_save_dir, "zero_shot_benchmark_results.json")
        
        # Load existing results
        existing_results = []
        existing_data = {}
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("results", [])
            except Exception as e:
                logger.warning(f"Could not load existing results for incremental save: {e}")
        
        # Find and update or append
        sample_id = result.get("sample_id")
        found = False
        for i, r in enumerate(existing_results):
            if r.get("sample_id") == sample_id:
                existing_results[i] = result
                found = True
                break
        
        if not found:
            existing_results.append(result)
        
        # Recompute stats
        stats = self._compute_stats(existing_results)
        
        # Calculate token statistics (exclude parse failures)
        valid_results = [r for r in existing_results if not r.get("parse_failures", 0)]
        results_with_tokens = [r for r in valid_results if "task_token_count" in r]
        results_with_input = [r for r in valid_results if "task_input_tokens" in r]
        results_with_output = [r for r in valid_results if "task_output_tokens" in r]
        
        # Write updated results
        with open(results_file, "w") as f:
            json.dump({
                "results": existing_results,
                "stats": stats,
                "run_info": {
                    "model_provider": self.model_provider,
                    "model_name": self.model_name,
                    "batch_api": self.batch_api,
                    "feedback_regime": feedback_regime,
                    "max_attempts": max_attempts,
                    "similarity_threshold": similarity_threshold,
                    "benchmark_token_count": sum(r.get("task_token_count", 0) for r in results_with_tokens),
                    "avg_token_count": sum(r.get("task_token_count", 0) for r in results_with_tokens) / 
                        len(results_with_tokens) if results_with_tokens else 0,
                    "benchmark_input_tokens": sum(r.get("task_input_tokens", 0) for r in results_with_input),
                    "avg_input_tokens": sum(r.get("task_input_tokens", 0) for r in results_with_input) / 
                        len(results_with_input) if results_with_input else 0,
                    "benchmark_output_tokens": sum(r.get("task_output_tokens", 0) for r in results_with_output),
                    "avg_output_tokens": sum(r.get("task_output_tokens", 0) for r in results_with_output) / 
                        len(results_with_output) if results_with_output else 0,
                    "last_updated": datetime.now().isoformat()
                }
            }, f, indent=2)
        
        logger.debug(f"Incrementally saved result for sample {sample_id}")

    def _single_eval(
        self,
        feedback_regime:bool,
        max_attempts: int,
        similarity_threshold: float,
        base_save_dir: str,
        continuous_save: bool = True
    ):
        """
        Handle one inference and evaluation at a time.
        
        Args:
            feedback_regime (bool): Feedback control flag.
                Defaults to None.
            max_attempts (int): Number of maximum retries.
                Defaults to 3.
            similarity_threshold (float): Target similarity score.
                Defaults to 1.0.
            base_save_dir (str): Base directory for saving results.
            continuous_save (bool): Whether to save results after each task.
                Defaults to True.
                
        Returns:
            Dictionary of computed statistics and results.
        """
        # Store results
        results = []
        
        # Iterate over all instruction prompt and output pair samples
        for id, sample in tqdm(self.eval_dataset.items(), desc="Processing examples"):

            # Extract task ID
            id = int(id.split("-")[-1])
            
            # Setup save path for this task using base directory
            current_task_dir = os.path.join(base_save_dir, f"Task_{id}")
            
            # Skip if task already completed (resume from checkpoint)
            if os.path.exists(os.path.join(current_task_dir, "evaluation_results.json")):
                logger.info(f"Skipping task {id} (already completed)")
                continue
            
            # Make a stop
            time.sleep(15)

            try:
                # Task ID info
                logger.info(f"Processing task {id}")

                # Extract sample data ingredients
                self.instruction = sample["instruction"]
                self.original_configs = sample["original_configs"]
                self.faulty_configs = sample["faulty_configs"]
                self.full_faulty_configs = sample.get("full_faulty_configs", self.faulty_configs)
                self.full_original_configs = sample.get("full_original_configs", self.original_configs)
                self.original_specs = sample.get("original_specs", [])
                self.specification_csv_path = sample.get("specification_csv_path")
                # self.faulty_specs = sample["faulty_specs"]

                self.save_dir = current_task_dir

                # Setup complementary paths for fix and fault operations
                self.fault_dir = os.path.join(current_task_dir, "fault")
                self.fix_dir = os.path.join(current_task_dir, "fix")
                
                os.makedirs(current_task_dir, exist_ok=True)
                os.makedirs(self.fault_dir, exist_ok=True)
                os.makedirs(self.fix_dir, exist_ok=True)

                # Sanity check to clear memory
                self.clear_memory()

                # Track best results for this example
                best_fixed_results = None
                best_processed_results = None
                best_evaluation = None
                best_score = 0.0
                best_metadata = None
                attempt_logs = []

                # Process with feedback loop
                self._save_results(self.instruction, "instruction_prompt")
                current_prompt = self.instruction

                for attempt in range(1, max_attempts + 1):
                    # Create attempt-specific directory if multiple attempts
                    if max_attempts > 1:
                        attempt_dir = os.path.join(self.fix_dir, f"attempt_{attempt}")
                        os.makedirs(attempt_dir, exist_ok=True)
                        
                        # Store original fix directory and temporarily set to attempt directory
                        original_fix_dir = self.fix_dir
                        self.fix_dir = attempt_dir

                    # Run inference
                    start_time = time.time()
                    parse_error_encountered = None
                    raw_response_error = None
                    
                    try:
                        # Run one-round model inference
                        fixed_results, metadata = self.inference(current_prompt, attempt)
                        parse_failed = False
                    except Exception as e:
                        parse_error_encountered = str(e)
                        raw_response_error = getattr(e, "response_content", None)
                        # Do not attempt format reinforcement; record failure and continue
                        fixed_results = {}
                        metadata = {
                            "problem_diagnosis": "Parsing failed",
                            "proposed_fix": "Unable to parse model response",
                            "parse_error": parse_error_encountered,
                            "raw_response": raw_response_error,
                        }
                        parse_failed = True
                    
                    inference_time = time.time() - start_time

                    # Reset fix directory if necessary
                    if max_attempts > 1:
                        self.fix_dir = original_fix_dir

                    # Handle failed parsing case
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
                        
                        # Save parse failure details
                        parse_fail_path = self._save_parse_failure(metadata)
                        
                        # Save the failed attempt metadata
                        suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                        self._save_results(evaluation, "evaluation_results", suffix)
                        
                        logger.info(f"Attempt {attempt} failed to parse... Assigning fix similarity score: 0.0 "
                                    f"(details: {parse_fail_path})")
                        
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
                            logger.info("No config changes detected; skipping network evaluation")
                        else:
                            # Save fixed configs to temporary directory to run environment
                            fixed_tmp = self._save_tempdir(fixed_results)

                            # Process with environment to get fixed specs
                            if self.net_env is not None:
                                processed_results = self._run_net_env(fixed_tmp.name)
                            else:
                                processed_results = None

                            # Evaluate results if possible
                            if (self.evaluator is not None 
                                and processed_results is not None):
                                # Initialize evaluation dictionary
                                evaluation = {}
                                
                                # Create a copy of kwargs for evaluation
                                temp_eval_kwargs = self.evaluator_kwargs.copy()
                                temp_eval_kwargs["specification_csv_path"] = self.specification_csv_path
                                temp_eval_kwargs["reference_spec"] = None if self.specification_csv_path else self.original_specs
                                
                                # Then update kwargs and evaluate the fix
                                temp_eval_kwargs["compared_spec"] = processed_results
                                evaluation["fix_evaluation"] = self.evaluator(**temp_eval_kwargs)
                                
                                # Save each attempt as a distinct logging file
                                suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                                self._save_results(evaluation, "evaluation_results", suffix)
                                
                                # Log attempt results
                                fix_score = evaluation.get("fix_evaluation", {}).get("summary", {}).get("fix_rate", 0.0)
                                
                                logger.info(f"Attempt {attempt} fix similarity score: {fix_score:.4f}")
                                
                            else:
                                logger.warning("Evaluator or processed results not available... \
                                               Assigning fix similarity score: 0.0")
                                # Create evaluation for pipeline stoppage
                                evaluation = {
                                    "fault_evaluation": {},
                                    "fix_evaluation": {
                                        "summary": {"fix_rate": 0.0},
                                        "evaluation_failed": True,
                                        "reason": "Evaluator or processed results not available"
                                    }
                                }
                                # Assign zero reward for failing tasks
                                fix_score = 0.0
                                
                                # Save the failed evaluation
                                suffix = f"_attempt_{attempt}" if max_attempts > 1 else ""
                                self._save_results(evaluation, "evaluation_results", suffix)

                    # Store attempt information
                    parse_error_value = None
                    if parse_failed:
                        parse_error_value = metadata.get("parse_error") or metadata.get("format_error")
                    else:
                        parse_error_value = parse_error_encountered
                    attempt_info = {
                        "attempt": attempt,
                        "fixed_results": fixed_results,
                        "processed_results": processed_results,
                        "fix_evaluation": evaluation["fix_evaluation"],
                        "metadata": metadata,
                        "parse_error": parse_error_value,
                        "raw_response_error": raw_response_error,
                        "token_count": metadata.get("token_count", {
                            "prompt_tokens": 0,
                            "system_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }),
                        "inference_time": inference_time,
                        "parse_failed": parse_failed,
                    }
                    attempt_logs.append(attempt_info)
                    
                    # Update best results
                    if fix_score >= best_score:
                        best_score = fix_score
                        best_fixed_results = fixed_results
                        best_processed_results = processed_results
                        best_evaluation = evaluation
                        best_metadata = metadata
                        logger.info(f"New best score: {fix_score:.4f}")

                        # Copy the best configs to the main fix directory
                        if max_attempts > 1 and not parse_failed:
                            for filename, content in fixed_results.items():
                                with open(os.path.join(self.fix_dir, filename), 'w') as f:
                                    f.write(content)
                
                    # Check if target similarity threshold is achieved
                    if fix_score >= similarity_threshold:
                        logger.info(f"Target similarity score reached: {fix_score:.4f}")
                        break

                    # Generate feedback prompt for next attempt if needed
                    if (attempt < max_attempts 
                        and feedback_regime 
                        and "fix_evaluation" in evaluation
                        and not parse_failed):
                        current_prompt = _feedback_prompt(
                            model_name=self.model_name,
                            attempt=attempt_info["attempt"],
                            fixed_specs=attempt_info["processed_results"],
                            fix_evaluation=attempt_info["fix_evaluation"],
                            fix_metadata=attempt_info["metadata"],
                        )
                
                # Save attempt logs if multiple attempts
                if (max_attempts > 1 and attempt_logs):
                    # Convert to serializable format
                    serializable_logs = []
                    for log in attempt_logs:
                        serializable_log = {
                            "attempt": log["attempt"],
                            "fix_evaluation": log["fix_evaluation"],
                            "metadata": log["metadata"],
                            "parse_failed": log["parse_failed"],
                        }
                        serializable_logs.append(serializable_log)
                    
                    # Add evaluation of how different the faulty data is
                    if "evaluation" in locals() and "fault_evaluation" in evaluation:
                        serializable_logs_extended = {
                            "fix_attempts_evaluation": serializable_logs,
                            "fault_evaluation": evaluation["fault_evaluation"]
                        }
                        self._save_results(serializable_logs_extended, "attempt_logs")
                
                parse_failures = sum(1 for log in attempt_logs if log.get("parse_failed", False))
                total_attempts = len(attempt_logs)
                context = self._summarize_task_context(sample)
                parse_failure_for_mode = parse_failures == total_attempts and total_attempts > 0
                failure_mode = self._failure_mode_tag(
                    parse_failure_for_mode, best_evaluation, metadata=best_metadata
                )
                # Extract detailed failure information including fuzzy matching stats
                failure_details = self._get_failure_details(metadata=best_metadata)
                diagnosis_eval = self._diagnosis_judge_eval(
                    (best_metadata or {}).get("problem_diagnosis"),
                    fallback_text=(best_metadata or {}).get("raw_response")
                    or parse_error_detail,
                )
                if best_evaluation is not None:
                    best_evaluation["diagnosis_evaluation"] = diagnosis_eval
                    # Save evaluation_results later with run_info after token stats are computed
                diagnosis_score = diagnosis_eval.get("mean_score") \
                    if not diagnosis_eval.get("skipped") else None
                diagnosis_completeness = diagnosis_eval.get("mean_completeness")
                diagnosis_soundness = diagnosis_eval.get("mean_soundness")
                raw_response = next(
                    (log.get("metadata", {}).get("raw_response") 
                        for log in reversed(attempt_logs) 
                        if log.get("metadata", {}).get("raw_response")),
                    None
                )
                parse_error_detail = next(
                    (log.get("metadata", {}).get("parse_error") 
                        for log in reversed(attempt_logs) 
                        if log.get("metadata", {}).get("parse_error")),
                    None
                )
                # Capture parse errors even if later attempts succeeded
                if not parse_error_detail:
                    parse_error_detail = next(
                        (log.get("parse_error") for log in reversed(attempt_logs) if log.get("parse_error")),
                        None
                    )
                if failure_mode == "none" and parse_error_detail:
                    failure_mode = "parse_retry"

                patch_fail_count = 0
                if best_metadata and isinstance(best_metadata, dict):
                    patch_fail_count = best_metadata.get("patch_fail_count", 0)
                routers_changed_proposed, loc_changed_proposed = \
                    self._proposed_edit_stats(self.faulty_configs, best_fixed_results)
                task_prompt_tokens = sum(
                    log.get("token_count", {}).get("prompt_tokens", 0) for log in attempt_logs
                )
                retrieval_meta = sample.get("retrieval_meta", {})
                retrieval_tokens = retrieval_meta.get("token_count", {}).get("total_tokens", 0)
                retrieval_prompt_tokens = retrieval_meta.get("token_count", {}).get("prompt_tokens", 0) + \
                    retrieval_meta.get("token_count", {}).get("system_tokens", 0)
                retrieval_completion_tokens = retrieval_meta.get("token_count", {}).get("completion_tokens", 0)
                retrieval_time = retrieval_meta.get("inference_time", 0)
                # Router identification metrics and counts
                gt_changed = {
                    name for name, cfg in self.faulty_configs.items()
                    if self.original_configs.get(name) != cfg
                }
                pred_changed = set()
                if best_fixed_results:
                    for name, cfg in self.faulty_configs.items():
                        if best_fixed_results.get(name, cfg) != cfg:
                            pred_changed.add(name)
                tp = len(gt_changed & pred_changed)
                fp = len(pred_changed - gt_changed)
                fn = len(gt_changed - pred_changed)
                if best_fixed_results:
                    precision = (
                        tp / len(pred_changed)
                        if len(pred_changed) > 0
                        else (1.0 if len(gt_changed) == 0 else 0.0)
                    )
                    recall = (
                        tp / len(gt_changed)
                        if len(gt_changed) > 0
                        else (1.0 if len(pred_changed) == 0 else 0.0)
                    )
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )
                else:
                    precision = recall = f1 = None

                fixed_cnt, unfixed_cnt, broken_cnt, fix_ratio, regression_rate = \
                    self._extract_fix_stats(best_evaluation if best_evaluation else {})
                input_tokens_fix = sum(
                    (log.get("token_count", {}).get("prompt_tokens", 0) + log.get("token_count", {}).get("system_tokens", 0))
                    for log in attempt_logs
                )
                output_tokens_fix = sum(
                    log.get("token_count", {}).get("completion_tokens", 0)
                    for log in attempt_logs
                )
                task_input_tokens = input_tokens_fix + retrieval_prompt_tokens
                task_output_tokens = output_tokens_fix + retrieval_completion_tokens

                # Store result for this example
                result = {
                    "sample_id": id,
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
                    "failure_category_group": failure_details.get("failure_category_group", "unknown"),
                    "fuzzy_match_used": failure_details.get("fuzzy_match_used", False),
                    "fuzzy_match_count": failure_details.get("fuzzy_match_count", 0),
                    "match_strategies_used": failure_details.get("match_strategies_used", []),
                    "repair_attempted": failure_details.get("repair_attempted", False),
                    "repair_succeeded": failure_details.get("repair_succeeded", False),
                    "original_failure_mode": failure_details.get("original_failure_mode"),
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
                
                # Save evaluation_results.json with run_info for this task
                if best_evaluation is not None:
                    task_run_info = {
                        "model_provider": self.model_provider,
                        "model_name": self.model_name,
                        "feedback_regime": feedback_regime,
                        "max_attempts": max_attempts,
                        "similarity_threshold": similarity_threshold,
                        "task_token_count": result["task_token_count"],
                        "task_input_tokens": result["task_input_tokens"],
                        "task_output_tokens": result["task_output_tokens"],
                        "task_inference_time": result["task_inference_time"],
                        "parse_failures": result["parse_failures"],
                        "total_attempts": result["total_attempts"],
                        "timestamp": datetime.now().isoformat()
                    }
                    self._save_results(best_evaluation, "evaluation_results", run_info=task_run_info)
                
                results.append(result)
                logger.info(
                    f"Task {id} completed with score: {best_score:.4f} "
                    f"(fixed={fixed_cnt}, unfixed={unfixed_cnt}, "
                    f"broken={broken_cnt}, fix_ratio={fix_ratio:.4f}, "
                    f"regression_rate={regression_rate if regression_rate is not None else 'n/a'})"
                )
                
                # Save incrementally if enabled
                if continuous_save:
                    self._save_incremental_result(
                        result, base_save_dir, feedback_regime, 
                        max_attempts, similarity_threshold
                    )
                
                # Clear memory between examples
                self.clear_memory()
                
            except Exception as e:
                logger.error(f"Error processing task {id}: {str(e)}")
                import traceback
                traceback.print_exc()

                context = self._summarize_task_context(sample)
                failure_mode = self._failure_mode_tag(False, None, e)
                failure_details = self._get_failure_details(error=e)
                raw_response = getattr(e, "response_content", None)
                routers_changed_proposed, loc_changed_proposed = None, None

                # Capture whatever token info is available from completed attempts
                task_input_tokens = sum(
                    (log.get("token_count", {}).get("prompt_tokens", 0)
                     + log.get("token_count", {}).get("system_tokens", 0))
                    for log in attempt_logs
                ) if attempt_logs else 0
                task_output_tokens = sum(
                    log.get("token_count", {}).get("completion_tokens", 0)
                    for log in attempt_logs
                ) if attempt_logs else 0
                task_inference_time = sum(
                    log.get("inference_time", 0)
                    for log in attempt_logs
                ) if attempt_logs else 0
                
                # Add failed result
                result = {
                    "sample_id": id,
                    "error": str(e),
                    "failure_mode": failure_mode,
                    "failure_category_group": failure_details.get("failure_category_group", "other"),
                    "fuzzy_match_used": False,
                    "fuzzy_match_count": 0,
                    "match_strategies_used": [],
                    "raw_response": raw_response,
                    "routers_changed_proposed": routers_changed_proposed,
                    "loc_changed_proposed": loc_changed_proposed,
                    "task_input_tokens": task_input_tokens,
                    "task_output_tokens": task_output_tokens,
                    "task_token_count": task_input_tokens + task_output_tokens,
                    "task_inference_time": task_inference_time,
                    "parse_failures": len(attempt_logs) if attempt_logs else 1,
                    "total_attempts": len(attempt_logs) if attempt_logs else 0,
                    **context,
                }
                results.append(result)

                # Save incrementally if enabled
                if continuous_save:
                    self._save_incremental_result(
                        result, base_save_dir, feedback_regime, 
                        max_attempts, similarity_threshold
                    )
        
        # Reset save directory to base directory after processing all tasks
        self.save_dir = base_save_dir
        
        # Calculate aggregate statistics and save all results
        return \
            self._save_metrics(
                results, 
                base_save_dir, 
                feedback_regime, 
                max_attempts, 
                similarity_threshold
            )

    # ====================================================================== #
    #                             Private Helpers                            #
    # ====================================================================== #
    def _save_metrics(
        self, 
        results: List[Dict], 
        base_save_dir: str,
        feedback_regime,
        max_attempts: int,
        similarity_threshold: float
    ):
        """
        Calculate and save all metrics and statistics.
        
        Args:
            results (List[Dict]): List of evaluation results.
            base_save_dir (str): Base directory for saving results.
            feedback_regime (bool): Feedback control flag.
            max_attempts (int): Number of maximum retries.
            similarity_threshold (float): Target similarity score.
            
        Returns:
            Dictionary of computed statistics.
        """ 
        # Save results using base directory
        results_file = os.path.join(base_save_dir, "zero_shot_benchmark_results.json")
        
        # Load existing results if resuming from a previous run
        existing_results = []
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("results", [])
                    logger.info(f"Loaded {len(existing_results)} existing results from previous run")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        
        # Merge results: keep existing + add new ones (avoid duplicates by sample_id)
        existing_ids = {r.get("sample_id") for r in existing_results if "sample_id" in r}
        new_results = [r for r in results if r.get("sample_id") not in existing_ids]
        all_results = existing_results + new_results
        
        logger.info(f"Total results: {len(all_results)} (existing: {len(existing_results)}, new: {len(new_results)})")
        
        # Calculate aggregate statistics on all results
        stats = self._compute_stats(all_results)
        
        # Calculate token statistics (exclude parse failures)
        valid_results = [r for r in all_results if not r.get("parse_failures", 0)]
        results_with_tokens = [r for r in valid_results if "task_token_count" in r]
        results_with_input = [r for r in valid_results if "task_input_tokens" in r]
        results_with_output = [r for r in valid_results if "task_output_tokens" in r]
        
        with open(results_file, "w") as f:
            json.dump({
                "results": all_results,
                "stats": stats,
                "run_info": {
                    "model_provider": self.model_provider,
                    "model_name": self.model_name,
                    "batch_api": self.batch_api,
                    "feedback_regime": feedback_regime,
                    "max_attempts": max_attempts,
                    "similarity_threshold": similarity_threshold,
                    "benchmark_token_count": sum(r.get("task_token_count", 0) for r in results_with_tokens),
                    "avg_token_count": sum(r.get("task_token_count", 0) for r in results_with_tokens) / 
                        len(results_with_tokens) if results_with_tokens else 0,
                    "benchmark_input_tokens": sum(r.get("task_input_tokens", 0) for r in results_with_input),
                    "avg_input_tokens": sum(r.get("task_input_tokens", 0) for r in results_with_input) / 
                        len(results_with_input) if results_with_input else 0,
                    "benchmark_output_tokens": sum(r.get("task_output_tokens", 0) for r in results_with_output),
                    "avg_output_tokens": sum(r.get("task_output_tokens", 0) for r in results_with_output) / 
                        len(results_with_output) if results_with_output else 0,
                    "benchmark_inference_time": sum(r.get("task_inference_time", 0) for r in results if "task_inference_time" in r),
                    "avg_inference_time": sum(r.get("task_inference_time", 0) for r in results if "task_inference_time" in r) / 
                        len([r for r in results if "task_inference_time" in r]) if results else 0,
                    "timestamp": datetime.now().isoformat()
                }
            }, f, indent=2)
        
        logger.info(f"\nZero-Shot benchmark completed. Success rate: {stats['success_rate']*100:.2f}%")
        return stats
        
    def _compute_stats(self, results: List[Dict]):
        """
        Compute aggregate statistics from evaluation results.

        Args:
            results (List[Dict]): List of evaluation results.
            
        Returns:
            Dictionary of computed statistics.
        """
        # Count successful examples
        successful = [r for r in results if r.get("success", False)]
        
        # Calculate success rate
        success_rate = len(successful) / len(results) if results else 0
        
        # Calculate average score
        scores = [r.get("best_score", 0) for r in results if "best_score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Router identification metrics (macro average over tasks that have values)
        precs = [r["router_precision"] for r in results if r.get("router_precision") is not None]
        recs = [r["router_recall"] for r in results if r.get("router_recall") is not None]
        f1s = [r["router_f1"] for r in results if r.get("router_f1") is not None]
        avg_router_precision = sum(precs) / len(precs) if precs else None
        avg_router_recall = sum(recs) / len(recs) if recs else None
        avg_router_f1 = sum(f1s) / len(f1s) if f1s else None
        regression_rates = [r["regression_rate"] for r in results if r.get("regression_rate") is not None]
        avg_regression_rate = sum(regression_rates) / len(regression_rates) if regression_rates else None

        diagnosis_scores = [r["diagnosis_score"] for r in results if r.get("diagnosis_score") is not None]
        diagnosis_comps = [r["diagnosis_completeness"] for r in results if r.get("diagnosis_completeness") is not None]
        diagnosis_sounds = [r["diagnosis_soundness"] for r in results if r.get("diagnosis_soundness") is not None]
        avg_diagnosis_score = sum(diagnosis_scores) / len(diagnosis_scores) if diagnosis_scores else None
        avg_diagnosis_completeness = sum(diagnosis_comps) / len(diagnosis_comps) if diagnosis_comps else None
        avg_diagnosis_soundness = sum(diagnosis_sounds) / len(diagnosis_sounds) if diagnosis_sounds else None
        
        # Calculate parse failure statistics
        total_parse_failures = sum(r.get("parse_failures", 0) for r in results if "parse_failures" in r)
        total_attempts = sum(r.get("total_attempts", 0) for r in results if "total_attempts" in r)
        
        # Calculate batch-specific statistics
        batch_stats = {}
        if self.batch_api:
            batch_stats = {
                "processing_mode": "batch",
                "total_samples": len(results),
                "avg_processing_time_per_sample": sum(r.get("task_inference_time", 0) \
                    for r in results) / len(results) if results else 0
            }
        else:
            batch_stats = {
                "processing_mode": "single",
                "avg_attempts_per_sample": total_attempts / len(results) if results else 0
            }
        
        return {
            "total": len(results),
            "successful": len(successful),
            "success_rate": success_rate,
            "average_score": avg_score,
            "total_parse_failures": total_parse_failures,
            "total_attempts": total_attempts,
            "avg_router_precision": avg_router_precision,
            "avg_router_recall": avg_router_recall,
            "avg_router_f1": avg_router_f1,
            "avg_regression_rate": avg_regression_rate,
            "avg_diagnosis_score": avg_diagnosis_score,
            "avg_diagnosis_completeness": avg_diagnosis_completeness,
            "avg_diagnosis_soundness": avg_diagnosis_soundness,
            **batch_stats
        }
    
    def _save_results(
        self, 
        results: Dict[str, Any],
        filename: str = None,
        suffix: Optional[str] = "",
        run_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Evaluation results to save.
            filename (str): Unique name to save the file.
                Defaults to None.
            suffix (Optional[str]): Optional suffix to append to filename.
                Defaults to empty string ''.
            run_info (Optional[Dict[str, Any]]): Run statistics to include.
                Defaults to None.
        """
        # Create output path
        if filename is None:
            logger.error("Provide a correct filename!")

        output_path = os.path.join(self.save_dir, f"{filename}{suffix}.json")
        
        try:
            # Create directory if it does not exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build output data with optional run_info
            # Handle both dict and non-dict results
            if isinstance(results, dict):
                output_data = results.copy()
                if run_info is not None:
                    output_data["run_info"] = run_info
            else:
                # For non-dict results (e.g., strings), wrap if run_info provided
                if run_info is not None:
                    output_data = {"content": results, "run_info": run_info}
                else:
                    output_data = results
            
            # Write results to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Evaluation results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")

    def token_counter(
        self, 
        text: str
    ) -> int:
        """
        Count the number of tokens in a text string using 'cl100k_base' encoding.
        
        Args:
            text (str): Text to tokenize.
            
        Returns:
            Number of tokens.
        """
        try:
            # Use base encoding for all models
            encoding = tiktoken.get_encoding("cl100k_base")
                
            # Count tokens
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Failed to count tokens: {str(e)}")
            return 0
        
    @staticmethod
    def _parse_content(
        content: str,
        faulty_configs: Dict[str, str],
        parser_func = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Common method to parse response content into configs and metadata.
        
        Args:
            content (str): Raw response content string.
            faulty_configs (Dict[str, str]): Original faulty configurations.
            parser_func: Parser function to use.
                Defaults to None.
            
        Returns:
            Tuple of parsed configs and metadata.
        """
        if parser_func is None:
            raise ValueError("Parser function must be provided")
        
        # Use the provided parser function
        configs, metadata = parser_func(content, faulty_configs)
        if isinstance(metadata, dict):
            metadata.setdefault("raw_response", content)
        return configs, metadata

    @staticmethod
    def _parse_single_response(
        response: AIMessage,
        faulty_configs: Dict[str, str],
        fault_dir: str,
        fix_dir: str,
        parser_func = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Parse single response and save files to disk.

        Args:
            response (AIMessage): Extracted model response.
            faulty_configs (Dict[str, str]): Original faulty configurations.
            fault_dir (str): Directory where data from faulty network is stored.
            fix_dir (str): Directory where data from fixed network is stored.
            parser_func: Parser function to use.
                Defaults to None.

        Returns:
            Tuple of parsed configs and metadata.
        """
        # Get response content
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Use specified parser
        all_configs, fix_metadata = ZeroShot._parse_content(
            content, 
            faulty_configs, 
            parser_func
        )
        
        # Save files
        os.makedirs(fix_dir, exist_ok=True)
        os.makedirs(fault_dir, exist_ok=True)
        
        # Save all faulty configs to fault directory first
        for filename, content in faulty_configs.items():
            fault_config_path = os.path.join(fault_dir, filename)
            with open(fault_config_path, "w") as f:
                f.write(content)
        
        # Note: Metadata is NOT saved here - it will be saved by the caller
        # after adding token counts and other runtime information
        
        # Save config files
        for filename, config_text in all_configs.items():                
            config_path = os.path.join(fix_dir, filename)
            
            if config_text != "No change needed":
                with open(config_path, "w") as f:
                    f.write(config_text)
            else:
                try:
                    fault_config_path = os.path.join(fault_dir, filename)
                    if os.path.exists(fault_config_path):
                        with open(fault_config_path, "r") as src:
                            duplicate_config = src.read()
                        with open(config_path, "w") as dest:
                            dest.write(duplicate_config)
                        all_configs[filename] = duplicate_config
                    else:
                        logger.warning(f"Faulty config file not found: {fault_config_path}")
                except Exception as e:
                    logger.warning(f"Error copying config file {filename}: {str(e)}")
        
        return all_configs, fix_metadata

    @staticmethod
    def _parse_batch_response(
        content: str,
        faulty_configs: Dict[str, str],
        parser_func = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Parse batch response content without saving files.
        
        Args:
            content (str): Raw response content string.
            faulty_configs (Dict[str, str]): Original faulty configurations.
            parser_func: Parser function to use.
            
        Returns:
            Tuple of parsed configs and metadata.
        """
        # Use common parsing method with specified parser
        return ZeroShot._parse_content(
            content, 
            faulty_configs, 
            parser_func
        )
    
    @staticmethod  
    def _save_tempdir(configs: Dict[str, str]) -> tempfile.TemporaryDirectory:
        """
        Save config files to a temporary directory.

        Args:
            configs (Dict[str, str]): Configuration files.
        
        Returns:
            Temporary directory object.
        """
        tempdir = tempfile.TemporaryDirectory()

        # Required for Batfish
        config_dir = os.path.join(tempdir.name, "configs")
        os.makedirs(config_dir, exist_ok=True)

        for filename, content in configs.items():
            # Ensure .cfg extension
            if not filename.endswith(".cfg"):
                filename = f"{filename}.cfg"

            config_file = os.path.join(config_dir, filename)
            with open(config_file, "w") as f:
                f.write(content)

        return tempdir

    @staticmethod
    def _fault_metrics_from_csv(
        specification_csv_path: Optional[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Infer number of applied faults and LOC changed (ground truth) from 'fault_metrics.csv'.

        Args:
            specification_csv_path (Optional[str]): Path to the csv file of network specs.

        Returns:
            Tuple containing number of applied faults and LOC changed.
        """
        if not specification_csv_path:
            return None, None
        try:
            base_dir = Path(specification_csv_path).resolve().parent.parent
            metrics_path = base_dir / "data_and_metrics" / "fault_metrics.csv"
            if not metrics_path.exists():
                return None, None
            with metrics_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                count = 0
                loc_changed = 0
                for row in reader:
                    if not row:
                        continue
                    status = (row.get("status") or "").lower()
                    if status and status != "applied":
                        continue
                    count += 1
                    try:
                        added = int(row.get("config_lines_added", 0))
                        removed = int(row.get("config_lines_removed", 0))
                        loc_changed += added + removed
                    except Exception:
                        pass
                return count, loc_changed if loc_changed > 0 else None
        except Exception:
            return None, None
    
    @staticmethod
    def _summarize_task_context(
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect lightweight context about a task for logging purposes.

        Args:
            sample (Dict[str, Any]): Single data sample.
        
        Returns:
            Dictionary of info about fault impact.
        """
        full_faulty_configs = sample.get("full_faulty_configs") or sample.get("faulty_configs", {}) or {}
        full_original_configs = sample.get("full_original_configs") or sample.get("original_configs", {}) or {}
        faulty_configs = sample.get("faulty_configs", {}) or {}

        node_count_total = len(full_faulty_configs)
        node_count_retrieved = len(faulty_configs)
        changed_routers_total = sum(
            1 for name, cfg in full_faulty_configs.items()
            if full_original_configs.get(name) != cfg
        )
        changed_routers_retrieved = sum(
            1 for name, cfg in faulty_configs.items()
            if full_original_configs.get(name) != cfg
        )

        fault_metadata = sample.get("fault_metadata", None)
        fault_count = len(fault_metadata) if isinstance(fault_metadata, list) else None
        loc_changed_ground_truth = None
        if fault_count is None:
            fault_count, loc_changed_ground_truth = ZeroShot._fault_metrics_from_csv(
                sample.get("specification_csv_path")
            )
        if fault_count is None:
            fault_count = changed_routers_total
        lines_of_code_total = sum(len((cfg or "").splitlines()) for cfg in full_faulty_configs.values())
        lines_of_code_retrieved = sum(len((cfg or "").splitlines()) for cfg in faulty_configs.values())
        return {
            "node_count": node_count_total,
            "node_count_retrieved": node_count_retrieved,
            "fault_count": fault_count,
            "changed_routers": changed_routers_total,
            "changed_routers_retrieved": changed_routers_retrieved,
            "loc_changed_ground_truth": loc_changed_ground_truth,
            "lines_of_code_total": lines_of_code_total,
            "lines_of_code_retrieved": lines_of_code_retrieved,
        }

    @staticmethod
    def _proposed_edit_stats(
        faulty_configs: Dict[str, str],
        fixed_configs: Optional[Dict[str, str]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute how many routers changed and LOC delta in the proposed fix vs faulty configs.

        Args:
            faulty_configs (Dict[str, str]): Dictionary of faulty configurations.
            fixed_configs (Optional[Dict[str, str]]): Dictionary of fixed configurations.

        Returns:
            Tuple of number of changed routers and LOC delta.
        """
        if fixed_configs is None:
            return None, None
        routers_changed = 0
        loc_changed = 0
        for filename, faulty in faulty_configs.items():
            fixed = fixed_configs.get(filename, faulty)
            if fixed != faulty:
                routers_changed += 1
                diff = difflib.ndiff(faulty.splitlines(), fixed.splitlines())
                loc_changed += sum(1 for line in diff if line.startswith("+ ") or line.startswith("- "))
        return routers_changed, loc_changed

    def _diagnosis_judge_eval(
        self,
        diagnosis_text: Optional[str],
        *,
        fallback_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run LLM judge scoring even when parsing fails. Uses 'diagnosis_text' when available; 
        falls back to 'fallback_text' (raw response or parse error) so judges still run on parse failures.

        Args:
            diagnosis_text (Optional[str]): Diagnosis text.
            fallback_text (Optional[str]): Fallback text.
                Defaults to None.
        """
        if not self.diagnosis_judge_config.get("enabled", False):
            return {"skipped": True, "reason": "disabled"}

        candidate = diagnosis_text or fallback_text or ""
        # If the parser left a placeholder like "Parsing failed", still run but
        # append any fallback text to give the judges more context.
        if candidate.strip().lower() in {"parsing failed", "parsing error", "unable to parse model response"}:
            if fallback_text:
                candidate = f"{candidate}\nRaw response / error:\n{fallback_text}"

        metrics_path = infer_fault_metrics_path(self.specification_csv_path)
        original_cfgs = getattr(self, "full_original_configs", None) or getattr(self, "original_configs", {}) or {}
        faulty_cfgs = getattr(self, "full_faulty_configs", None) or getattr(self, "faulty_configs", {}) or {}

        return evaluate_diagnosis_with_llm_judges(
            diagnosis=candidate,
            fault_metrics_path=metrics_path,
            original_configs=original_cfgs,
            faulty_configs=faulty_cfgs,
            judge_model_configs=self.diagnosis_judge_config.get("models"),
            max_fault_rows=self.diagnosis_judge_config.get("max_fault_rows", 8),
            max_diff_lines=self.diagnosis_judge_config.get("max_diff_lines", 400),
        )

    def _run_net_env(
        self,
        scenario_path: str,
        retries: int = 3,
        delay: float = 5.0,
        timeout: int = 300,
    ):
        """
        Run network environment with timeout, retry, and backoff to handle transient failures.
        
        Args:
            scenario_path (str): Path to the network scenario.
            retries (int): Number of retry attempts.
                Defaults to 3.
            delay (float): Delay between retries in seconds.
                Defaults to 5.0.
            timeout (int): Timeout in seconds for each attempt.
                Defaults to 300.
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Batfish processing timed out after {timeout} seconds")
        
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                # Set up timeout for this attempt
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                
                result = self.net_env(local_scenario_path=scenario_path)
                
                # Cancel the alarm on success
                signal.alarm(0)
                return result
                
            except TimeoutError as e:
                signal.alarm(0)  # Cancel alarm
                last_error = e
                logger.warning(
                    f"Network environment attempt {attempt}/{retries} timed out after {timeout}s"
                )
                if attempt < retries:
                    time.sleep(delay)
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                last_error = e
                logger.warning(
                    f"Network environment attempt {attempt}/{retries} failed: {str(e)}"
                )
                if attempt < retries:
                    time.sleep(delay)
        logger.warning("Network environment failed after retries; skipping evaluation")
        return None

    @staticmethod
    def _failure_mode_tag(
        parse_failed: bool,
        evaluation: Dict[str, Any] = None,
        error: Exception = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a concise tag describing why a task failed.
        Uses granular failure categories from parser errors.
        
        Args:
            parse_failed (bool): Boolean flag indicating parse fail.
            evaluation (Dict[str, Any]): Evaluation results.
                Defaults to None.
            error (Exception): Type of error occured.
                Defaults to None.
            metadata (Dict[str, Any]): Metadata from model execution.
                Defaults to None.

        Returns one of:
            - "none" - Success
            - "parse_retry" - Parsing failed but retry succeeded
            - FailureCategory value (e.g., "yaml_syntax", "search_not_found", etc.)
            - "exception:<ExceptionClassName>" - For unclassified exceptions
        """
        # Check for specific parser errors first
        if error is not None:
            if isinstance(error, ParserError):
                return error.category.value
            # Classify legacy ValueError messages
            category = classify_error(error)
            if category != FailureCategory.UNKNOWN:
                return category.value
            return f"exception:{error.__class__.__name__}"
        
        # Check parse_error in metadata for category info
        if metadata and "parse_error" in metadata:
            parse_error_str = str(metadata.get("parse_error", ""))
            err_lower = parse_error_str.lower()
            
            # Check for replacement/search errors FIRST (before YAML checks)
            # since repair errors wrap the message
            if "invalid replacement" in err_lower or "missing 'replace'" in err_lower:
                return FailureCategory.REPLACEMENT_INVALID.value
            if "search block" in err_lower or "not found" in err_lower:
                if "multiple" in err_lower:
                    return FailureCategory.SEARCH_MULTIPLE.value
                return FailureCategory.SEARCH_NOT_FOUND.value
            if "empty search" in err_lower or "empty block" in err_lower:
                return FailureCategory.SEARCH_EMPTY.value
            
            # YAML errors
            if "yaml" in err_lower:
                if "empty" in err_lower:
                    return FailureCategory.YAML_EMPTY.value
                if "structure" in err_lower or "expected" in err_lower:
                    return FailureCategory.YAML_STRUCTURE.value
                return FailureCategory.YAML_SYNTAX.value
            
            # Config errors
            if "missing" in err_lower and "config" in err_lower:
                return FailureCategory.MISSING_CONFIG.value
            
            # API errors
            if "api" in err_lower or "rate limit" in err_lower or "quota" in err_lower:
                if "timeout" in err_lower:
                    return FailureCategory.API_TIMEOUT.value
                if "quota" in err_lower or "rate limit" in err_lower:
                    return FailureCategory.API_QUOTA.value
                return FailureCategory.API_ERROR.value
            
            # Context overflow
            if "context" in err_lower and ("overflow" in err_lower or "too long" in err_lower or "length" in err_lower):
                return FailureCategory.CONTEXT_OVERFLOW.value
        
        if parse_failed:
            return FailureCategory.YAML_SYNTAX.value
        
        fix_eval = (evaluation or {}).get("fix_evaluation", {})
        if fix_eval.get("parse_failed"):
            return FailureCategory.YAML_SYNTAX.value
        if fix_eval.get("evaluation_failed"):
            return fix_eval.get("reason", "evaluation_failed")
        
        return FailureCategory.NONE.value
    
    @staticmethod
    def _get_failure_details(
        error: Exception = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract detailed failure information for logging.

        Args:
            error (Exception): Type of error occured.
                Defaults to None.
            metadata (Dict[str, Any]): Metadata from model execution.
                Defaults to None.
        
        Returns:
            Dictionary with failure details including category, group, and specifics.
        """
        details = {
            "failure_category": FailureCategory.NONE.value,
            "failure_category_group": "success",
            "fuzzy_match_used": False,
            "fuzzy_match_count": 0,
            "match_strategies_used": [],
            "repair_attempted": False,
            "repair_succeeded": False,
            "original_failure_mode": None,
        }
        
        if metadata:
            details["fuzzy_match_used"] = metadata.get("fuzzy_match_used", False)
            details["fuzzy_match_count"] = metadata.get("replacements_fuzzy_applied", 0)
            details["match_strategies_used"] = metadata.get("match_strategies_used", [])
            # Extract repair information if present (from ZeroShotRepair)
            details["repair_attempted"] = metadata.get("repair_attempted", False)
            details["repair_succeeded"] = metadata.get("repair_succeeded", False)
            details["original_failure_mode"] = metadata.get("original_failure_mode")
            # Also check repair_context for backward compatibility
            repair_context = metadata.get("repair_context", {})
            if repair_context and not details["original_failure_mode"]:
                details["original_failure_mode"] = repair_context.get("original_failure_category")
        
        if error is not None:
            if isinstance(error, ParserError):
                details["failure_category"] = error.category.value
                details["failure_category_group"] = _get_failure_group(error.category)
                details["error_details"] = error.details
            else:
                category = classify_error(error)
                details["failure_category"] = category.value
                details["failure_category_group"] = _get_failure_group(category)
        
        return details

    @staticmethod
    def _extract_fix_stats(
        evaluation: Dict[str, Any]
    ) -> Tuple[int, int, int, float, Optional[float]]:
        """
        Extract fixed/not-fixed/broken counts, fix ratio, and regression rate
        (side_effects / (fixed + not_fixed + side_effects)) from an evaluation dict.

        Args:
            evaluation (Dict[str, Any]): Evaluation results.
        
        Returns:
            fixed, not_fixed, broken, ratio (fixed / (fixed + not_fixed)), regression_rate
        """
        summary = evaluation.get("fix_evaluation", {}).get("summary", {}) if evaluation else {}
        fixed = summary.get("fixed", 0)
        not_fixed = summary.get("not_fixed", 0)
        side_effects = summary.get("side_effects", summary.get("broken", 0))
        broken = side_effects
        denom = fixed + not_fixed
        ratio = fixed / denom if denom else 0.0
        total_with_side = fixed + not_fixed + side_effects
        regression_rate = side_effects / total_with_side if total_with_side else None
        return fixed, not_fixed, broken, ratio, regression_rate

    def _save_parse_failure(
        self, 
        metadata: Dict[str, Any], 
        filename: str = "parse_failed.json"
    ) -> str:
        """
        Persist parse failure metadata and return the saved path.

        Args:
            metadata (Dict[str, Any]): Metadata from model execution.
            filename (str): Filename for the parse failure file.
                Defaults to 'parse_failed.json'.
        """
        path = os.path.join(self.fix_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Parse failure details saved to: {path}")
        except Exception as e:
            logger.error(f"Could not save parse failure details to {path}: {str(e)}")
        return path
