"""
Retrieval-augmented zero-shot pipeline.

Stage 1: Retrieval call given only the problem description/topology/specs (no raw
configs). The model returns a set of config filenames to inspect; we score
recall against ground-truth changed routers.

Stage 2: Repair call (search/replace + repair if needed) using only the retrieved configs
as context, leveraging ZeroShotRepair for parsing and correction.
"""

import copy
import logging
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.prompt_utils.prompts_alt_patch import _init_prompt as _search_replace_prompt
from src.utils.token_counter import token_counter

from src.modules.ZeroShotRepair import ZeroShotRepair

logger = logging.getLogger(__name__)


class ZeroShotRetrieval(ZeroShotRepair):
    """
    Two-stage retrieval + repair pipeline.
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
        max_retrieved_configs: int = 100,
        **model_kwargs,
    ):
        """Retrieval + repair pipeline.

        Accept model kwargs directly (provider, model_name, etc.) to match
        the signature used by ZeroShot and ZeroShotRepair.
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
            **model_kwargs,
        )
        self.max_retrieved_configs = max_retrieved_configs

    # ------------------------------------------------------------------ #
    #                        Retrieval Utilities                         #
    # ------------------------------------------------------------------ #
    def _build_retrieval_prompt(self, sample: Dict[str, Any], max_files: Optional[int] = None) -> str:
        """
        Build a retrieval prompt that excludes raw config bodies.
        """
        instruction = sample.get("instruction", "")
        spec_path = sample.get("specification_csv_path")
        router_names = list((sample.get("faulty_configs") or {}).keys())
        topology = sample.get("topology") or ""
        max_clause = (
            f"- Return at most {max_files} filenames; choose the most relevant ones. Return only the filenames you need (a superset of the truly affected routers is fine, as more router configs than the ones that need to be edited might be needed to diagnose the issue)."
            if max_files
            else "- Return only the filenames you need (a superset of the truly affected routers is fine, as more router configs than the ones that need to be edited might be needed to diagnose the issue)."
        )

        lines = [
            "You are given a network troubleshooting task.",
            "Goal: Identify **ALL** the root causes of the faults in the network. You must select the Minimum Viable Set of configuration files required to debug this issue.",
            "Rules:",
            "- Do NOT include raw configuration bodies.",
            "- Include any router config that is relevant either for fixing or diagnosis (context); over-selection is acceptable to ensure proper diagnosis, as long as the context remains manageable.",
            max_clause,
            "- Format your answer as YAML with top-level key 'retrieved_configs' listing filenames.",
            "",
            "Problem description (may include specs/topology, but configs are withheld):",
            instruction,
            "",
            "Topology (high-level, no configs):",
            str(topology),
            "",
            "Available router config filenames:",
            "\n".join(f"- {name}" for name in router_names),
        ]
        if spec_path:
            lines.append(f"\nSpecification source: {spec_path}")

        lines.append("\nRequired YAML format:\nretrieved_configs:\n  - Router1.cfg\n  - Router2.cfg\nmetadata:\n  rationale: <brief reason>\n")

        return "\n".join(lines)

    def _parse_retrieved_configs(self, response_text: str) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Parse retrieval response YAML and return a set of filenames plus metadata.
        """
        metadata: Dict[str, Any] = {"raw_retrieval_response": response_text}
        configs: Set[str] = set()

        # Try structured YAML first
        try:
            parsed = yaml.safe_load(response_text) or {}
            if isinstance(parsed, dict):
                candidate_configs = parsed.get("retrieved_configs") or parsed.get("routers") or []
                if candidate_configs is None:
                    candidate_configs = []
                if isinstance(candidate_configs, dict):
                    candidate_configs = list(candidate_configs.keys())
                if isinstance(candidate_configs, list):
                    configs.update({str(x).strip() for x in candidate_configs if str(x).strip()})
                parsed_meta = parsed.get("metadata")
                if isinstance(parsed_meta, dict):
                    metadata.update(parsed_meta)
        except Exception as e:
            logger.warning(f"Failed to parse retrieval response as YAML: {str(e)}")
            metadata["parse_error"] = str(e)

        # Fallback: regex all *.cfg occurrences to avoid brittle parsing
        try:
            import re
            regex_hits = re.findall(r"[A-Za-z0-9_.\\-]+\\.cfg", response_text)
            configs.update(hit.strip() for hit in regex_hits if hit.strip())
        except Exception as e:
            logger.warning(f"Regex extraction of cfg filenames failed: {str(e)}")

        return configs, metadata

    def _run_retrieval(self, sample: Dict[str, Any], sample_id: str = "") -> Tuple[Set[str], Dict[str, Any]]:
        """
        Execute retrieval-only call and compute recall against ground-truth changed routers.
        """
        retrieval_prompt = self._build_retrieval_prompt(sample, self.max_retrieved_configs)

        # Keep retrieval stateless
        memory_snapshot = self._snapshot_memory()
        if hasattr(self.model, "memory") and hasattr(self.model.memory, "chat_memory"):
            self.model.memory.chat_memory.clear()

        try:
            start_time = time.time()
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=retrieval_prompt),
            ]
            response = self.model.invoke(messages)
            elapsed = time.time() - start_time
            completion_tokens = token_counter(
                response.content if hasattr(response, "content") else str(response)
            )
            retrieved_set, retrieval_metadata = self._parse_retrieved_configs(
                response.content if hasattr(response, "content") else str(response)
            )
            faulty_configs = sample.get("faulty_configs") or {}
            original_configs = sample.get("original_configs") or {}
            total_candidates = len(faulty_configs)
            gt_changed = {
                name for name, cfg in faulty_configs.items()
                if original_configs.get(name) != cfg
            }
            tp = len(gt_changed & retrieved_set)
            gt_count = len(gt_changed)
            recall = tp / gt_count if gt_count > 0 else 1.0
            retrieval_metadata.update({
                "retrieved_configs": sorted(retrieved_set),
                "retrieved_count": len(retrieved_set),
                "retrieval_candidates": total_candidates,
                "retrieval_tp": tp,
                "retrieval_gt": gt_count,
                "retrieval_recall": recall,
                "inference_time": elapsed,
                "token_count": {
                    "prompt_tokens": token_counter(retrieval_prompt),
                    "system_tokens": token_counter(self.system_prompt),
                    "completion_tokens": completion_tokens,
                    "total_tokens": token_counter(retrieval_prompt)
                    + token_counter(self.system_prompt)
                    + completion_tokens,
                },
                "raw_response": response.content if hasattr(response, "content") else str(response),
            })
            logger.info(
                f"[Retrieval] {sample_id or 'sample'} -> recall={recall:.3f}, "
                f"retrieved {len(retrieved_set)}/{total_candidates}, "
                f"affected_count={len(gt_changed)}"
            )
            return retrieved_set, retrieval_metadata
        finally:
            self._restore_memory(memory_snapshot)

    # ------------------------------------------------------------------ #
    #                      Pipeline Override Hooks                       #
    # ------------------------------------------------------------------ #
    def inference_and_eval(
        self,
        feedback_regime=None,
        max_attempts: int = 3,
        similarity_threshold: float = 1.0,
    ):
        """
        Pre-run retrieval for each sample, filter configs to the retrieved subset,
        and then reuse the repair pipeline.
        """
        augmented_dataset = {}
        self.retrieval_logs: Dict[str, Dict[str, Any]] = {}

        for sample_id, sample in (self.eval_dataset or {}).items():
            sample_copy = copy.deepcopy(sample)
            sample_copy["full_faulty_configs"] = sample.get("faulty_configs", {})
            sample_copy["full_original_configs"] = sample.get("original_configs", {})
            retrieved_set, retrieval_meta = self._run_retrieval(sample_copy, sample_id)

            # Ensure we have at least one config to avoid parser edge cases
            if not retrieved_set:
                retrieved_set = set((sample_copy.get("faulty_configs") or {}).keys())
                retrieval_meta.setdefault("used_full_on_empty_retrieval", True)

            # Filter configs to retrieved subset
            sample_copy["faulty_configs"] = {
                name: cfg for name, cfg in (sample.get("faulty_configs") or {}).items()
                if name in retrieved_set
            }
            sample_copy["original_configs"] = {
                name: cfg for name, cfg in (sample.get("original_configs") or {}).items()
                if name in retrieved_set
            }
            # Rebuild instruction to include only the retrieved configs to keep context small
            try:
                narrowed_prompt = _search_replace_prompt(
                    final_configs=sample_copy["faulty_configs"],
                    topology=sample.get("topology", ""),
                    preds=sample.get("original_specs", []),
                )
                note = (
                    "NOTE: The configurations provided below are only the subset you retrieved "
                    "as relevant for diagnosis/repair. Other routers exist in the network but "
                    "are intentionally omitted from this prompt.\n\n"
                )
                sample_copy["instruction"] = note + narrowed_prompt
            except Exception as e:
                logger.warning(f"Failed to rebuild instruction for {sample_id}: {e}")
            sample_copy["retrieval_meta"] = retrieval_meta

            augmented_dataset[sample_id] = sample_copy
            self.retrieval_logs[sample_id] = retrieval_meta

        logger.info(f"Retrieval complete; running repair on {len(augmented_dataset)} examples")

        # Swap in the filtered dataset and run the normal repair pipeline
        original_dataset = self.eval_dataset
        self.eval_dataset = augmented_dataset
        try:
            return super().inference_and_eval(
                feedback_regime=feedback_regime,
                max_attempts=max_attempts,
                similarity_threshold=similarity_threshold,
            )
        finally:
            self.eval_dataset = original_dataset

    def _summarize_task_context(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extend base context with retrieval metrics.
        """
        base = super()._summarize_task_context(sample)
        retrieval_meta = sample.get("retrieval_meta") or {}
        base["retrieval"] = {
            "recall": retrieval_meta.get("retrieval_recall"),
            "tp": retrieval_meta.get("retrieval_tp"),
            "gt": retrieval_meta.get("retrieval_gt"),
            "retrieved_count": retrieval_meta.get("retrieved_count"),
            "retrieval_candidates": retrieval_meta.get("retrieval_candidates"),
        }
        return base

    def _compute_stats(self, results: List[Dict]):
        """
        Add average retrieval recall to aggregate stats.
        """
        base = super()._compute_stats(results)
        recalls = [r["retrieval_recall"] for r in results if r.get("retrieval_recall") is not None]
        base["avg_retrieval_recall"] = sum(recalls) / len(recalls) if recalls else None
        return base

    def _save_tempdir(self, configs: Dict[str, str]) -> tempfile.TemporaryDirectory:
        """
        Override to ensure evaluation runs on full network: merge fixed subset into full configs if available.
        """
        merged = {}
        if hasattr(self, "full_faulty_configs") and self.full_faulty_configs:
            merged.update(self.full_faulty_configs)
        merged.update(configs or {})
        return super()._save_tempdir(merged)
