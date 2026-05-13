"""LLM-judge scoring for diagnosis quality.

This module lets three LLM "judges" grade a model's diagnosis text against
ground-truth faults and configuration diffs. It returns per-judge scores and
the aggregate mean completeness/soundness/overall score.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from src.modules.chat.model_registry import create_chat_model


SYSTEM_PROMPT = """
You are a rigorous network incidents judge. Given:
1) Ground-truth faults from fault_metrics.csv
2) Literal configuration diffs (original -> faulty)
3) A model-provided diagnosis statement

Decide how well the diagnosis localizes the real faults.
- Completeness: proportion of true faults that are explicitly identified.
- Soundness: proportion of the diagnosis claims that match real faults (no hallucinated faults).
Return JSON only: {"completeness": float 0-1, "soundness": float 0-1, "overall_score": float 0-1, "rationale": str}
"""


def infer_fault_metrics_path(specification_csv_path: Optional[str | Path]) -> Optional[Path]:
    """Infer the fault_metrics.csv path from a specification CSV path."""

    if not specification_csv_path:
        return None
    base_dir = Path(specification_csv_path).resolve().parent.parent
    metrics_path = base_dir / "data_and_metrics" / "fault_metrics.csv"
    return metrics_path if metrics_path.exists() else None


def _load_fault_rows(metrics_path: Path, limit: int | None) -> list[MutableMapping[str, Any]]:
    rows: list[MutableMapping[str, Any]] = []
    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if row:
                rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    return rows


def _config_diffs(
    original_configs: Mapping[str, str],
    faulty_configs: Mapping[str, str],
    max_lines: int,
) -> Dict[str, str]:
    """Compute unified diffs per config file, trimmed to max_lines."""

    diffs: Dict[str, str] = {}
    for filename, faulty in faulty_configs.items():
        original = original_configs.get(filename, "") or ""
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(),
                faulty.splitlines(),
                fromfile=f"original/{filename}",
                tofile=f"faulty/{filename}",
                lineterm="",
            )
        )
        if max_lines and len(diff_lines) > max_lines:
            diff_lines = diff_lines[:max_lines] + ["... trimmed ..."]
        diffs[filename] = "\n".join(diff_lines)
    return diffs


def _format_faults(fault_rows: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for row in fault_rows:
        fault_id = row.get("fault_id", "?")
        target = row.get("target_id", "?")
        status = row.get("status", "?")
        cfg_files = row.get("config_files", "") or ""
        added = row.get("config_lines_added", "") or "0"
        removed = row.get("config_lines_removed", "") or "0"
        lines.append(
            f"- {fault_id} (status={status}, target={target}, files={cfg_files}, loc_added={added}, loc_removed={removed})"
        )
    return "\n".join(lines)


def _format_diffs(diff_map: Mapping[str, str]) -> str:
    parts = []
    for name, diff in diff_map.items():
        parts.append(f"[{name}]\n{diff}\n")
    return "\n".join(parts)


def _extract_json(text: str) -> Optional[MutableMapping[str, Any]]:
    """Extract the first JSON object from text."""

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def evaluate_diagnosis_with_llm_judges(
    diagnosis: str,
    fault_metrics_path: Path,
    original_configs: Mapping[str, str],
    faulty_configs: Mapping[str, str],
    judge_model_configs: Optional[Sequence[Mapping[str, Any]]],
    *,
    max_fault_rows: int = 8,
    max_diff_lines: int = 400,
) -> Dict[str, Any]:
    """Score diagnosis text with three LLM judges.

    Args:
        diagnosis: The model-produced diagnosis text.
        fault_metrics_path: Path to fault_metrics.csv.
        original_configs: Baseline configs keyed by filename.
        faulty_configs: Faulty configs keyed by filename.
        judge_model_configs: Sequence of model kwargs; each needs a ``provider`` key.
        max_fault_rows: Optional cap on how many fault rows to show in the prompt.
        max_diff_lines: Optional cap on diff lines per file shown to the judge.

    Returns:
        Dict containing per-judge scores and mean aggregates. If evaluation is
        skipped, ``skipped`` is True and ``reason`` explains why.
    """

    if not diagnosis or not diagnosis.strip():
        return {"skipped": True, "reason": "empty_diagnosis"}

    if not fault_metrics_path or not Path(fault_metrics_path).exists():
        return {"skipped": True, "reason": "missing_fault_metrics"}

    judge_configs = list(judge_model_configs or [])
    if not judge_configs:
        return {"skipped": True, "reason": "no_judges_configured"}

    fault_rows = _load_fault_rows(Path(fault_metrics_path), max_fault_rows)
    if not fault_rows:
        return {"skipped": True, "reason": "fault_metrics_empty"}

    diffs = _config_diffs(original_configs, faulty_configs, max_diff_lines)
    fault_text = _format_faults(fault_rows)
    diff_text = _format_diffs(diffs)

    user_prompt = textwrap.dedent(
        f"""
        Ground truth faults:\n{fault_text}\n\nConfig diffs (baseline -> faulty):\n{diff_text}\n\nDiagnosis to grade:\n{diagnosis}\n\nInstructions:\n- Score completeness (all real faults captured) and soundness (only real faults claimed).\n- Return JSON only with keys completeness, soundness, overall_score, rationale.
        """
    ).strip()

    judge_results = []
    completeness_vals: list[float] = []
    soundness_vals: list[float] = []
    overall_vals: list[float] = []

    for cfg in judge_configs:
        provider = cfg.get("provider")
        model_kwargs = {k: v for k, v in cfg.items() if k != "provider"}
        # Ensure sufficient tokens for reasoning models (they use tokens for CoT before output)
        if "max_tokens" not in model_kwargs:
            model_kwargs["max_tokens"] = 2000
        model_label = f"{provider}:{model_kwargs.get('model_name', 'unknown')}"

        try:
            judge = create_chat_model(provider, **model_kwargs)
            ai_message = judge.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
            print(content)
            parsed = _extract_json(str(content)) or {}
        except Exception as exc:  # pragma: no cover - defensive
            parsed = {"error": str(exc)}

        comp = _parse_float(parsed.get("completeness")) if isinstance(parsed, dict) else None
        snd = _parse_float(parsed.get("soundness")) if isinstance(parsed, dict) else None
        overall = _parse_float(parsed.get("overall_score")) if isinstance(parsed, dict) else None

        if overall is None and comp is not None and snd is not None:
            overall = (comp + snd) / 2

        if comp is not None:
            completeness_vals.append(comp)
        if snd is not None:
            soundness_vals.append(snd)
        if overall is not None:
            overall_vals.append(overall)

        judge_results.append({
            "model": model_label,
            "raw_response": parsed,
            "completeness": comp,
            "soundness": snd,
            "overall_score": overall,
        })

    mean_comp = sum(completeness_vals) / len(completeness_vals) if completeness_vals else None
    mean_snd = sum(soundness_vals) / len(soundness_vals) if soundness_vals else None
    mean_overall = sum(overall_vals) / len(overall_vals) if overall_vals else None

    return {
        "skipped": False,
        "judges": judge_results,
        "mean_completeness": mean_comp,
        "mean_soundness": mean_snd,
        "mean_score": mean_overall,
        "ground_truth_faults": len(fault_rows),
        "diff_files": len(diffs),
    }
