"""
Comprehensive specification evaluation report generator.

This module provides functionality to generate complete evaluation reports
comparing network configurations using forwarding analysis and predicate scoring.
Reports are saved in JSON format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .forwarding_analysis.orchestrator import orchestrate_single_forwarding_analysis
from .forwarding_analysis.predicates import (
    IsolationPredicate,
    LoadBalancingPredicate,
    Predicate,
    PredicateSet,
    ReachabilityPredicate,
    WaypointPredicate,
)
from .new_scoring import add_fix_rate_to_report


def _should_keep_source(raw_value: Any, broken_only: bool) -> bool:
    """Return True if a row should be considered based on its source value."""
    if not broken_only:
        return True

    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return False

    lower = str(raw_value).strip().lower()
    if "intact" in lower and "broken" not in lower:
        return False

    return True


def read_specification_from_csv(file_path: str, broken_only: bool = False) -> Dict[Predicate, str]:
    """
    Read a CSV file containing network specifications and return as a PredicateSet.
    
    Args:
        file_path: Path to the CSV file

    Returns:
        PredicateSet containing all extracted predicates

    CSV Format Example:
    predicate_type,node,prefix,waypoint,num_routes,sources
    reachability,0_as65003_0,10.180.0.1/32,,,intact
    reachability,0_as65003_0,10.180.0.10/32,,,intact
    reachability,0_as65003_0,10.180.0.26/32,,,broken_removed
    reachability,100_as65002_100,10.180.0.101/32,,,intact
    reachability,100_as65002_100,10.180.0.101/32,,,broken_added
    """
    df = pd.read_csv(file_path)
    dict_preds = {}
    # Map CSV rows to Predicate objects and give a value of 'intact' or 'broken' based on sources
    for _, row in df.iterrows():
        sources = row.get('sources', 'intact')
        if not _should_keep_source(sources, broken_only):
            continue

        pred_type = row['predicate_type']
        node = row['node']
        prefix = row['prefix']
        waypoint_raw = row.get('waypoint', "")
        waypoint = "" if pd.isna(waypoint_raw) else str(waypoint_raw)
        num_routes_raw = row.get('num_routes', None)
        num_routes = None if pd.isna(num_routes_raw) else int(num_routes_raw)

        if pred_type == 'reachability':
            predicate = ReachabilityPredicate(node=node, prefix=prefix)
            dict_preds[predicate] = sources
        elif pred_type == 'waypointing':
            predicate = WaypointPredicate(node=node, prefix=prefix, waypoint=waypoint)
            dict_preds[predicate] = sources
        elif pred_type == 'load_balancing':
            if num_routes is None:
                continue
            predicate = LoadBalancingPredicate(node=node, prefix=prefix, num_routes=num_routes)
            dict_preds[predicate] = sources
        elif pred_type == 'isolation':
            predicate = IsolationPredicate(node=node, prefix=prefix)
            dict_preds[predicate] = sources
        else:
            continue  # Unknown predicate type
    return dict_preds

def parse_spec_list(spec_lines: list[str], broken_only: bool = False) -> Dict[Predicate, str]:
    """
    Convert list of CSV-style spec strings into Dict[Predicate, str].
    
    Expected line format:
    predicate_type,node,prefix,waypoint,num_routes,sources
    """
    dict_preds = {}

    for line in spec_lines:
        parts = line.split(",")

        if len(parts) < 6:
            # skip malformed line
            continue

        predicate_type = parts[0].strip()
        node = parts[1].strip()
        prefix = parts[2].strip()
        waypoint = parts[3].strip() if parts[3] != "nan" else ""
        num_routes_raw = parts[4].strip()
        sources = parts[5].strip()

        if not _should_keep_source(sources, broken_only):
            continue

        # Convert types where needed
        num_routes = int(num_routes_raw) if num_routes_raw not in ("", "nan") else None

        # Build predicate
        if predicate_type == "reachability":
            pred = ReachabilityPredicate(node=node, prefix=prefix)
        elif predicate_type == "isolation":
            pred = IsolationPredicate(node=node, prefix=prefix)
        elif predicate_type == "waypointing":
            pred = WaypointPredicate(node=node, prefix=prefix, waypoint=waypoint)
        elif predicate_type == "load_balancing":
            if num_routes is None:
                continue
            pred = LoadBalancingPredicate(node=node, prefix=prefix, num_routes=num_routes)
        else:
            continue

        dict_preds[pred] = sources

    return dict_preds

def _predicate_to_dict(predicate: Predicate) -> Dict[str, Any]:
    """Serialize predicate dataclass for JSON output."""
    data: Dict[str, Any] = {
        "type": type(predicate).__name__,
    }

    if isinstance(predicate, ReachabilityPredicate):
        data.update({"node": predicate.node, "prefix": predicate.prefix})
    elif isinstance(predicate, IsolationPredicate):
        data.update({"node": predicate.node, "prefix": predicate.prefix})
    elif isinstance(predicate, WaypointPredicate):
        data.update({
            "node": predicate.node,
            "prefix": predicate.prefix,
            "waypoint": predicate.waypoint,
        })
    elif isinstance(predicate, LoadBalancingPredicate):
        data.update({
            "node": predicate.node,
            "prefix": predicate.prefix,
            "num_routes": predicate.num_routes,
        })

    return data


def _build_candidate_sets(predicate_set: PredicateSet) -> Dict[str, set[Predicate]]:
    """Create lookup sets by predicate type for fast membership checks."""
    return {
        "reachability": set(predicate_set.reachability),
        "isolation": set(predicate_set.isolation),
        "waypointing": set(predicate_set.waypointing),
        "load_balancing": set(predicate_set.load_balancing),
    }


def _parse_sources_value(raw_value: Any) -> Dict[str, Any]:
    """Normalize specification source metadata into evaluation directives."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        normalized = ""
    else:
        normalized = str(raw_value).strip()

    lower = normalized.lower()

    if "broken" in lower:
        initial_state = "broken"
    elif "intact" in lower:
        initial_state = "intact"
    else:
        initial_state = "unknown"

    if "broken_added" in lower or ("broken" in lower and "added" in lower):
        expected_presence = False
    elif "broken_removed" in lower or ("broken" in lower and "removed" in lower):
        expected_presence = True
    elif "added" in lower and "intact" in lower:
        expected_presence = False
    elif "removed" in lower and "intact" in lower:
        expected_presence = True
    elif "absent" in lower:
        expected_presence = False
    else:
        expected_presence = True

    return {
        "raw": normalized or "unknown",
        "initial_state": initial_state,
        "expected_presence": expected_presence,
    }


def _predicate_in_candidate(predicate: Predicate, candidate_sets: Dict[str, set[Predicate]]) -> bool:
    """Check whether the predicate is satisfied in the candidate set."""
    if isinstance(predicate, ReachabilityPredicate):
        return predicate in candidate_sets["reachability"]
    if isinstance(predicate, IsolationPredicate):
        return predicate in candidate_sets["isolation"]
    if isinstance(predicate, WaypointPredicate):
        return predicate in candidate_sets["waypointing"]
    if isinstance(predicate, LoadBalancingPredicate):
        return predicate in candidate_sets["load_balancing"]
    return False


def _summarize_predicate_set(predicate_set: PredicateSet) -> Dict[str, int]:
    """Produce simple counts for each predicate category."""
    return {
        "reachability": len(predicate_set.reachability),
        "isolation": len(predicate_set.isolation),
        "waypointing": len(predicate_set.waypointing),
        "load_balancing": len(predicate_set.load_balancing),
        "total": len(predicate_set.all()),
    }


def evaluate_candidate_against_specs(
    specification: Dict[Predicate, str],
    candidate_predicates: PredicateSet,
) -> Dict[str, list[Dict[str, Any]]]:
    """Classify predicates as fixed, not fixed, or side effects."""
    candidate_sets = _build_candidate_sets(candidate_predicates)

    fixed: list[Dict[str, Any]] = []
    not_fixed: list[Dict[str, Any]] = []
    side_effects: list[Dict[str, Any]] = []

    for predicate, raw_status in specification.items():
        status_info = _parse_sources_value(raw_status)
        expected_presence = status_info["expected_presence"]
        candidate_present = _predicate_in_candidate(predicate, candidate_sets)

        record = {
            "predicate": _predicate_to_dict(predicate),
            "initial_status": status_info["raw"],
            "expected_presence": "present" if expected_presence else "absent",
            "candidate_presence": "present" if candidate_present else "absent",
        }

        initial_state = status_info["initial_state"]

        if initial_state == "broken":
            predicate_fixed = (
                (expected_presence and candidate_present)
                or (not expected_presence and not candidate_present)
            )
            if predicate_fixed:
                fixed.append(record)
            else:
                not_fixed.append(record)
        elif initial_state == "intact":
            predicate_changed = (
                (expected_presence and not candidate_present)
                or (not expected_presence and candidate_present)
            )
            if predicate_changed:
                side_effects.append(record)
        else:
            # Unknown initial state; skip classification but keep visibility for debugging if needed.
            continue

    return {
        "fixed": fixed,
        "not_fixed": not_fixed,
        "side_effects": side_effects,
    }


def generate_spec_fix_report(
    reference_spec: Optional[list] = None,
    compared_spec: Optional[Union[PredicateSet, list]] = None, 
    specification_csv_path: Optional[str] = None,
    candidate_snapshot_path: Optional[str] = None,
    output_file: Optional[str] = None,
    bf_host: str = "localhost",
    metadata: Optional[Dict[str, Any]] = None,
    broken_only: bool = False,
) -> Dict[str, Any]:
    """Run the new workflow and dump the classification report as json."""
    if specification_csv_path:
        parsed_spec = read_specification_from_csv(specification_csv_path, broken_only=broken_only)
    elif reference_spec is not None:
        parsed_spec = parse_spec_list(reference_spec, broken_only=broken_only)
    else:
        raise ValueError("Either specification_csv_path or reference_spec must be provided.")

    if compared_spec is None:
        if candidate_snapshot_path is None:
            raise ValueError("Either compared_spec or candidate_snapshot_path must be provided.")
        compared_spec = orchestrate_single_forwarding_analysis(candidate_snapshot_path, bf_host=bf_host)

    # Convert list of CSV-like strings to PredicateSet
    if isinstance(compared_spec, list):
        pred_dict = parse_spec_list(compared_spec, broken_only=False)
        reachability = []
        isolation = []
        waypointing = []
        load_balancing = []
        for pred in pred_dict:
            if isinstance(pred, ReachabilityPredicate):
                reachability.append(pred)
            elif isinstance(pred, IsolationPredicate):
                isolation.append(pred)
            elif isinstance(pred, WaypointPredicate):
                waypointing.append(pred)
            elif isinstance(pred, LoadBalancingPredicate):
                load_balancing.append(pred)
        compared_spec = PredicateSet(
            reachability=tuple(reachability),
            isolation=tuple(isolation),
            waypointing=tuple(waypointing),
            load_balancing=tuple(load_balancing),
        )

    classification = evaluate_candidate_against_specs(parsed_spec, compared_spec)

    summary_counts: dict[str, float | int] = {
        key: len(value) for key, value in classification.items()
    }
    summary_counts["broken"] = summary_counts.get("side_effects", 0)

    total_spec = len(parsed_spec)
    evaluated = (
        summary_counts.get("fixed", 0)
        + summary_counts.get("not_fixed", 0)
        + summary_counts.get("broken", 0)
    )

    summary_counts["total_spec_predicates"] = total_spec
    summary_counts["evaluated_predicates"] = evaluated
    summary_counts["skipped_predicates"] = total_spec - evaluated

    add_fix_rate_to_report(summary_counts)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "specification_csv": str(specification_csv_path) if specification_csv_path else None,
            "candidate_snapshot": str(candidate_snapshot_path) if candidate_snapshot_path else None,
            **(metadata or {}),
        },
        "summary": summary_counts,
        "results": classification,
        "candidate_predicate_summary": _summarize_predicate_set(compared_spec),
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    return report
