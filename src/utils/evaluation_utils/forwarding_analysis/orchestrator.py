"""
Orchestrator for forwarding analysis. This module coordinates the extraction of forwarding predicates.
Then, it compares the observed forwarding behaviour against expected behaviour.
It works in the following steps:

Input: The paths to the configuration snapshots for the base and the to-be-compared networks.
Output: A report detailing the differences in forwarding behaviour.

1. Initialize Batfish and load the network snapshots.
2. Extract forwarding predicates from both networks using Batfish analysis.
3. Compare the extracted predicates to identify discrepancies.
"""

from typing import Tuple
from pybatfish.client.session import Session
from .get_forwarding_behaviour_table import get_forwarding_behaviour_table
from .extract_forwarding_predicates import extract_forwarding_predicates
from .predicates import diff_predicate_sets, PredicateDiff, PredicateSet


def orchestrate_single_forwarding_analysis(
    snapshot_path: str,
    bf: Session = None,
    bf_host: str = "localhost",
    snapshot_name: str = None
) -> PredicateSet:
    """
    Extract forwarding predicates from a single network snapshot.

    Args:
        snapshot_path (str): Path to the network snapshot.
        bf (Session, optional): Existing Batfish session to reuse.
        bf_host (str, optional): Host where Batfish is running. Defaults to "localhost".
        snapshot_name (str, optional): Name for the snapshot. If None, uses path basename.

    Returns:
        PredicateSet: Forwarding predicates extracted from the snapshot.
    """
    # Create session if not provided
    if bf is None:
        bf = Session(host=bf_host)
    
    # Generate snapshot name from path if not provided
    if snapshot_name is None:
        import os
        snapshot_name = os.path.basename(snapshot_path.rstrip('/'))
    
    # Load the tables from the snapshot
    single_df = get_forwarding_behaviour_table(
        snapshot_path, 
        snapshot_name=snapshot_name, 
        bf=bf
    )

    # Extract forwarding predicates from single snapshot
    single_preds = extract_forwarding_predicates(
        single_df, 
        snapshot=snapshot_name, 
        bf=bf
    )

    return single_preds

def orchestrate_paired_forwarding_analysis(
    base_snapshot_path: str,
    compare_snapshot_path: str,
    bf_host: str = "localhost"
) -> Tuple[PredicateDiff, PredicateSet, PredicateSet]:
    """
    Orchestrate the forwarding analysis between two network snapshots.

    Args:
        base_snapshot_path (str): Path to the base network snapshot.
        compare_snapshot_path (str): Path to the network snapshot to compare.
        bf_host (str, optional): Host where Batfish is running. Defaults to "localhost".

    Returns:
        Tuple[PredicateDiff, PredicateSet, PredicateSet]: 
            - Report detailing the differences in forwarding behaviour
            - Base snapshot predicates
            - Compare snapshot predicates
    """
    # Initialize Batfish session
    bf = Session(host=bf_host)
    
    # Extract forwarding predicates from each snapshot with unique names
    base_preds = orchestrate_single_forwarding_analysis(
        base_snapshot_path, 
        bf=bf,
        snapshot_name="base"
    )
    compare_preds = orchestrate_single_forwarding_analysis(
        compare_snapshot_path, 
        bf=bf,
        snapshot_name="compare"
    )

    # Compare the extracted predicates to identify discrepancies
    report = diff_predicate_sets(base_preds, compare_preds)

    return report, base_preds, compare_preds

# =========================================================================== #
#                                 Driver                                      #
# =========================================================================== #
if __name__ == "__main__":
    import pprint
    from ..scoring import calculate_similarity_scores, calculate_fix_scores

    base_snapshot = "/local/home/iprotogeros/benchmark-thesis-repo/benchmarking-llms-for-network-data-understanding/src/configs/single-003/initial_configs"
    compare_snapshot = "/local/home/iprotogeros/benchmark-thesis-repo/benchmarking-llms-for-network-data-understanding/src/configs/single-003/final_configs"

    print("=" * 80)
    print("FORWARDING ANALYSIS REPORT")
    print("=" * 80)
    print(f"Base snapshot: {base_snapshot}")
    print(f"Compare snapshot: {compare_snapshot}")
    print()
    
    report, base_preds, compare_preds = orchestrate_paired_forwarding_analysis(base_snapshot, compare_snapshot)
    
    # Calculate similarity scores
    scores = calculate_similarity_scores(base_preds, compare_preds)
    
    print("Similarity Scores:")
    print("-" * 80)
    print(f"SpecSimilarity: {scores['SpecSimilarity']}")
    print(f"ReachabilitySpecSimilarity: {scores['ReachabilitySpecSimilarity']}")
    print(f"IsolationSpecSimilarity: {scores['IsolationSpecSimilarity']}")
    print(f"WaypointingSpecSimilarity: {scores['WaypointingSpecSimilarity']}")
    print(f"LoadBalancingSpecSimilarity: {scores['LoadBalancingSpecSimilarity']}")
    print()
    
    print("Predicate Counts:")
    print("-" * 80)
    total_added = (
        len(report.added_reachability)
        + len(report.added_isolation)
        + len(report.added_waypointing)
        + len(report.added_load_balancing)
    )
    total_removed = (
        len(report.removed_reachability)
        + len(report.removed_isolation)
        + len(report.removed_waypointing)
        + len(report.removed_load_balancing)
    )
    print(f"Total Added Predicates: {total_added}")
    print(f"Total Removed Predicates: {total_removed}")
    print(f"  - Added Reachability: {len(report.added_reachability)}")
    print(f"  - Removed Reachability: {len(report.removed_reachability)}")
    print(f"  - Added Isolation: {len(report.added_isolation)}")
    print(f"  - Removed Isolation: {len(report.removed_isolation)}")
    print(f"  - Added Waypointing: {len(report.added_waypointing)}")
    print(f"  - Removed Waypointing: {len(report.removed_waypointing)}")
    print(f"  - Added Load Balancing: {len(report.added_load_balancing)}")
    print(f"  - Removed Load Balancing: {len(report.removed_load_balancing)}")
    print()
    
    print("Differences found:")
    print("-" * 80)
    if report.is_empty():
        print("✓ No differences detected in forwarding behavior!")
    else:
        print("\nAdded Reachability Predicates:")
        for pred in report.added_reachability:
            print(f"  + {pred}")
        
        print("\nRemoved Reachability Predicates:")
        for pred in report.removed_reachability:
            print(f"  - {pred}")
        
        print("\nAdded Isolation Predicates:")
        for pred in report.added_isolation:
            print(f"  + {pred}")

        print("\nRemoved Isolation Predicates:")
        for pred in report.removed_isolation:
            print(f"  - {pred}")

        print("\nAdded Waypointing Predicates:")
        for pred in report.added_waypointing:
            print(f"  + {pred}")
        
        print("\nRemoved Waypointing Predicates:")
        for pred in report.removed_waypointing:
            print(f"  - {pred}")
        
        print("\nAdded Load Balancing Predicates:")
        for pred in report.added_load_balancing:
            print(f"  + {pred}")
        
        print("\nRemoved Load Balancing Predicates:")
        for pred in report.removed_load_balancing:
            print(f"  - {pred}")
    
    print()
    print("=" * 80)
    print("Full report object:")
    pprint.pprint(report)
    
    # ========================================================================= #
    # Optional: Fix Scoring Example                                            #
    # ========================================================================= #
    # To use fix scoring, uncomment the section below and provide paths to:
    # - original_snapshot: The original working network (before fault)
    # - faulty_snapshot: The faulty network (with broken predicates)
    # - candidate_snapshot: The candidate solution attempting to fix the fault
    #
    # Example usage:
    # print("\n")
    # print("=" * 80)
    # print("FIX SCORING ANALYSIS")
    # print("=" * 80)
    # 
    # original_snapshot = "path/to/original/configs"
    # faulty_snapshot = "path/to/faulty/configs"
    # candidate_snapshot = "path/to/candidate/configs"
    # 
    # # Extract predicates from all three snapshots
    # _, original_preds, _ = orchestrate_paired_forwarding_analysis(original_snapshot, original_snapshot)
    # broken_diff, faulty_preds, _ = orchestrate_paired_forwarding_analysis(original_snapshot, faulty_snapshot)
    # _, _, candidate_preds = orchestrate_paired_forwarding_analysis(original_snapshot, candidate_snapshot)
    # 
    # # Calculate fix scores
    # fix_scores = calculate_fix_scores(original_preds, broken_diff, candidate_preds)
    # 
    # print(f"Fix Rate: {fix_scores['FixRate']:.2%}")
    # print(f"Total Broken: {fix_scores['TotalBrokenPredicates']}")
    # print(f"Total Fixed: {fix_scores['TotalFixedPredicates']}")
    # print()
    # print("Fix Rates by Type:")
    # print(f"  Reachability: {fix_scores['ReachabilityFixRate']:.2%}" if isinstance(fix_scores['ReachabilityFixRate'], float) else f"  Reachability: {fix_scores['ReachabilityFixRate']}")
    # print(f"  Waypointing: {fix_scores['WaypointingFixRate']:.2%}" if isinstance(fix_scores['WaypointingFixRate'], float) else f"  Waypointing: {fix_scores['WaypointingFixRate']}")
    # print(f"  Load Balancing: {fix_scores['LoadBalancingFixRate']:.2%}" if isinstance(fix_scores['LoadBalancingFixRate'], float) else f"  Load Balancing: {fix_scores['LoadBalancingFixRate']}")
    # print()
    # if fix_scores['UnfixedReachability'] or fix_scores['UnfixedWaypointing'] or fix_scores['UnfixedLoadBalancing']:
    #     print("Unfixed Predicates:")
    #     for pred in fix_scores['UnfixedReachability']:
    #         print(f"  - [Reachability] {pred}")
    #     for pred in fix_scores['UnfixedWaypointing']:
    #         print(f"  - [Waypointing] {pred}")
    #     for pred in fix_scores['UnfixedLoadBalancing']:
    #         print(f"  - [LoadBalancing] {pred}")
    # else:
    #     print("✓ All broken predicates were successfully fixed!")