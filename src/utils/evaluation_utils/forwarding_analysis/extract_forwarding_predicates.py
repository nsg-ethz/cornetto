"Given a forwarding behaviour table from Batfish, extract forwarding predicates"

import pandas as pd
from pybatfish.client.session import Session
from typing import Dict, Set, List
from dataclasses import dataclass
from .predicates import *
from .build_forwarding_graphs import build_forwarding_graphs
from .utils.layer1 import build_layer1_mapping

def extract_forwarding_predicates(
    df: pd.DataFrame,
    snapshot: str = "default",
    bf: Session = None
) -> PredicateSet:
    """
    Extract forwarding predicates from the forwarding graphs.
    
    Args:
        df: Forwarding behavior table from Batfish
        snapshot: Name of the snapshot being analyzed
        bf: Batfish session object
    
    Returns:
        PredicateSet containing all extracted predicates
    """
    # Create session if not provided
    if bf is None:
        bf = Session(host="localhost")
    
    # Build the layer1 topology map (interface connections)
    layer1_edges = bf.q.layer1Edges().answer(snapshot=snapshot).frame()
    layer1_map = build_layer1_mapping(layer1_edges)
    
    # Build the IP owners map (which node owns which IP address)
    ip_owners_df = bf.q.ipOwners().answer(snapshot=snapshot).frame()
    ip_owners_map = _build_ip_owners_map(ip_owners_df)
    
    # Build forwarding graphs from the forwarding behavior table
    graph_groups = build_forwarding_graphs(
        forwarding_behaviour_df=df,
        snapshot=snapshot,
        layer1_map=layer1_map,
        ip_owners_map=ip_owners_map,
        bf=bf
    )
    
    # Extract and collect predicates from all graph groups
    predicates = collect_predicates(graph_groups)
    
    return predicates


def _build_ip_owners_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build a map from IP addresses to the nodes that own them."""
    if df.empty:
        return {}

    if "IP" not in df.columns or "Node" not in df.columns:
        raise ValueError("ipOwners table missing required columns: IP, Node")

    mapping: Dict[str, Set[str]] = {}
    for ip_value, node_value in df[["IP", "Node"]].itertuples(index=False, name=None):
        if pd.isna(ip_value) or pd.isna(node_value):
            continue
        ip_text = str(ip_value)
        node_text = str(node_value)
        if not ip_text or not node_text:
            continue
        mapping.setdefault(ip_text, set()).add(node_text)
    return mapping


