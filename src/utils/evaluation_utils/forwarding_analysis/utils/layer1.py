"""Helpers for working with Batfish layer-1 edge data."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd


def parse_interface_token(token: str) -> Tuple[str, str]:
    """Split a ``Node[Interface]`` token into its components."""

    if "[" not in token or not token.endswith("]"):
        raise ValueError(f"Unexpected interface format: {token}")
    node, iface = token[:-1].split("[", 1)
    return node, iface


def build_layer1_mapping(edges_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build an adjacency map from the layer-1 edges table."""

    if edges_df.empty:
        return {}

    for column in ("Interface", "Remote_Interface"):
        if column not in edges_df.columns:
            raise ValueError(f"Layer-1 table missing required column: {column}")

    mapping: Dict[str, Set[str]] = {}
    for local, remote in edges_df[["Interface", "Remote_Interface"]].itertuples(index=False, name=None):
        if pd.isna(local) or pd.isna(remote):
            continue
        local_str = str(local)
        remote_str = str(remote)
        mapping.setdefault(local_str, set()).add(remote_str)
        mapping.setdefault(remote_str, set()).add(local_str)
    return mapping


def remote_nodes_for_interface(interface_token: str, mapping: Dict[str, Set[str]]) -> Set[str]:
    """Return the remote node names connected to ``interface_token``."""

    neighbors = mapping.get(interface_token, set())
    nodes: Set[str] = set()
    for neighbor in neighbors:
        node, _ = parse_interface_token(neighbor)
        nodes.add(node)
    return nodes


def remote_interfaces_for_interface(interface_token: str, mapping: Dict[str, Set[str]]) -> Set[str]:
    """Return the remote interface tokens connected to ``interface_token``."""

    return mapping.get(interface_token, set()) or set()
