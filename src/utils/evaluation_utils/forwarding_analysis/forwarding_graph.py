#!/usr/bin/env python3
"""Data structures for forwarding graph analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence, Set, Tuple


@dataclass(frozen=True, slots=True)
class ForwardingEdge:
    """Directed edge representing a single forwarding decision."""

    src: str
    dst: str
    action: str
    next_hop_interface: str | None = None
    next_hop_ip: str | None = None

    def as_tuple(self) -> Tuple[str, str, str, str, str]:
        """Return a stable tuple form for hashing/sorting."""

        return (
            self.src,
            self.dst,
            self.action,
            self.next_hop_interface or "",
            self.next_hop_ip or "",
        )


@dataclass(frozen=True, slots=True)
class ForwardingGraph:
    """Directed graph for a single prefix within a snapshot."""

    snapshot: str
    prefix: str
    edges: Tuple[ForwardingEdge, ...]

    @property
    def nodes(self) -> Set[str]:
        nodes: Set[str] = set()
        for edge in self.edges:
            nodes.add(edge.src)
            nodes.add(edge.dst)
        # Ensure the prefix sink is considered part of the graph
        nodes.add(f"prefix:{self.prefix}")
        return nodes

    @property
    def signature(self) -> str:
        """Return a deterministic signature for grouping equivalent graphs."""

        hasher = hashlib.sha256()
        for edge_tuple in sorted(edge.as_tuple() for edge in self.edges):
            hasher.update("|".join(edge_tuple).encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def to_networkx(self):  # pragma: no cover - optional networkx dependency
        """Convert the graph into a NetworkX DiGraph."""

        try:
            import networkx as nx  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("networkx is required to build DiGraph objects") from exc

        graph = nx.DiGraph()
        graph.graph["snapshot"] = self.snapshot
        graph.graph["prefix"] = self.prefix

        for edge in self.edges:
            graph.add_edge(
                edge.src,
                edge.dst,
                action=edge.action,
                next_hop_interface=edge.next_hop_interface,
                next_hop_ip=edge.next_hop_ip,
            )
        return graph


@dataclass(frozen=True, slots=True)
class ForwardingGraphGroup:
    """Collection of prefixes sharing the same forwarding graph signature (hash)."""

    snapshot: str
    signature: str
    graph: ForwardingGraph
    prefixes: Tuple[str, ...]
    fec_prefixes: Tuple[str, ...]

    def to_networkx(self):  # pragma: no cover - optional networkx dependency
        """Expose the representative graph as a NetworkX DiGraph."""

        return self.graph.to_networkx()

    @property
    def edges(self) -> Sequence[ForwardingEdge]:
        """Convenience accessor to the graph edges."""

        return self.graph.edges

    @property
    def nodes(self) -> Set[str]:
        """Convenience accessor to the graph nodes."""

        return self.graph.nodes
