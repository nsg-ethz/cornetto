#!/usr/bin/env python3
"""Infer forwarding predicates from forwarding graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .forwarding_graph import ForwardingGraph, ForwardingGraphGroup


@dataclass(frozen=True, slots=True)
class ReachabilityPredicate:
    node: str
    prefix: str

@dataclass(frozen=True, slots=True)
class WaypointPredicate:
    node: str
    prefix: str
    waypoint: str


@dataclass(frozen=True, slots=True)
class LoadBalancingPredicate:
    node: str
    prefix: str
    num_routes: int


@dataclass(frozen=True, slots=True)
class IsolationPredicate:
    node: str
    prefix: str


Predicate = Union[
    ReachabilityPredicate,
    IsolationPredicate,
    WaypointPredicate,
    LoadBalancingPredicate,
]


@dataclass(frozen=True, slots=True)
class PredicateSet:
    reachability: Tuple[ReachabilityPredicate, ...]
    isolation: Tuple[IsolationPredicate, ...]
    waypointing: Tuple[WaypointPredicate, ...]
    load_balancing: Tuple[LoadBalancingPredicate, ...]

    def all(self) -> Tuple[Predicate, ...]:
        """Return all predicates as a flat tuple."""

        return (*self.reachability, *self.isolation, *self.waypointing, *self.load_balancing)

    def as_sets(
        self,
    ) -> Tuple[
        Set[ReachabilityPredicate],
        Set[IsolationPredicate],
        Set[WaypointPredicate],
        Set[LoadBalancingPredicate],
    ]:
        """Return predicates grouped into mutable sets for diffing."""

        return (
            set(self.reachability),
            set(self.isolation),
            set(self.waypointing),
            set(self.load_balancing),
        )


@dataclass(frozen=True, slots=True)
class PredicateDiff:
    added_reachability: Tuple[ReachabilityPredicate, ...]
    removed_reachability: Tuple[ReachabilityPredicate, ...]
    added_isolation: Tuple[IsolationPredicate, ...]
    removed_isolation: Tuple[IsolationPredicate, ...]
    added_waypointing: Tuple[WaypointPredicate, ...]
    removed_waypointing: Tuple[WaypointPredicate, ...]
    added_load_balancing: Tuple[LoadBalancingPredicate, ...]
    removed_load_balancing: Tuple[LoadBalancingPredicate, ...]

    def is_empty(self) -> bool:
        return not (
            self.added_reachability
            or self.removed_reachability
            or self.added_isolation
            or self.removed_isolation
            or self.added_waypointing
            or self.removed_waypointing
            or self.added_load_balancing
            or self.removed_load_balancing
        )


def _is_sink(node: str, prefix: str) -> bool:
    return node == f"prefix:{prefix}"


def _is_device_node(node: str) -> bool:
    return not (
        node.startswith("prefix:")
        or node.startswith("terminal:")
        or node.startswith("interface:")
        or node.startswith("next-hop:")
        or node.startswith("unknown-")
    )


def _device_graph(graph: ForwardingGraph):
    try:
        import networkx as nx  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("networkx is required to infer predicates") from exc

    raw_graph = graph.to_networkx()
    prefix_node = f"prefix:{graph.prefix}"

    device_graph = nx.DiGraph()
    device_graph.add_node(prefix_node)

    for src, dst in raw_graph.edges():
        src_ok = _is_device_node(src) or _is_sink(src, graph.prefix)
        dst_ok = _is_device_node(dst) or _is_sink(dst, graph.prefix)
        if not (src_ok and dst_ok):
            continue
        device_graph.add_edge(src, dst)

    return device_graph


def _dominators(graph: ForwardingGraph):
    device_graph = _device_graph(graph)
    prefix_node = f"prefix:{graph.prefix}"

    if prefix_node not in device_graph:
        return {}, device_graph

    rev_graph = device_graph.reverse(copy=True)

    try:
        import networkx as nx  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("networkx is required to infer predicates") from exc

    try:
        immediate = nx.immediate_dominators(rev_graph, prefix_node)  # type: ignore[attr-defined]
    except nx.NetworkXError:
        return {}, device_graph

    return immediate, device_graph


def _tree_children(immediate: Dict[str, str]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {node: [] for node in immediate}
    for node, parent in immediate.items():
        if node == parent:
            continue
        if parent in children:
            children[parent].append(node)
    return children


def _dominates(immediate: Dict[str, str], start: str) -> Set[str]:
    children = _tree_children(immediate)
    stack = list(children.get(start, ()))
    dominated: Set[str] = set()
    while stack:
        node = stack.pop()
        dominated.add(node)
        stack.extend(children.get(node, ()))
    return dominated


def infer_predicates_for_graph(
    graph: ForwardingGraph,
    *,
    waypoint_nodes: Optional[Set[str]] = None,
) -> PredicateSet:
    immediate, device_graph = _dominators(graph)
    prefix_node = f"prefix:{graph.prefix}"

    device_nodes_all = {
        node
        for node in device_graph
        if node != prefix_node and _is_device_node(node)
    }

    if not immediate:
        isolation = tuple(
            sorted(
                (IsolationPredicate(node=node, prefix=graph.prefix) for node in device_nodes_all),
                key=lambda item: (item.prefix, item.node),
            )
        )
        return PredicateSet((), isolation, (), ())

    device_nodes = {
        node
        for node in immediate
        if node != prefix_node and _is_device_node(node)
    }

    isolation_nodes = device_nodes_all - device_nodes
    isolation = tuple(
        sorted(
            (IsolationPredicate(node=node, prefix=graph.prefix) for node in isolation_nodes),
            key=lambda item: (item.prefix, item.node),
        )
    )

    reachability = tuple(
        sorted(
            (ReachabilityPredicate(node=node, prefix=graph.prefix) for node in device_nodes),
            key=lambda item: item.node,
        )
    )

    if waypoint_nodes is None:
        waypoint_candidates = device_nodes
    else:
        waypoint_candidates = {node for node in device_nodes if node in waypoint_nodes}

    forwarded_targets: Dict[str, Set[str]] = {}
    for edge in graph.edges:
        if edge.action.lower() != "forwarded":
            continue
        if not _is_device_node(edge.src):
            continue
        if not (_is_device_node(edge.dst) or _is_sink(edge.dst, graph.prefix)):
            continue
        forwarded_targets.setdefault(edge.src, set()).add(edge.dst)

    waypoint_preds: List[WaypointPredicate] = []
    for waypoint in sorted(waypoint_candidates):
        dominated = _dominates(immediate, waypoint)
        for node in dominated:
            if node == waypoint or not _is_device_node(node):
                continue
            waypoint_preds.append(WaypointPredicate(node=node, prefix=graph.prefix, waypoint=waypoint))

    waypointing = tuple(sorted(waypoint_preds, key=lambda item: (item.waypoint, item.node)))

    load_balancing_preds: List[LoadBalancingPredicate] = []
    for node in sorted(device_nodes):
        targets = forwarded_targets.get(node, set())
        routes = len(targets)
        if routes > 1:
            load_balancing_preds.append(
                LoadBalancingPredicate(node=node, prefix=graph.prefix, num_routes=routes)
            )

    load_balancing = tuple(
        sorted(load_balancing_preds, key=lambda item: (item.node, item.prefix, item.num_routes))
    )

    return PredicateSet(
        reachability=reachability,
        isolation=isolation,
        waypointing=waypointing,
        load_balancing=load_balancing,
    )


def infer_predicates_for_group(
    group: ForwardingGraphGroup,
    *,
    waypoint_nodes: Optional[Set[str]] = None,
) -> PredicateSet:
    return infer_predicates_for_graph(group.graph, waypoint_nodes=waypoint_nodes)


def _sorted_reachability(preds: Iterable[ReachabilityPredicate]) -> Tuple[ReachabilityPredicate, ...]:
    return tuple(sorted(preds, key=lambda item: (item.prefix, item.node)))


def _sorted_isolation(preds: Iterable[IsolationPredicate]) -> Tuple[IsolationPredicate, ...]:
    return tuple(sorted(preds, key=lambda item: (item.prefix, item.node)))


def _sorted_waypointing(preds: Iterable[WaypointPredicate]) -> Tuple[WaypointPredicate, ...]:
    return tuple(sorted(preds, key=lambda item: (item.prefix, item.waypoint, item.node)))


def _sorted_load_balancing(preds: Iterable[LoadBalancingPredicate]) -> Tuple[LoadBalancingPredicate, ...]:
    return tuple(sorted(preds, key=lambda item: (item.prefix, item.node, item.num_routes)))


def collect_predicates(
    groups: Iterable[ForwardingGraphGroup],
    *,
    waypoint_nodes: Optional[Set[str]] = None,
) -> PredicateSet:
    reachability: Set[ReachabilityPredicate] = set()
    isolation: Set[IsolationPredicate] = set()
    waypointing: Set[WaypointPredicate] = set()
    load_balancing: Set[LoadBalancingPredicate] = set()

    for group in groups:
        predicates = infer_predicates_for_group(group, waypoint_nodes=waypoint_nodes)
        reachability.update(predicates.reachability)
        isolation.update(predicates.isolation)
        waypointing.update(predicates.waypointing)
        load_balancing.update(predicates.load_balancing)

    return PredicateSet(
        reachability=_sorted_reachability(reachability),
        isolation=_sorted_isolation(isolation),
        waypointing=_sorted_waypointing(waypointing),
        load_balancing=_sorted_load_balancing(load_balancing),
    )


def diff_predicate_sets(before: PredicateSet, after: PredicateSet) -> PredicateDiff:
    before_reach, before_iso, before_way, before_lb = before.as_sets()
    after_reach, after_iso, after_way, after_lb = after.as_sets()

    added_reach = after_reach - before_reach
    removed_reach = before_reach - after_reach
    added_iso = after_iso - before_iso
    removed_iso = before_iso - after_iso
    added_way = after_way - before_way
    removed_way = before_way - after_way
    added_lb = after_lb - before_lb
    removed_lb = before_lb - after_lb

    return PredicateDiff(
        added_reachability=_sorted_reachability(added_reach),
        removed_reachability=_sorted_reachability(removed_reach),
        added_isolation=_sorted_isolation(added_iso),
        removed_isolation=_sorted_isolation(removed_iso),
        added_waypointing=_sorted_waypointing(added_way),
        removed_waypointing=_sorted_waypointing(removed_way),
        added_load_balancing=_sorted_load_balancing(added_lb),
        removed_load_balancing=_sorted_load_balancing(removed_lb),
    )
