import pandas as pd
from pybatfish.client.session import Session
from ipaddress import IPv4Network
from typing import Dict, Iterable, List, Set, Tuple, cast, Optional
from .forwarding_graph import ForwardingGraph, ForwardingGraphGroup, ForwardingEdge
from .utils.layer1 import remote_nodes_for_interface
from .utils.fecs import FECFinder


def build_forwarding_graphs(
        forwarding_behaviour_df: pd.DataFrame,
        snapshot: str,
        layer1_map: Dict[str, Set[str]],
        ip_owners_map: Dict[str, Set[str]],
        bf: Session = None
    ) -> List[ForwardingGraphGroup]:
    """
    Build forwarding graphs from the forwarding behaviour DataFrame, for each prefix in the df.
    """
    # Create session if not provided
    if bf is None:
        bf = Session(host="localhost")

    # For each prefix present in the forwarding behaviour table,
    # construct a directed graph where nodes are routers and edges represent forwarding paths.
    table = forwarding_behaviour_df
    df = table.copy()
    if df.empty:
        return []
    
    ip_owners_df = bf.q.ipOwners().answer(snapshot=snapshot).frame()
    ip_owners_map = _build_ip_owners_map(ip_owners_df)

    required_columns = {"Node", "Prefix", "Action"}
    for column in required_columns:
        if column not in table.columns:
            raise ValueError(f"Forwarding behaviour table missing required column: {column}")
        
    graphs: Dict[str, ForwardingGraph] = {}

    blocking_actions = {
        "denied",
        "denied_in",
        "denied_out",
        "drop",
        "dropped",
        "null_routed",
    }

    for prefix, group in df.groupby("Prefix", sort=False):
        prefix_str = str(prefix)
        edges: Dict[tuple, ForwardingEdge] = {}
        blocked_nodes: Set[str] = set()

        for row in group.itertuples(index=False):
            node = str(getattr(row, "Node"))
            action = str(getattr(row, "Action"))
            next_hop_interface = _safe_string(getattr(row, "Next_Hop_Interface", None))
            next_hop_ip = _safe_string(getattr(row, "Next_Hop_IP", None))

            action_normalized = action.lower()

            if action_normalized == "forwarded":
                targets = _forward_targets(
                    node,
                    next_hop_interface,
                    next_hop_ip,
                    layer1_map,
                    ip_owners_map,
                )
                for target in targets:
                    edge = ForwardingEdge(
                        src=node,
                        dst=target,
                        action=action,
                        next_hop_interface=next_hop_interface,
                        next_hop_ip=next_hop_ip,
                    )
                    edges[edge.as_tuple()] = edge
            else:
                terminal = _terminal_target(prefix_str, action)
                edge = ForwardingEdge(
                    src=node,
                    dst=terminal,
                    action=action,
                    next_hop_interface=next_hop_interface,
                    next_hop_ip=next_hop_ip,
                )
                edges[edge.as_tuple()] = edge
                if action_normalized in blocking_actions or action_normalized.startswith("forwarded_blocked"):
                    blocked_nodes.add(node)

        if blocked_nodes:
            edges = {
                key: edge
                for key, edge in edges.items()
                if not (edge.action.lower() == "forwarded" and edge.src in blocked_nodes)
            }

        has_prefix_sink = any(edge.dst.startswith("prefix:") for edge in edges.values())

        if not has_prefix_sink and not blocked_nodes:
            full_subset = df[df["Prefix"].astype(str) == prefix_str]
            for row in full_subset.itertuples(index=False):
                action = str(getattr(row, "Action"))
                if action.lower() != "accepted":
                    continue

                node = str(getattr(row, "Node"))
                next_hop_interface = _safe_string(getattr(row, "Next_Hop_Interface", None))
                next_hop_ip = _safe_string(getattr(row, "Next_Hop_IP", None))
                terminal = _terminal_target(prefix_str, action)
                edge = ForwardingEdge(
                    src=node,
                    dst=terminal,
                    action=action,
                    next_hop_interface=next_hop_interface,
                    next_hop_ip=next_hop_ip,
                )
                edges[edge.as_tuple()] = edge

            has_prefix_sink = any(edge.dst.startswith("prefix:") for edge in edges.values())

        if not edges:
            continue

        sorted_edges = tuple(edges[key] for key in sorted(edges))
        graphs[prefix_str] = ForwardingGraph(snapshot=snapshot, prefix=prefix_str, edges=sorted_edges)

    # After processing all prefixes, group the graphs into ForwardingGraphGroup objects and return them.
    return group_graphs(graphs.values())

def group_graphs(graphs: Iterable[ForwardingGraph]) -> List[ForwardingGraphGroup]:
    """Group graphs that share the same signature and collapse their prefixes into FECs."""

    buckets: Dict[str, Dict[str, object]] = {}

    for graph in graphs:
        signature = graph.signature
        entry = buckets.setdefault(signature, {"graph": graph, "prefixes": set()})
        prefixes = cast(Set[IPv4Network], entry["prefixes"])
        prefixes.add(IPv4Network(graph.prefix, strict=False))

    results: List[ForwardingGraphGroup] = []
    for signature in sorted(buckets):
        entry = buckets[signature]
        graph = cast(ForwardingGraph, entry["graph"])
        prefixes_set = cast(Set[IPv4Network], entry["prefixes"])

        finder = FECFinder()
        finder.extend(prefixes_set)

        fec_networks: List[IPv4Network] = []
        for fec in finder.get_all_fecs():
            fec_networks.extend(fec.get_all_prefixes())

        results.append(
            ForwardingGraphGroup(
                snapshot=graph.snapshot,
                signature=signature,
                graph=graph,
                prefixes=_sorted_network_strings(prefixes_set),
                fec_prefixes=_sorted_network_strings(fec_networks),
            )
        )

    return results

def _build_ip_owners_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
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


def _safe_string(value: object | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _forward_targets(
    node: str,
    next_hop_interface: Optional[str],
    next_hop_ip: Optional[str],
    layer1_map: Dict[str, Set[str]],
    ip_owners_map: Dict[str, Set[str]],
) -> Set[str]:
    """Resolve forwarding targets for a row in the forwarding table."""

    targets: Set[str] = set()

    if next_hop_interface:
        local_token = f"{node}[{next_hop_interface}]"
        remote_nodes = remote_nodes_for_interface(local_token, layer1_map)
        if remote_nodes:
            targets.update(remote_nodes)

        if not targets and next_hop_ip:
            owners = ip_owners_map.get(next_hop_ip)
            if owners:
                targets.update(owners)

        if not targets:
            targets.add(f"interface:{local_token}")

    elif next_hop_ip:
        owners = ip_owners_map.get(next_hop_ip)
        if owners:
            targets.update(owners)

    if not targets and next_hop_ip:
        targets.add(f"next-hop:{next_hop_ip}")

    if not targets and next_hop_interface:
        targets.add(f"interface:{node}[{next_hop_interface}]")

    if not targets:
        targets.add("unknown-next-hop")

    return targets


def _terminal_target(prefix: str, action: str) -> str:
    normalized = action.lower()
    if normalized == "accepted":
        return f"prefix:{prefix}"
    return f"terminal:{normalized}"

def _sorted_network_strings(networks: Iterable[IPv4Network]) -> Tuple[str, ...]:
    ordered = sorted(networks, key=lambda net: (int(net.network_address), net.prefixlen))
    return tuple(str(net) for net in ordered)