"""Utilities for building forwarding equivalence classes (FECs).

Inspired/repurposed from https://github.com/nsg-ethz/config2spec/blob/master/config2spec/dataplane/fecs.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv4Network, summarize_address_range
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(slots=True)
class TrieNode:
    """Single node inside the prefix trie used to derive FECs."""

    bitstring: str
    is_rule: bool = False
    children: Dict[str, "TrieNode"] = field(default_factory=dict)


@dataclass(slots=True)
class EquivalenceClass:
    """Represents a continuous IPv4 address range that shares forwarding state."""

    first: Optional[int] = None
    last: Optional[int] = None
    max_length: int = 32
    _iter_cursor: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def __hash__(self) -> int:
        if self.first is None or self.last is None:
            raise ValueError("EquivalenceClass boundaries not initialised")
        return hash((self.first, self.last))

    def __iter__(self) -> Iterator[int]:
        self._iter_cursor = self.first
        return self

    def __next__(self) -> int:
        if self.first is None or self.last is None or self._iter_cursor is None:
            raise StopIteration

        current = self._iter_cursor
        if current > self.last:
            raise StopIteration

        self._iter_cursor += 1
        return current

    def add_prefix(self, prefix: IPv4Network) -> None:
        if not isinstance(prefix, IPv4Network):
            raise TypeError("prefix must be an IPv4Network")
        self.add_range(int(prefix.network_address), int(prefix.broadcast_address))

    def add_bitstring(self, bitstring: str) -> None:
        if not isinstance(bitstring, str):
            raise TypeError("bitstring must be a string")
        prefix_length = len(bitstring)
        first = int(bitstring + "0" * (self.max_length - prefix_length), 2)
        last = int(bitstring + "1" * (self.max_length - prefix_length), 2)
        self.add_range(first, last)

    def add_range(self, first: int, last: int) -> None:
        if first > last:
            raise ValueError("range start cannot be greater than range end")

        if self.first is None and self.last is None:
            self.first = first
            self.last = last
            return

        if self.first is None or self.last is None:
            raise ValueError("EquivalenceClass boundaries not initialised")

        new_first = min(self.first, first)
        new_last = max(self.last, last)

        if new_first > self.last + 1 or new_last < self.first - 1:
            raise DisjointECException("Cannot extend equivalence class with disjoint range")

        self.first = new_first
        self.last = new_last

    def get(self, item: int = 0) -> int:
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        if self.first is None or self.last is None:
            raise ValueError("EquivalenceClass boundaries not initialised")
        if item == -1:
            return self.last
        if item < 0 or item > self.last - self.first:
            raise IndexError("EquivalenceClass index out of bounds")
        return self.first + item

    def get_ip(self, item: int = 0) -> IPv4Address:
        return IPv4Address(self.get(item))

    def get_prefix(self) -> Optional[IPv4Network]:
        prefixes = self.get_all_prefixes()
        if not prefixes:
            return None
        return prefixes[0]

    def get_all_prefixes(self) -> List[IPv4Network]:
        if self.first is None or self.last is None:
            return []
        return list(summarize_address_range(IPv4Address(self.first), IPv4Address(self.last)))

    def __str__(self) -> str:
        if self.first is None or self.last is None:
            return "EquivalenceClass(<empty>)"
        return f"EquivalenceClass({IPv4Address(self.first)} - {IPv4Address(self.last)})"

    def __repr__(self) -> str:
        return self.__str__()


class FECFinder:
    """Builds a trie over IPv4 prefixes and exposes contiguous FECs."""

    def __init__(self, *, max_depth: int = 32) -> None:
        self._max_depth = max_depth
        self._root = TrieNode(bitstring="")
        self._prefixes: set[IPv4Network] = set()

    def insert_prefix(self, prefix: IPv4Network) -> None:
        if not isinstance(prefix, IPv4Network):
            raise TypeError("prefix must be an IPv4Network")
        if prefix in self._prefixes:
            return

        bitstring = self._ipv4_to_bitstring(prefix)
        node = self._root
        for idx, bit in enumerate(bitstring):
            node = node.children.setdefault(bit, TrieNode(bitstring=bitstring[: idx + 1]))
        node.is_rule = True
        self._prefixes.add(prefix)

    def extend(self, prefixes: Iterable[IPv4Network]) -> None:
        for prefix in prefixes:
            self.insert_prefix(prefix)

    def get_all_prefixes(self) -> List[IPv4Network]:
        return sorted(self._prefixes, key=lambda net: (int(net.network_address), net.prefixlen))

    def get_all_fecs(self) -> List[EquivalenceClass]:
        fecs: List[EquivalenceClass] = []
        current_fec_id = 0
        last_fec_id = 0
        stack: List[Tuple[TrieNode | str, int, bool]] = [(self._root, 0, False)]

        while stack:
            node_or_prefix, fec_id, active = stack.pop()

            if isinstance(node_or_prefix, TrieNode):
                node = node_or_prefix

                if not node.is_rule and not node.children:
                    raise ValueError("Encountered orphan trie node without a rule")

                if node.is_rule:
                    last_fec_id += 1
                    fec_id = last_fec_id
                    active = True

                if node.children:
                    for bit in ("0", "1"):
                        child = node.children.get(bit)
                        if child is not None:
                            stack.append((child, fec_id, active))
                        elif active:
                            stack.append((node.bitstring + bit, fec_id, active))
                else:
                    current_fec_id = fec_id
                    fec = EquivalenceClass(max_length=self._max_depth)
                    fec.add_bitstring(node.bitstring)
                    fecs.append(fec)
            else:
                bitstring = node_or_prefix
                if not fecs or fec_id != current_fec_id:
                    current_fec_id = fec_id
                    fecs.append(EquivalenceClass(max_length=self._max_depth))
                fecs[-1].add_bitstring(bitstring)

        return fecs

    def _ipv4_to_bitstring(self, prefix: IPv4Network) -> str:
        network_int = int(prefix.network_address)
        return f"{network_int:032b}"[: prefix.prefixlen]


def build_fecs_from_prefixes(prefixes: Iterable[IPv4Network]) -> List[EquivalenceClass]:
    """Convenience helper that returns FEC ranges for the supplied prefixes."""

    finder = FECFinder()
    finder.extend(prefixes)
    return finder.get_all_fecs()


class DisjointECException(Exception):
    """Raised when attempting to extend an equivalence class with a disjoint range."""