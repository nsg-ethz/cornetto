"""Utilities for calculating high-level spec evaluation metrics."""

from __future__ import annotations

from typing import Mapping, MutableMapping


def compute_fix_rate(counts: Mapping[str, float | int]) -> float | None:
	"""Return fix rate = fixed / (fixed + not_fixed + broken).

	The caller may pass either a ``broken`` key or reuse ``side_effects`` for
	newly broken predicates. If the denominator is zero, ``None`` is returned.
	"""

	fixed = counts.get("fixed", 0)
	not_fixed = counts.get("not_fixed", 0)
	broken = counts.get("broken", counts.get("side_effects", 0))

	denominator = fixed + not_fixed + broken
	if denominator == 0:
		return None

	return fixed / denominator


def add_fix_rate_to_report(summary: MutableMapping[str, float | int]) -> None:
	"""Populate ``fix_rate`` in-place on a summary dictionary if missing."""

	if "fix_rate" in summary:
		return

	rate = compute_fix_rate(summary)
	summary["fix_rate"] = rate if rate is not None else 0.0

