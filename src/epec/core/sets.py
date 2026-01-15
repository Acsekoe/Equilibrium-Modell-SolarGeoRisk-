from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Sets:
    """Index sets.

    R  : regions
    A  : all directed arcs (e -> r), INCLUDING domestic (r -> r)
    RR : cross-border arcs only (e != r) (optional convenience)
    """

    R: List[str]
    A: List[Tuple[str, str]]
    RR: List[Tuple[str, str]]


def build_sets(regions: List[str]) -> Sets:
    a = [(e, r) for e in regions for r in regions]            # includes domestic arcs
    rr = [(e, r) for e in regions for r in regions if e != r] # cross-border arcs
    return Sets(R=regions, A=a, RR=rr)
