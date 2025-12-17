from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Sets:
    R: List[str]
    RR: List[Tuple[str, str]]  # directed arcs (e,r) with e!=r

def build_sets(regions: List[str]) -> Sets:
    rr = [(e, r) for e in regions for r in regions if e != r]
    return Sets(R=regions, RR=rr)
