"""Hierarchical / grouped time series.

A `Hierarchy` is fully described by:
  * bottom-level series, indexed by a tuple key (e.g. (region, store, sku))
  * an ordered list of group levels — every prefix of the key defines an aggregation

From this we construct the summation matrix S (n_total × n_bottom) such that
   y_all = S @ y_bottom
where y_all stacks rows in a canonical order: [top, level-1 groups..., bottom].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass
class Hierarchy:
    bottom: dict[tuple, pd.Series]    # key → series (same DatetimeIndex)
    levels: list[str]                 # names of grouping dimensions, top→bottom
    node_order: list[tuple] = field(default_factory=list)   # canonical row order
    S: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # ---------- construction ----------
    @classmethod
    def from_frame(cls, df: pd.DataFrame, value_col: str,
                   time_col: str, levels: list[str]) -> "Hierarchy":
        """Build from a long dataframe with columns [time_col, *levels, value_col]."""
        idx = pd.DatetimeIndex(sorted(df[time_col].unique()))
        bottom: dict[tuple, pd.Series] = {}
        for key, sub in df.groupby(levels):
            key = key if isinstance(key, tuple) else (key,)
            s = sub.set_index(time_col)[value_col].reindex(idx).astype(float)
            bottom[tuple(key)] = s
        return cls(bottom=bottom, levels=list(levels)).build()

    def build(self) -> "Hierarchy":
        bottom_keys = sorted(self.bottom.keys())
        n_bottom = len(bottom_keys)

        # Enumerate every aggregation node: prefixes of length 0..L-1, plus bottom (full key).
        # length-0 prefix is the grand total represented as ().
        nodes: list[tuple] = [()]
        for L in range(1, len(self.levels)):
            seen = set()
            for k in bottom_keys:
                p = k[:L]
                if p not in seen:
                    seen.add(p); nodes.append(p)
        # bottom rows last
        nodes.extend(bottom_keys)

        # Build S
        S = np.zeros((len(nodes), n_bottom), dtype=float)
        bk_index = {k: j for j, k in enumerate(bottom_keys)}
        for i, node in enumerate(nodes):
            if node == ():
                S[i, :] = 1.0
            elif len(node) == len(self.levels):
                S[i, bk_index[node]] = 1.0
            else:
                for k, j in bk_index.items():
                    if k[:len(node)] == node:
                        S[i, j] = 1.0
        self.node_order = nodes
        self.S = S
        return self

    # ---------- views ----------
    @property
    def n_total(self) -> int:  return self.S.shape[0]
    @property
    def n_bottom(self) -> int: return self.S.shape[1]

    def stack_actuals(self) -> pd.DataFrame:
        """(T × n_total) frame of all node-level actuals."""
        bottom_keys = self.node_order[-self.n_bottom:]
        Yb = np.column_stack([self.bottom[k].values for k in bottom_keys])  # (T, nb)
        Yall = Yb @ self.S.T
        idx = self.bottom[bottom_keys[0]].index
        cols = [self._label(n) for n in self.node_order]
        return pd.DataFrame(Yall, index=idx, columns=cols)

    def _label(self, node: tuple) -> str:
        if node == ():
            return "TOTAL"
        return "/".join(f"{self.levels[i]}={v}" for i, v in enumerate(node))