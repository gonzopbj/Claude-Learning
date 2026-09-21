"""The `oracle` ceiling agent (SPEC.md "Agents" table, last row; "Baseline details").

It answers every query with the exact interventional mean of the current SCM, obtained
through `OracleAccess.truth(i, v)` (INTERFACES.md section 1.4) - the ONLY method of the access
object this agent uses.  All six quantiles equal the point, so every interval has width 0 and
covers the truth (the evaluator's coverage is inclusive); it never abstains and never
intervenes.  It ignores the observational batch.

This module never imports `world`: the access object is handed in by `build_agent`.
"""

from __future__ import annotations

import numpy as np

from .base import QUANTILE_LEVELS, Agent


class OracleAgent(Agent):
    intervenes = False
    never_abstains = True

    def observe(self, X_obs, t):
        pass                                   # the oracle learns nothing from data

    def refit(self):
        pass

    def answer(self, query_i, query_v):
        qi = np.asarray(query_i, dtype=np.int64)
        qv = np.asarray(query_v, dtype=np.float64)
        point = np.stack([np.asarray(self.access.truth(int(i), float(v)), dtype=np.float64)
                          for i, v in zip(qi, qv)])                        # (Q, d)
        quantiles = np.repeat(point[:, :, None], len(QUANTILE_LEVELS), axis=2)  # (Q, d, 6)
        return {"point": point, "quantiles": quantiles}
