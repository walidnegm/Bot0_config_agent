"""utils/stats/welford.py"""

from __future__ import annotations
import torch
from typing import Dict


@torch.no_grad()
def welford_update(x: torch.Tensor, state: Dict[str, float]) -> None:
    """
    Update running stats with a new chunk `x` (any shape).
    - Flattens, casts to float64 on CPU for numerically stable mean/var.

    state: {"count": int, "mean": float, "M2": float}
    """
    x = x.reshape(-1)
    n = x.numel()
    if n == 0:
        return
    x64 = x.detach().to("cpu", dtype=torch.float64)
    xm = x64.mean().item()
    try:
        xv = x64.var(correction=0).item()  # PyTorch 2.x
    except TypeError:
        xv = x64.var(unbiased=False).item()  # PyTorch 1.x

    count = state["count"]
    mean = state["mean"]
    M2 = state["M2"]

    tot = count + n
    delta = xm - mean
    mean = mean + delta * (n / tot)
    M2 = M2 + xv * n + (delta * delta) * (count * n / tot)

    state["count"] = tot
    state["mean"] = mean
    state["M2"] = M2


@torch.no_grad()
def welford_finalize(state: Dict[str, float]) -> tuple[float, float]:
    """Return (mean, std) from a Welford state; (nan, nan) if empty."""
    if state["count"] == 0:
        return float("nan"), float("nan")
    var = state["M2"] / state["count"]
    return float(state["mean"]), float(var**0.5)
