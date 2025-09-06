"""
utils/quant/quant_core.py

Core grouped dequantization utilities shared by AWQ and GPTQ paths.

Schematic:
    PackedLayer { Wq[int8 O×I], Sc[float O×G], Z?[float O×G], g_idx?[I] }
                                  │
                                  ▼
                    _dequant_grouped(Wq, Sc, Z, g_idx, out_dtype)
                      ├─ g_idx is None  → AWQ-style contiguous groups
                      │                    (requires I % G == 0)
                      └─ g_idx provided → GPTQ-style mapped groups

Notes:
- AWQ: groups are assumed contiguous and uniform; pass g_idx=None.
  If your export has a tail group (I % G != 0), synthesize g_idx upstream
  and use the GPTQ-style branch.
- GPTQ: g_idx maps input columns→group ids (0..G-1). Z is optional.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import torch

logger = logging.getLogger(__name__)


@dataclass
class PackedLayer:
    """
    Container of unpacked-but-quantized layer tensors ready for grouped dequant.

    Attributes:
        wq (torch.Tensor): Quantized weights (int8) of shape [O, I].
        sc (torch.Tensor): Group scales (float*) of shape [O, G].
        z (Optional[torch.Tensor]): Optional group zero-points (float*) of
            shape [O, G].
        g_idx (Optional[torch.Tensor]): Optional column→group index (long) of
            shape [I].
            - None for AWQ contiguous groups.
            - Required for GPTQ mapped groups.
    """

    wq: torch.Tensor  # [O, I], int8 (after unpack)
    sc: torch.Tensor  # [O, G], float*
    z: Optional[torch.Tensor]  # [O, G] or None
    g_idx: Optional[torch.Tensor]  # [I] or None (GPTQ)


def _dequant_grouped(
    wq: torch.Tensor,
    sc: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize grouped int8 weights using per-group scales (and optional zeros).

    Formulas:
        AWQ contiguous groups (g_idx is None, I % G == 0):
            W[:, s:e] = (Wq[:, s:e] - Z[:, g]) * Sc[:, g]   if Z given
                      =  Wq[:, s:e]            * Sc[:, g]   otherwise

        GPTQ mapped groups (g_idx provided):
            Scols = Sc[:, g_idx]  # broadcast per column
            Zcols = Z[:, g_idx]   # optional
            W     = (Wq - Zcols) * Scols               if Z given
                  =  Wq           * Scols               otherwise

    Args:
        Wq: Quantized weights, shape [O, I], dtype int8 (math domain after unpack).
        Sc: Group scales, shape [O, G], dtype float*.
        Z:  Optional group zeros, shape [O, G], dtype float* (may be None).
        g_idx: Optional column→group mapping, shape [I], dtype long.
        out_dtype: Output dtype (e.g., torch.float16 or torch.float32).

    Returns:
        torch.Tensor: Dequantized weights of shape [O, I] with dtype `out_dtype`.

    Raises:
        ValueError: If shapes are incompatible or mapping is invalid.
        TypeError:  If unexpected dtypes are provided.
    """
    # --- Basic rank and dtype checks ---
    if wq.ndim != 2 or sc.ndim != 2:
        raise ValueError(
            f"Wq and Sc must be rank-2; got Wq.ndim={wq.ndim}, Sc.ndim={sc.ndim}"
        )
    if wq.dtype != torch.int8:
        raise TypeError(f"Wq must be int8 (after unpack); got {wq.dtype}")
    if sc.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"Sc must be float16/float32/bfloat16; got {sc.dtype}")
    if z is not None and z.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"Z must be float16/float32/bfloat16; got {z.dtype}")
    if g_idx is not None and g_idx.dtype not in (torch.int64, torch.long):
        raise TypeError(f"g_idx must be torch.long; got {g_idx.dtype}")

    O, I = wq.shape
    O2, G = sc.shape
    if O2 != O:
        logger.error("Row mismatch: Wq=%s, Sc=%s", tuple(wq.shape), tuple(sc.shape))
        raise ValueError(f"O mismatch: Wq={wq.shape}, Sc={sc.shape}")
    if z is not None and z.shape != sc.shape:
        logger.error(
            "Z/Sc shape mismatch: Z=%s, Sc=%s", tuple(z.shape), tuple(sc.shape)
        )
        raise ValueError(
            f"Z shape {tuple(z.shape)} must equal Sc shape {tuple(sc.shape)}"
        )

    # Minimal log to help trace branch and sizes
    logger.debug(
        "[_dequant_grouped] O=%d, I=%d, G=%d, has_Z=%s, has_g_idx=%s, out_dtype=%s, device=%s",
        O,
        I,
        G,
        bool(z is not None),
        bool(g_idx is not None),
        str(out_dtype),
        str(wq.device),
    )

    wq32 = wq.to(torch.float32)

    # --- AWQ-style contiguous groups (no g_idx) ---
    if g_idx is None:
        if G <= 0 or (I % G) != 0:
            logger.error(
                "Bad AWQ group config: I=%d, G=%d (requires I %% G == 0)", I, G
            )
            raise ValueError(f"Bad group config: I={I}, G={G} (requires I%G==0)")
        group_size = I // G
        logger.debug(
            "[_dequant_grouped] AWQ contiguous groups with group_size=%d", group_size
        )

        w = torch.empty((O, I), dtype=torch.float32, device=wq.device)
        # Iterate contiguous groups along input dimension
        for g in range(G):
            s_idx = g * group_size
            e_idx = (
                g + 1
            ) * group_size  # consider min(..., wq32.shape[1]) if last group is ragged

            # keep 2D for broadcasting: [O, 1]
            scale = sc[:, g : g + 1].to(torch.float32)
            if z is not None:
                zero = z[:, g : g + 1].to(torch.float32)
                w[:, s_idx:e_idx] = (wq32[:, s_idx:e_idx] - zero) * scale
            else:
                w[:, s_idx:e_idx] = wq32[:, s_idx:e_idx] * scale

        return w.to(out_dtype)

    # --- GPTQ-style mapped groups (g_idx provided) ---
    if g_idx.numel() != I:
        logger.error("g_idx length mismatch: len=%d vs I=%d", g_idx.numel(), I)
        raise ValueError(f"g_idx length {g_idx.numel()} != I {I}")

    # Defensive: ensure g_idx is in-range [0, G-1]
    gmin = int(g_idx.min().item())
    gmax = int(g_idx.max().item())
    if gmin < 0 or gmax >= G:
        logger.error(
            "g_idx out of range: min=%d, max=%d, allowed=[0..%d]", gmin, gmax, G - 1
        )
        raise ValueError(f"g_idx out of range: [{gmin},{gmax}] for G={G}")

    # Broadcast scales/zeros per column via index gather
    scols = sc[:, g_idx]  # [O, I], float*
    if z is not None:
        zcols = z[:, g_idx].to(torch.float32)
        w = (wq32 - zcols) * scols
    else:
        w = wq32 * scols

    return w.to(out_dtype)
