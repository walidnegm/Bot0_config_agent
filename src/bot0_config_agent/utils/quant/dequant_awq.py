"""
utils/quant/dequant_awq.py

Dequantization pipeline for AWQ-style 4-bit packed weights.

Schematic:

    raw state_dict tensors
    (.qweight, .scales, .qzeros?)  ─┐
                                    │ normalize_awq_layer()
                                    ▼
                            PackedLayer { Wq[int8 O×I], Sc[float O×G],
                                        Z?[float O×G], g_idx[long I] }
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            dequantize_awq_full()             stream_dequant_awq()
            (full [O×I] float)                (yield [rows×I] chunks)

Notes:
- Synthesize g_idx for AWQ to robustly handle tail groups (I % G != 0).
- Scales are auto-oriented to [O, G]; zeros align to scales if present.
- Streaming keeps VRAM usage low and supports CUDA or CPU.
"""

from __future__ import annotations
from typing import Iterator, Optional
import logging
import torch

from bot0_config_agent.utils.quant.dequant_core import PackedLayer, _dequant_grouped
from bot0_config_agent.utils.quant.quant_unpack import (
    unpack_qweight_4bit_int32,
    unpack_qzeros_4bit_int32,
)

logger = logging.getLogger(__name__)


def _orient_scales_to_O_G(sc_raw: torch.Tensor, O: int) -> torch.Tensor:
    """
    Return scales as [O, G] given an input that may be [O, G] or [G, O].
    Prefer the orientation that matches O on dim0; if both match, pick larger G.

    Raises:
        ValueError if cannot coerce to [O, G].
    """
    if sc_raw is None or sc_raw.ndim != 2:
        raise ValueError("AWQ scales must be rank-2")
    cands = []
    for t in (sc_raw, sc_raw.T):
        if t.shape[0] == O and t.shape[1] > 0:
            cands.append(t)
    if not cands:
        raise ValueError(
            f"cannot orient AWQ scales to [O, G]; got {tuple(sc_raw.shape)} for O={O}"
        )
    cands.sort(key=lambda t: t.shape[1], reverse=True)
    return cands[0].contiguous()


def normalize_awq_layer(tensors: dict[str, torch.Tensor], base: str) -> PackedLayer:
    """
    Normalize an AWQ layer into a PackedLayer usable by _dequant_grouped.

    Expected keys:
        base + ".qweight"  (packed 4-bit in int32 tiles)
        base + ".scales"   (float, [O, G] or [G, O])
        base + ".qzeros"   (optional packed zeros; becomes float [O, G])

    Steps:
        - Unpack qweight (4-bit packed int32) into int8 math domain and
        transpose to [O, I].
        - Orient scales to [O, G] (no I%G constraint).
        - If present, unpack qzeros and align to scales shape [O, G].

    Returns:
        PackedLayer with: wq[int8 O×I], sc[float O×G], z[float O×G]
    """
    # --- qweight ---
    qweight_key = base + ".qweight"
    if qweight_key not in tensors:
        raise KeyError(f"{base}: missing '{qweight_key}'")
    wq = unpack_qweight_4bit_int32(tensors[qweight_key]).T.contiguous()
    if wq.ndim != 2:
        raise ValueError(f"{base}: unpacked wq must be 2D, got {tuple(wq.shape)}")
    O, I = wq.shape

    # --- scales ---
    scales_key = base + ".scales"
    if scales_key not in tensors:
        raise KeyError(f"{base}: missing '{scales_key}'")
    sc_raw = tensors[scales_key].float()
    sc = _orient_scales_to_O_G(sc_raw, O)  # -> [O, G]
    G = sc.shape[1]
    if G <= 0:
        raise ValueError(f"{base}: invalid G={G} (from scales)")

    # --- qzeros (optional) ---
    z = None
    qz_key = base + ".qzeros"
    qz_raw = tensors.get(qz_key)
    if qz_raw is not None:
        zc = unpack_qzeros_4bit_int32(qz_raw).float()
        if zc.shape != sc.shape:
            zc = zc.T
        if zc.shape != sc.shape:
            raise ValueError(
                f"{base}: qzeros shape {tuple(zc.shape)} incompatible with scales {tuple(sc.shape)}"
            )
        z = zc.contiguous()

    # --- enforce AWQ contiguous grouping (no mapping) ---
    if (I % G) != 0:
        raise ValueError(
            f"{base}: AWQ contiguous mode requires I % G == 0; got I={I}, G={G}. "
            "This export appears to have a tail group; re-export with uniform groups "
            "or switch to a mapped path that supplies g_idx."
        )

    logger.debug(
        "%s: AWQ normalized -> wq[%d,%d] int8, sc[%d,%d] float, z=%s, g_idx=None (G=%d)",
        base,
        O,
        I,
        sc.shape[0],
        sc.shape[1],
        "yes" if z is not None else "no",
        G,
    )

    # IMPORTANT: PackedLayer must also use lowercase fields in dequant_core.py
    return PackedLayer(wq=wq, sc=sc, z=z, g_idx=None)


def dequantize_awq_full(
    layer: PackedLayer, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Materialize the full dequantized weight matrix [O, I].

    Args:
        layer: PackedLayer from normalize_awq_layer.
        out_dtype: Output dtype, e.g. torch.float16 for perf or torch.float32
            for analysis.

    Returns:
        torch.Tensor shaped [O, I].
    """
    if layer.wq.ndim != 2 or layer.sc.ndim != 2:
        raise ValueError("PackedLayer tensors must be rank-2 for wq and sc")

    # g_idx is None → _dequant_grouped uses AWQ contiguous branch (requires I % G == 0)
    return _dequant_grouped(layer.wq, layer.sc, layer.z, None, out_dtype)


def stream_dequant_awq(
    layer: PackedLayer,
    chunk_rows: int = 2048,
    out_dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> Iterator[torch.Tensor]:
    """
    Yield dequantized row-chunks of size up to `chunk_rows`.

    Args:
        layer: PackedLayer from normalize_awq_layer.
        chunk_rows: Number of output rows (O dimension) per yielded chunk (>0).
        out_dtype: Output dtype for each chunk.
        device: "cuda" to move chunk tensors to CUDA; None/"cpu" stays on CPU.

    Yields:
        torch.Tensor: Dequantized chunk of shape [min(chunk_rows, remaining_O), I].
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows}")
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"device must be None, 'cpu', or 'cuda'; got {device}")

    if layer.wq.shape[0] != layer.sc.shape[0]:
        raise ValueError(
            f"wq rows ({layer.wq.shape[0]}) must match sc rows ({layer.sc.shape[0]})"
        )
    if layer.z is not None and layer.z.shape != layer.sc.shape:
        raise ValueError(
            f"z shape {tuple(layer.z.shape)} must equal sc shape {tuple(layer.sc.shape)}"
        )

    O, _ = layer.wq.shape
    logger.debug(
        "Streaming dequant (AWQ contiguous): O=%d, chunk_rows=%d", O, chunk_rows
    )

    # no g_idx handling at all
    for s in range(0, O, chunk_rows):
        e = min(s + chunk_rows, O)
        wq = layer.wq[s:e, :]
        sc = layer.sc[s:e, :]
        z = layer.z[s:e, :] if layer.z is not None else None

        if device == "cuda":
            wq = wq.to("cuda", non_blocking=True)
            sc = sc.to("cuda", non_blocking=True)
            if z is not None:
                z = z.to("cuda", non_blocking=True)

        yield _dequant_grouped(wq, sc, z, None, out_dtype)
