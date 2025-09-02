"""utils/quant/dequant_gptq.py

Utilities for dequantizing GPTQModel-style packed 4-bit weights.

This module normalizes per-layer tensors from a safetensors/SD state_dict into a
`PackedLayer` (Wq, Sc, Z?, g_idx), then provides:
- `dequantize_gptq_full(...)` to materialize the full FP matrix [O, I]
- `stream_dequant_gptq(...)` to yield row-chunks for memory-friendly dequant

Conventions:
- O = out_features, I = in_features
- G = number of groups (grouped scales/zeros)
- wq unpacked math domain is int8; sc/z are float

...
│  wq[int8 O×I]
│  sc[float O×G]
│  z?[float O×G]
│  g_idx[long I]
...

Notes:
- normalize_gptqmodel_layer() only unpacks and orients tensors.
- Actual float weights are produced on demand by dequantize/stream.
- Streaming avoids materializing the full matrix in VRAM at once.

Workflow Schematic:

    ┌───────────────────────┐
    │ raw state_dict tensors│
    │  (.qweight, .scales,  │
    │   .qzeros?, .g_idx)   │
    └───────────┬───────────┘
                │ normalize_gptqmodel_layer()
                ▼
        ┌─────────────────────┐
        │  PackedLayer bundle │
        │  wq[int8 O×I]       │
        │  sc[float O×G]      │
        │  z?[float O×G]      │
        │  g_idx[long I]      │
        └─────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
dequantize_gptq_full()   stream_dequant_gptq()
(full [O×I] float)       (yield chunks [rows×I])

"""

from typing import cast, Iterator, Optional
import logging
import torch
from utils.quant.dequant_core import PackedLayer, _dequant_grouped
from utils.quant.quant_unpack import (
    unpack_qweight_4bit_int32,
    unpack_qzeros_4bit_int32,
)

logger = logging.getLogger(__name__)


def normalize_gptqmodel_layer(t: dict[str, torch.Tensor], base: str) -> PackedLayer:
    """
    Normalize a GPTQModel layer into a PackedLayer usable by _dequant_grouped.

    Layout expectations (GPTQModel export):
        - scales: (G, O) or (O, G) → coerced to (O, G)
        - qzeros: packed int32, unpacked to (G, O) or (O, G) → coerced to (O, G)
        - g_idx:  (I,), long, mapping each input column to a group 0..G-1
        - qweight: packed int32, unpacked (int8 nibbles expanded) to either:
            A) (I//pack, O*pack)
            B) (O//pack, I*pack)
        → reshaped to a dense (O, I) int8 matrix

    Process:
        1. Unpack qweight to int8, reshape/collapse packed dimension → (O, I)
        2. Coerce scales to (O, G) float
        3. If present, unpack qzeros to float and align to (O, G)
        4. Validate g_idx length = I and range ∈ [0, G-1]

    Args:
        t:   State dict (or layer dict) containing the quantized tensors.
        base: Base name of the layer (e.g., "model.layers.0.mlp.down_proj").

    Returns:
        PackedLayer:
            wq   (int8)   [O, I]
            sc   (float)  [O, G]
            z?   (float)  [O, G] or None
            g_idx (long)  [I]

    Raises:
        KeyError:  If required tensors are missing in `t`.
        ValueError: If shapes cannot be coerced or g_idx is invalid.
    """
    pack = 8  # 4b → 8 vals per int32

    qweight_key = base + ".qweight"
    scales_key = base + ".scales"
    g_idx_key = base + ".g_idx"

    if qweight_key not in t or scales_key not in t or g_idx_key not in t:
        missing = [k for k in (qweight_key, scales_key, g_idx_key) if k not in t]
        raise KeyError(f"{base}: missing {missing}")

    g_idx = t[g_idx_key].to(torch.long).contiguous()
    sc_raw = t[scales_key].float().contiguous()

    # Dimensions from scales/g_idx (source of truth)
    # (works whether scales starts as (G,O) or (O,G))
    O = sc_raw.shape[1] if sc_raw.shape[1] != 0 else sc_raw.shape[0]
    G = sc_raw.shape[0] if O == sc_raw.shape[1] else sc_raw.shape[1]
    I = int(g_idx.numel())

    # qweight: unpack → reshape to (O, I)
    wq_raw = unpack_qweight_4bit_int32(t[qweight_key])
    wq = _reshape_qweight_to_O_I(wq_raw, O, I, pack=pack)

    # scales / zeros: orient to (O, G)
    sc = _orient_to_O_G(sc_raw, O)
    z = None
    qz = t.get(base + ".qzeros")
    if qz is not None:
        z_raw = unpack_qzeros_4bit_int32(qz).float().contiguous()
        z = _orient_to_O_G(z_raw, O)

    # Sanity checks
    if I % G != 0:
        raise ValueError(
            f"{base}: I({I}) not divisible by G({G}); group_size must be integer"
        )
    gmin, gmax = int(g_idx.min().item()), int(g_idx.max().item())
    if gmin < 0 or gmax >= G:
        raise ValueError(f"{base}: g_idx out of range [{gmin},{gmax}] for G={G}")
    if wq.shape != (O, I):
        raise AssertionError(
            f"{base}: wq shaped to {tuple(wq.shape)} != (O,I)=({O},{I})"
        )
    if sc.shape != (O, G):
        raise AssertionError(
            f"{base}: sc shaped to {tuple(sc.shape)} != (O,G)=({O},{G})"
        )
    if z is not None and z.shape != (O, G):
        raise AssertionError(f"{base}: z shaped to {tuple(z.shape)} != (O,G)=({O},{G})")

    logger.debug(
        "%s: GPTQ normalized -> wq=%s, sc=%s, z=%s, I=%d, O=%d, G=%d, group=%d",
        base,
        tuple(wq.shape),
        tuple(sc.shape),
        "none" if z is None else tuple(z.shape),
        I,
        O,
        G,
        I // G,
    )

    return PackedLayer(wq=wq.contiguous(), sc=sc.contiguous(), z=z, g_idx=g_idx)


def dequantize_gptq_full(
    layer: PackedLayer, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Materialize the full dequantized weight matrix [O, I].

    Args:
        layer: PackedLayer from `normalize_gptqmodel_layer`.
        out_dtype: Output dtype (e.g., torch.float32 or torch.float16).

    Returns:
        torch.Tensor: Dequantized weights shaped [O, I].
    """
    if layer.wq.ndim != 2 or layer.sc.ndim != 2:
        raise ValueError("PackedLayer tensors must be rank-2 for wq and sc")
    return _dequant_grouped(layer.wq, layer.sc, layer.z, layer.g_idx, out_dtype)


def stream_dequant_gptq(
    layer: PackedLayer,
    *,
    chunk_rows: int = 2048,
    out_dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> Iterator[torch.Tensor]:
    """Yield dequantized row-chunks of size up to `chunk_rows`.

    Useful when the full [O, I] matrix would be too large to materialize at once.

    Args:
        layer: PackedLayer from `normalize_gptqmodel_layer`.
        chunk_rows: Number of output rows (O dimension) per yielded chunk. Must be > 0.
        out_dtype: Output dtype for each chunk.
        device: If "cuda", tensors in each chunk are moved to CUDA before dequant.
                If "cpu" or None, stay on CPU. (Any other string raises.)

    Yields:
        torch.Tensor: Dequantized chunk of shape [min(chunk_rows, remaining_O), I].

    Raises:
        ValueError: For invalid `chunk_rows` or `device` value.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows}")
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"device must be None, 'cpu', or 'cuda'; got {device}")

    if layer.wq.shape[0] != layer.sc.shape[0]:
        raise ValueError(
            f"wq rows ({layer.wq.shape[0]}) must match Sc rows ({layer.sc.shape[0]})"
        )
    if layer.z is not None and layer.z.shape != layer.sc.shape:
        raise ValueError(
            f"z shape {tuple(layer.z.shape)} must equal sc shape {tuple(layer.sc.shape)}"
        )

    O, _ = layer.wq.shape
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
            # keep g_idx device-aligned with compute
            g_idx = layer.g_idx.to("cuda", non_blocking=True)
        else:
            g_idx = layer.g_idx  # CPU / None path

        yield _dequant_grouped(wq, sc, z, g_idx, out_dtype)


def _orient_to_O_G(t: torch.Tensor, O: int) -> torch.Tensor:
    """
    Return a tensor as [O, G] given an input that may be [O, G] or [G, O].

    Heuristic: try both orientations and choose the one with matching O on dim0.
    If both match (rare), prefer the candidate with larger G (dim1).

    Args:
        t: 2D tensor shaped [O, G] or [G, O].
        O: Expected out_features (rows).

    Returns:
        torch.Tensor: Contiguous scales tensor shaped [O, G] (float).

    Raises:
        ValueError: If `t` cannot be coerced to [O, G].
    """
    if t.ndim != 2:
        raise ValueError(f"rank-2 expected, got {t.ndim}D")
    if t.shape[0] == O:  # already (O, G)
        return t.contiguous()
    if t.shape[1] == O:  # (G, O) -> (O, G)
        return t.T.contiguous()
    raise ValueError(f"cannot orient to (O,G); got {tuple(t.shape)} for O={O}")


def _reshape_qweight_to_O_I(
    wq_raw: torch.Tensor, O: int, I: int, pack: int = 8
) -> torch.Tensor:
    """
    Accepts unpacked 4b qweight expanded to int8 with one packed axis:
        A) (I//pack, O*pack)  or
        B) (O//pack, I*pack)

    Returns dense (O, I) int8.
    """
    if wq_raw.ndim != 2:
        raise ValueError(f"qweight must be rank-2, got {wq_raw.ndim}D")
    r, c = wq_raw.shape

    # Case A: (I//pack, O*pack)
    if r == I // pack and c == O * pack:
        return (
            wq_raw.view(I // pack, O, pack).permute(1, 0, 2).reshape(O, I).contiguous()
        )

    # Case B: (O//pack, I*pack)
    if r == O // pack and c == I * pack:
        return wq_raw.view(O // pack, I, pack).reshape(O, I).contiguous()
    raise ValueError(
        f"qweight shape {tuple(wq_raw.shape)} not in "
        f"{{(I//{pack}, O*{pack}), (O//{pack}, I*{pack})}} for O={O}, I={I}"
    )
