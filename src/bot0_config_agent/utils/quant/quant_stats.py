"""
utils/quant_stats

Quantized weight statistics (mean/std) with streaming dequant.

- Auto-detects backend (GPTQ if any *.g_idx exists; else AWQ), but you can override
  via backend="gptq" or backend="awq".
- Reuses normalization from dequant_* modules and the shared core dequant.
- Streams row chunks to keep memory low; stable Welford reductions in float64.
""" """
Quantized weight statistics (mean/std) with streaming dequant.

- Auto-detects backend (GPTQ if any *.g_idx exists; else AWQ), but you can override
  via backend="gptq" or backend="awq".
- Reuses normalization from dequant_* modules and the shared core dequant.
- Streams row chunks to keep memory low; stable Welford reductions in float64.
"""


from __future__ import annotations
from typing import Dict, Iterator, Optional, Tuple
from collections import OrderedDict
import logging
import torch

# Import from project modules
from bot0_config_agent.utils.quant.dequant_core import _dequant_grouped, PackedLayer
from bot0_config_agent.utils.quant.dequant_awq import normalize_awq_layer
from bot0_config_agent.utils.quant.dequant_gptq import normalize_gptqmodel_layer

logger = logging.getLogger(__name__)


# -------- Backend detection & layer listing --------
def detect_backend(tensors: Dict[str, torch.Tensor]) -> str:
    """Heuristic: if any '*.g_idx' exists -> 'gptq', else 'awq'."""
    for k in tensors:
        if k.endswith(".g_idx"):
            return "gptq"
    return "awq"


def list_layers(tensors: Dict[str, torch.Tensor]) -> Iterator[str]:
    """Yield base names for layers having both .qweight and .scales."""
    seen = set()
    for k in tensors:
        if k.endswith(".qweight"):
            base = k[:-8]  # strip ".qweight"
            if base + ".scales" in tensors:
                seen.add(base)
    for base in sorted(seen):
        yield base


# -------- Streaming dequant for a single layer --------
def stream_dequant_chunks(
    tensors: Dict[str, torch.Tensor],
    base: str,
    *,
    backend: Optional[str] = None,  # "awq" | "gptq" | "auto/None"
    chunk_rows: int = 2048,
    out_dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,  # "cuda" to dequant per-chunk on GPU
) -> Iterator[torch.Tensor]:
    """
    Yield dequantized chunks [O_chunk, I] for a single layer.

    Args:
        tensors: safetensors state dict (name->tensor).
        base: base name of the layer (e.g., "model.layers.0.mlp.down_proj").
        backend: "awq", "gptq", or None/"auto" to detect automatically.
        chunk_rows: number of rows per yielded chunk (O dimension).
        out_dtype: dtype for the dequantized chunk.
        device: if "cuda", do per-chunk dequant on GPU.

    Yields:
        torch.Tensor shaped [min(chunk_rows, remaining_O), I] with dtype out_dtype.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows}")
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"device must be None/'cpu'/'cuda'; got {device}")

    mode = (backend or "auto").lower()
    if mode not in {"awq", "gptq", "auto"}:
        raise ValueError(f"backend must be 'awq', 'gptq', or 'auto'; got {backend}")

    if mode == "auto":
        mode = detect_backend(tensors)

    # Normalize to PackedLayer
    try:
        if mode == "gptq":
            layer: PackedLayer = normalize_gptqmodel_layer(tensors, base)
        else:  # "awq"
            layer: PackedLayer = normalize_awq_layer(tensors, base)
    except KeyError as e:
        logger.debug("[quant_stats] missing key(s) for %s: %s", base, e)
        return
    except Exception as e:
        logger.warning("[quant_stats] normalize failed for %s (%s): %s", base, mode, e)
        return

    O, _ = layer.wq.shape
    logger.debug(
        "[quant_stats] stream %s: O=%d, chunk_rows=%d, device=%s, out_dtype=%s",
        base,
        O,
        chunk_rows,
        device or "cpu",
        str(out_dtype),
    )

    # Sanity
    if layer.sc.ndim != 2 or layer.sc.shape[0] != O:
        logger.warning(
            "[quant_stats] skip %s: Sc shape %s incompatible with O=%d",
            base,
            tuple(layer.sc.shape),
            O,
        )
        return
    if layer.z is not None and layer.z.shape != layer.sc.shape:
        logger.warning(
            "[quant_stats] skip %s: Z shape %s != Sc %s",
            base,
            tuple(layer.z.shape),
            tuple(layer.sc.shape),
        )
        return

    # Chunked dequant along rows
    for s in range(0, O, chunk_rows):
        e = min(s + chunk_rows, O)

        Wq = layer.wq[s:e, :]
        Sc = layer.sc[s:e, :]
        Z = layer.z[s:e, :] if layer.z is not None else None

        if device == "cuda":
            Wq = Wq.to("cuda", non_blocking=True)
            Sc = Sc.to("cuda", non_blocking=True)
            if Z is not None:
                Z = Z.to("cuda", non_blocking=True)
            g_idx = (
                layer.g_idx.to("cuda", non_blocking=True)
                if layer.g_idx is not None
                else None
            )
        else:
            g_idx = layer.g_idx

        yield _dequant_grouped(Wq, Sc, Z, g_idx=g_idx, out_dtype=out_dtype)


# -------- Layer-wise and global stats (Welford) --------
@torch.no_grad()
def dequant_weights_layer_stats(
    tensors: Dict[str, torch.Tensor],
    *,
    backend: Optional[str] = None,  # "awq" | "gptq" | "auto/None"
    chunk_rows: int = 2048,
    out_dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
    return_counts: bool = False,
) -> Dict[str, Tuple[float, float] | Tuple[float, float, int]]:
    """
    Compute per-layer mean/std by streaming dequantized weights in row chunks.

    Returns:
        OrderedDict[base, (mean, std)] or (mean, std, count) if return_counts=True.
    """
    results: "OrderedDict[str, Tuple[float, float] | Tuple[float, float, int]]" = (
        OrderedDict()
    )

    for base in list_layers(tensors):
        count = 0
        mean = 0.0
        M2 = 0.0

        def _upd(x: torch.Tensor):
            nonlocal count, mean, M2
            x = x.reshape(-1)
            n = x.numel()
            if n == 0:
                return
            x64 = x.to(torch.float64)
            xm = x64.mean().item()
            try:
                xv = x64.var(correction=0).item()  # population variance
            except TypeError:
                xv = x64.var(unbiased=False).item()
            tot = count + n
            delta = xm - mean
            mean = mean + delta * (n / tot)
            M2 = M2 + xv * n + (delta * delta) * (count * n / tot)
            count = tot

        for deq in stream_dequant_chunks(
            tensors,
            base,
            backend=backend,
            chunk_rows=chunk_rows,
            out_dtype=out_dtype,
            device=device,
        ):
            _upd(deq.cpu())
            del deq
            if device == "cuda":
                torch.cuda.empty_cache()

        if count > 0:
            var = M2 / count
            std = var**0.5
            results[base] = (
                (float(mean), float(std), int(count))
                if return_counts
                else (float(mean), float(std))
            )

    return results


@torch.no_grad()
def dequant_weights_global_stats(
    tensors: Dict[str, torch.Tensor],
    *,
    backend: Optional[str] = None,  # "awq" | "gptq" | "auto/None"
    chunk_rows: int = 2048,
    out_dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Compute global (mean, std) over ALL dequantized weights across all layers.

    - Streams layer-by-layer in row-chunks so the full dequantized model is never
    resident in memory.
    - Uses Welford’s algorithm for numerically stable mean/std accumulation over
    streamed chunks.
    - Temporarily casts each chunk to float64 *only for reductions* (mean/var) to
    improve numerical accuracy; weights are not stored in float64 (saves VRAM/RAM).
    - Returns Python floats for global mean and std.

    Note:
        Welford’s algorithm is an online (single-pass) method to compute mean and
        variance stably. It’s ideal here because we process dequantized weights in
        chunks and cannot keep the whole tensor in memory.

    Args:
        tensors (Dict[str, torch.Tensor]):
            Mapping of tensor names → tensors (e.g., from safetensors.load_file).
        chunk_rows (int, optional):
            Number of out_features rows to dequantize per chunk. Defaults to 2048.
        out_dtype (torch.dtype, optional):
            Dtype for dequantized chunks before reduction (e.g., torch.float32).
            Defaults to torch.float32.
        device (Optional[str], optional):
            If "cuda", per-chunk dequant happens on GPU; reductions are still done
            on CPU in float64. Defaults to None.

    Returns:
        Tuple[float, float]:
            Global (mean, std) over all dequantized weights.

    Example:
        >>> import safetensors.torch as st
        >>> from utils.quant_stats import global_mean_std
        >>> tensors = st.load_file("model.safetensors")  # CPU dict[str, torch.Tensor]
        >>> mean, std = global_mean_std(
        ...     tensors,
        ...     chunk_rows=2048,
        ...     out_dtype=torch.float32,  # safer for stats
        ...     device=None,              # or "cuda" for per-chunk CUDA math
        ... )
        >>> print(mean, std)
    """

    mode = (backend or "auto").lower()
    if mode == "auto":
        mode = detect_backend(tensors)
    logger.info(
        "[quant_stats] backend=%s, chunk_rows=%d, out_dtype=%s, device=%s",
        mode,
        chunk_rows,
        str(out_dtype),
        device or "cpu",
    )

    count = 0
    mean = 0.0
    M2 = 0.0

    def _upd(x: torch.Tensor):
        nonlocal count, mean, M2
        x = x.reshape(-1)
        n = x.numel()
        if n == 0:
            return
        x64 = x.to(torch.float64)
        xm = x64.mean().item()
        try:
            xv = x64.var(correction=0).item()
        except TypeError:
            xv = x64.var(unbiased=False).item()
        tot = count + n
        delta = xm - mean
        mean = mean + delta * (n / tot)
        M2 = M2 + xv * n + (delta * delta) * (count * n / tot)
        count = tot

    for base in list_layers(tensors):
        for deq in stream_dequant_chunks(
            tensors,
            base,
            backend=mode,
            chunk_rows=chunk_rows,
            out_dtype=out_dtype,
            device=device,
        ):
            _upd(deq.cpu())
            del deq
            if device == "cuda":
                torch.cuda.empty_cache()

    if count == 0:
        return float("nan"), float("nan")
    var = M2 / count
    return float(mean), float(var**0.5)
