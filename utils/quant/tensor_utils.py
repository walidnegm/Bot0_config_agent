"""
utils/quant/awq_tensor_utils.py
Helper functions to help analyze tensors for quantization.
"""

from pathlib import Path
from typing import Optional
import logging
import math
import torch
import safetensors.torch as st

from utils.quant.quant_unpack import unpack_qweight_4bit_int32, unpack_qzeros_4bit_int32
import logging_config

logger = logging.getLogger(__name__)


# def dequantize_qweights(
#     qweight: torch.Tensor,
#     scales: torch.Tensor,
#     group_size: Optional[int] = 128,  # hint only; we infer actual size
#     qzeros: Optional[torch.Tensor] = None,
#     out_dtype: torch.dtype = torch.float16,  # match real inference default
# ) -> torch.Tensor:
#     """
#     Dequantize AWQ-style quantized weights using group-wise scales and
#     optional group-wise zero points.

#     This function reconstructs float32 weights from unpacked int8 quantized weights.
#     For each group of input features, it applies the corresponding scale
#     (and zero point if provided):

#         W_fp32 = (Q - Z) * S  if qzeros is provided (asymmetric quantization)
#         W_fp32 = Q * S        if qzeros is None (symmetric quantization)

#     Args:
#         qweight (torch.Tensor): Unpacked quantized weights, shape [out_features,
#             in_features], dtype torch.int8.
#         scales (torch.Tensor): Group-wise scales, shape [out_features, num_groups],
#             dtype float16 or float32.
#         group_size: optional hint (default 128). Will be *overridden* by inference
#             from shapes.
#         qzeros (Optional[torch.Tensor]): Optional group-wise zero points,
#                                             shape [out_features, num_groups],
#                                             dtype torch.int8 or torch.uint8.
#                                             If None, symmetric quantization is assumed.
#         out_dtype: dtype of returned dequantized weights (default torch.float16).


#     Returns:
#         torch.Tensor: Dequantized weights, shape [out_features, in_features],
#         dtype torch.float32.

#     Raises:
#         TypeError: If input dtypes are incorrect.
#         ValueError: If input shapes are incompatible or inconsistent with group_size.

#     Example:
#         >>> qweight = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
#         >>> scales = torch.rand(4, 4) * 0.05  # for group_size = 4
#         >>> w_fp32 = dequantize_awq_weights(qweight, scales, group_size=4)
#     """
#     # --- Type/shape checks
#     if qweight.dtype != torch.int8:
#         raise TypeError(f"Expected int8 for qweight, got {qweight.dtype}")
#     if scales.dtype not in (torch.float16, torch.float32, torch.bfloat16):
#         raise TypeError(
#             f"Expected float16/float32/bfloat16 for scales, got {scales.dtype}"
#         )
#     if qweight.ndim != 2 or scales.ndim != 2:
#         raise ValueError("qweight and scales must be 2D tensors")

#     O, I = qweight.shape
#     s0, s1 = scales.shape

#     # Auto-fix common mix-up: scales saved as [G, O]
#     if s0 != O and s1 == O:
#         scales = scales.transpose(0, 1)
#         s0, s1 = scales.shape

#     if s0 != O:
#         raise ValueError(f"scales first dim must match out_features O={O}; got {s0}")

#     # Infer groups (G) and effective group size
#     G = s1
#     inferred_group_size = math.ceil(I / G) if G > 0 else 0

#     if group_size is not None and group_size != inferred_group_size:
#         logger.warning(
#             f"group_size hint={group_size} differs from inferred={inferred_group_size}; using inferred.",
#             RuntimeWarning,
#         )
#     group_size = inferred_group_size

#     # qzeros checks
#     if qzeros is not None:
#         if qzeros.ndim != 2:
#             raise ValueError("qzeros must be 2D")
#         if qzeros.shape != (O, G):
#             raise ValueError(f"qzeros shape {qzeros.shape} must be (O, G)=({O}, {G})")
#         if qzeros.dtype not in (torch.int8, torch.uint8):
#             raise TypeError(f"Expected int8/uint8 for qzeros, got {qzeros.dtype}")

#     # Use fp32 for scale math, then cast to out_dtype at the end
#     scales_fp32 = scales.to(torch.float32)
#     w = torch.empty((O, I), dtype=torch.float32, device=qweight.device)

#     # Group-wise dequant; allow a smaller tail if I % G != 0
#     for g in range(G):
#         start = g * group_size
#         end = min((g + 1) * group_size, I)
#         if start >= end:
#             break

#         q_slice = qweight[:, start:end].to(torch.float32)
#         if qzeros is not None:
#             zero = qzeros[:, g].to(torch.float32).unsqueeze(1)  # [O,1]
#             q_slice = q_slice - zero

#         scale = scales_fp32[:, g].unsqueeze(1)  # [O,1]
#         w[:, start:end] = q_slice * scale

#     return w.to(out_dtype)


def load_tensor_from_safetensors(path: str | Path, tensor_name: str) -> torch.Tensor:
    """
    Load a specific tensor by name from a safetensors file with
    safetensors.torch's .load_file (load all tensors / memory heavy)

    Args:
        path (str): Path to the .safetensors file.
        tensor_name (str): Name of the tensor inside the file
            (e.g. 'model.layers.0.self_attn.q_proj.qweight').

    Returns:
        torch.Tensor: Loaded tensor.
    """
    path = str(path) if isinstance(path, Path) else path

    tensors = st.load_file(path)
    if tensor_name not in tensors:
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}")
    return tensors[tensor_name]


def load_tensor_from_pt(path: str, tensor_name: str) -> torch.Tensor:
    """
    Load a specific tensor by name from a PyTorch .pt or .bin file containing
    a state_dict.

    Args:
        path (str): Path to the file (e.g., "model.layers.0.self_attn.q_proj.pt").
        tensor_name (str): Key in the state_dict (e.g., "qweight").

    Returns:
        torch.Tensor: The requested tensor.
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(obj)}")
    if tensor_name not in obj:
        raise KeyError(f"'{tensor_name}' not found in {path}")
    return obj[tensor_name]
