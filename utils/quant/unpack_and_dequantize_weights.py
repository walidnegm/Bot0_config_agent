"""utils/quant/unpack_and_dequantize_weights.py"""

import sys
import logging
import torch
from typing import Dict, List, Tuple, Any, Optional, TextIO
from safetensors.torch import load_file
from pathlib import Path
from utils.quant.tensor_utils import (
    unpack_qweight_4bit_int32,
    unpack_qzeros_4bit_int32,
    # dequantize_qweights,
)

# todo: fix this later; may not need this module at all!

logger = logging.getLogger(__name__)


def unpack_and_dequantize_weights(
    model: Dict[str, torch.Tensor], layers: List[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """
    For each layer:
      - Unpacks qweight and qzeros from AWQ-packed int32 format.

      - Expects all tensors in **mathematical shape convention** before unpack/transpose:
          * qweight: [in_features_packed, out_features], int32 (packed 4-bit)
          * qzeros:  [in_features_packed, num_groups], int32 (packed 4-bit)
          * scales:  [in_features, num_groups], float16 or float32

      - Unpacks using provided utils:
          * unpack_qweight_4bit_int32(qweight): returns [in_features, out_features], int8
          * unpack_qzeros_4bit_int32(qzeros): returns [in_features, num_groups], int8

      - Transposes all unpacked tensors to standard PyTorch shape convention:
          * qweight: [out_features, in_features], int8
          * qzeros:  [out_features, num_groups], int8 or uint8
          * scales:  [out_features, num_groups], float32

      - Infers group size per layer (group_size = in_features // num_groups).
      - Dequantizes using groupwise scales and (optionally) zero points.
      - Skips layers with incompatible or unexpected shapes.

    Args:
        model: Dict mapping tensor names to tensors, where:
            * '{base}.qweight': [in_features_packed, out_features], int32
            * '{base}.qzeros':  [in_features_packed, num_groups], int32
            * '{base}.scales':  [in_features, num_groups], float16 or float32
        layers: List of layer base names (excluding .qweight, .qzeros, .scales).

    Returns:
        Tuple:
            - Dict[str, torch.Tensor]: Dequantized float32 weights for each layer,
                [out_features, in_features]
            - Dict[str, Dict[str, torch.Tensor]]: Unpacked (and transposed) tensors
                per layer:
                * 'qweight': [out_features, in_features], int8
                * 'qzeros':  [out_features, num_groups], int8 or uint8
                * 'scales':  [out_features, num_groups], float32

    Raises:
        ValueError: On shape mismatch or incompatible dimensions.
        TypeError: On dtype mismatch.

    Example:
        For a layer 'model.layers.0.self_attn.q_proj':
          model['model.layers.0.self_attn.q_proj.qweight']:
            [in_features_packed, out_features], int32
          model['model.layers.0.self_attn.q_proj.qzeros']:
            [in_features_packed, num_groups], int32
          model['model.layers.0.self_attn.q_proj.scales']:
            [in_features, num_groups], float16/float32

        After unpacking and transposing, qweight: [out_features, in_features], etc.
    """
    deq_weights = {}
    unpacked_artifacts = {}

    for base in layers:
        try:
            qw_raw = model[base + ".qweight"]  # [in_features_packed, out_features]
            qz_raw = model[base + ".qzeros"]  # [in_features_packed, num_groups]
            sc_raw = model[base + ".scales"]  # [in_features, num_groups] or similar

            # Unpack (still in math convention: in_features first)
            qw = unpack_qweight_4bit_int32(qw_raw)  # [in_features, out_features], int8
            qz = unpack_qzeros_4bit_int32(qz_raw)  # [in_features, num_groups], int8
            sc = (
                sc_raw.float() if sc_raw.dtype != torch.float32 else sc_raw
            )  # [in_features, num_groups], float32

            # Transpose to PyTorch convention ([out_features, in_features], etc.)
            qw = qw.T.contiguous()  # [out_features, in_features], int8
            qz = (
                qz.T.contiguous()
            )  # [num_groups, in_features] → [in_features, num_groups] if needed
            sc = (
                sc.T.contiguous()
            )  # [num_groups, in_features] → [in_features, num_groups] if needed

            # Now align all to [out_features, ...] on axis 0
            O, I = qw.shape
            G = sc.shape[0]
            if O != sc.shape[0]:
                # If after transpose, sc is [num_groups, in_features], swap
                if sc.shape[1] == O:
                    sc = sc.T.contiguous()
                    qz = qz.T.contiguous()
                else:
                    raise ValueError(
                        f"Can't align scales/qzeros for {base}: {sc.shape}, {qz.shape}, {qw.shape}"
                    )
            # Now, should have [out_features, in_features], [out_features, num_groups]
            O, I = qw.shape
            O2, G = sc.shape
            if O != O2:
                logger.info(f"Skipping {base}: out_features mismatch {O} vs {O2}")
                continue
            if I % G != 0:
                logger.info(
                    f"Skipping {base}: in_features ({I}) not divisible by num_groups ({G})"
                )
                continue
            if qz.shape != sc.shape:
                logger.info(
                    f"Skipping {base}: qzeros shape {qz.shape} does not match scales {sc.shape}"
                )
                continue

            group_size = I // G

            deq = dequantize_qweights(qw, sc, group_size, qz)
            deq_weights[base] = deq
            unpacked_artifacts[base] = {"qweight": qw, "qzeros": qz, "scales": sc}
        except Exception as e:
            logger.info(f"Skipping {base}: {e}")

    return deq_weights, unpacked_artifacts
