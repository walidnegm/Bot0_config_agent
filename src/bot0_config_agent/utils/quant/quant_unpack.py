"""utils/quant_unpack.py"""

import logging
import torch

logger = logging.getLogger(__name__)


def unpack_qzeros_4bit_int32(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit unsigned zero-points stored in int32 format.
    Each int32 contains 8 values.

    This is often used in AWQ models to store zero-points per group in compact form.

    Args:
        qzeros (torch.Tensor): Packed zero-points, shape [*, N], dtype int32

    Returns:
        torch.Tensor: Unpacked zero-points, shape [*, N * 8], dtype int8
    """
    if qzeros.dtype != torch.int32:
        raise TypeError(f"Expected int32 tensor for qzeros, got {qzeros.dtype}")

    logger.debug(f"Unpacking qzeros with shape {qzeros.shape}")

    unpacked = torch.stack(
        [((qzeros >> (4 * i)) & 0xF).to(torch.int8) for i in range(8)], dim=-1
    )  # shape: [..., 8]

    unpacked = unpacked.view(*qzeros.shape[:-1], -1)
    logger.debug(f"Unpacked qzeros shape: {unpacked.shape}")

    return unpacked


def unpack_qweight_4bit_int32(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit signed quantized weights from int32 format. Each int32 stores 8 values.
    Encoded using 2's complement: range is [-8, +7].

    Args:
        qweight (torch.Tensor): Packed quantized weights, shape [..., N], dtype int32

    Returns:
        torch.Tensor: Unpacked weights, shape [..., N * 8], dtype int8
    """
    if qweight.dtype != torch.int32:
        raise TypeError(f"Expected int32 tensor for qweight, got {qweight.dtype}")

    logger.debug(f"Unpacking qweight with shape {qweight.shape}")

    unpacked = torch.stack(
        [((qweight >> (4 * i)) & 0xF).to(torch.int8) for i in range(8)], dim=-1
    )  # shape: [..., 8]

    # Convert unsigned 4-bit to signed int8 using 2's complement
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)

    unpacked = unpacked.view(*qweight.shape[:-1], -1)
    logger.debug(f"Unpacked qweight shape: {unpacked.shape}")

    return unpacked
