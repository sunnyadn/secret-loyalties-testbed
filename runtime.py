"""Device and dtype resolution, so the same scripts run on CUDA and on Apple MPS.

The original sweep ran on an A6000 in bfloat16. Apple MPS supports bfloat16 but
AdamW on bf16 LoRA parameters is unstable there, and a 0.5B base in float32 is
only ~2GB, so MPS gets float32. Every number in a single experiment batch must
come from one device and one dtype: ratios computed against a control trained
elsewhere are not comparable.
"""
from __future__ import annotations

import os

import torch


def resolve_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("SL_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    """float32 everywhere, deliberately.

    Every number in one experiment batch has to come from one dtype, and the
    batch has to be able to land on whichever GPU frees up first. V100-class
    hardware has no usable bfloat16, and AdamW over bf16 LoRA parameters is
    unstable on Apple MPS, so bf16 cannot be the common denominator. A 0.5B base
    in float32 is about 2GB, which fits everywhere and costs little at this
    scale, and it makes the local fallback batch directly comparable to the
    cluster batch instead of a separate universe.
    """
    del device
    return torch.float32


DEVICE = resolve_device()
DTYPE = resolve_dtype(DEVICE)
