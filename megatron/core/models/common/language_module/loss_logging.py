# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor

_loss_logging_suppressed = ContextVar("loss_logging_suppressed", default=False)


def _accuracy_compatible_loss_logging_enabled() -> bool:
    return (
        os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"
        and not _loss_logging_suppressed.get()
    )


@contextmanager
def suppress_accuracy_compatible_loss_logging():
    """Temporarily suppress standard LM-loss anchors for auxiliary losses."""
    token = _loss_logging_suppressed.set(True)
    try:
        yield
    finally:
        _loss_logging_suppressed.reset(token)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def log_accuracy_compatible_per_token_loss(loss: Tensor) -> None:
    """Log the unmasked per-token CE anchor used by model-alignment CI."""
    if not _accuracy_compatible_loss_logging_enabled():
        return
    loss = loss.detach().float().contiguous().cpu()
    print(
        f"\nper_token_loss: rank={_rank()} shape={list(loss.shape)} "
        f"md5={hashlib.md5(loss.numpy().tobytes()).hexdigest()}",
        flush=True,
    )


def log_accuracy_compatible_final_loss(loss: Tensor) -> None:
    """Log the local valid-token-normalized scalar loss anchor."""
    if not _accuracy_compatible_loss_logging_enabled():
        return
    loss = loss.detach().float().reshape(1).cpu()
    print(
        f"\nfinal_loss: rank={_rank()} val={float(loss.item()):.20f} "
        f"md5={hashlib.md5(loss.numpy().tobytes()).hexdigest()}",
        flush=True,
    )
