"""Deterministic real-data replay helper for the local DSv4 alignment harness."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

_REPLAY_ITERATION = None
_REPLAY_MICRO_STEP = 0
_MANIFEST_INITIALIZED = False


def _array_md5(array: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def _next_replay_indices(args) -> tuple[int, int]:
    global _REPLAY_ITERATION, _REPLAY_MICRO_STEP

    iteration = int(getattr(args, "curr_iteration", 0))
    if _REPLAY_ITERATION != iteration:
        _REPLAY_ITERATION = iteration
        _REPLAY_MICRO_STEP = 0
    micro_step = _REPLAY_MICRO_STEP
    _REPLAY_MICRO_STEP += 1
    return iteration, micro_step


def _write_manifest(record: dict) -> None:
    global _MANIFEST_INITIALIZED

    manifest_dir = os.environ.get("MEGATRON_LOAD_MANIFEST_DIR")
    if not manifest_dir:
        return
    rank = int(record["rank"])
    path = Path(manifest_dir) / f"megatron_load_manifest_rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if _MANIFEST_INITIALIZED else "w"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    _MANIFEST_INITIALIZED = True


def override_batch_with_replay(batch: dict, args) -> dict:
    """Replace one dataloader microbatch with its exact dumped tokens and labels."""
    replay_dir = os.environ.get("LOAD_FIXED_DATA_PATH")
    if not replay_dir:
        return batch

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    step, micro = _next_replay_indices(args)
    suffix = f"step{step}_micro{micro}_rank{rank}_seq{args.seq_length}.npy"
    tokens_path = Path(replay_dir) / f"tokens_{suffix}"
    labels_path = Path(replay_dir) / f"labels_{suffix}"
    if not tokens_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Missing DSv4 replay microbatch: tokens={tokens_path}, labels={labels_path}"
        )

    tokens_np = np.load(tokens_path)
    labels_np = np.load(labels_path)
    expected_shape = (args.micro_batch_size, args.seq_length)
    if tokens_np.shape != expected_shape or labels_np.shape != expected_shape:
        raise ValueError(
            "Unexpected DSv4 replay shape: "
            f"tokens={tokens_np.shape}, labels={labels_np.shape}, expected={expected_shape}"
        )

    device = torch.cuda.current_device()
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    loss_mask = (labels != -100).to(dtype=torch.float32)
    position_ids = (
        torch.arange(args.seq_length, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(args.micro_batch_size, -1)
        .contiguous()
    )

    batch.update(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=None,
        position_ids=position_ids,
        cu_seqlens=None,
        cu_seqlens_padded=None,
        max_seqlen=None,
        local_cp_size=None,
        padding_mask=None,
    )
    _write_manifest(
        {
            "rank": rank,
            "step": step,
            "micro": micro,
            "tokens_file": tokens_path.name,
            "labels_file": labels_path.name,
            "tokens_shape": list(tokens_np.shape),
            "labels_shape": list(labels_np.shape),
            "input_ids_md5": _array_md5(tokens_np),
            "labels_md5": _array_md5(labels_np),
        }
    )
    return batch
