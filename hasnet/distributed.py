"""Small DDP helpers with no external provenance dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return DistributedContext(distributed, rank, world_size, local_rank, device)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return float(tensor.cpu())


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()
