"""Shared training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from hasnet.data import DvXrayDataset, Perturbation
from hasnet.distributed import DistributedContext, reduce_mean
from hasnet.metrics import compute_calibration_metrics, compute_metrics


@dataclass
class PredictionResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    targets: torch.Tensor
    indices: torch.Tensor
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    loss: float


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def resolve_split_path(config: dict, split: str) -> Path:
    path = Path(config["data"]["splits"][split])
    if path.is_absolute():
        return path
    return Path.cwd() / path


def make_loader(
    config: dict,
    split: str,
    *,
    context: DistributedContext | None = None,
    train: bool = False,
    perturbation: Perturbation | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    world_size = context.world_size if context else 1
    rank = context.rank if context else 0
    global_batch = int(config["train"]["global_batch_size"])
    if global_batch % world_size:
        raise ValueError(
            f"Global batch {global_batch} must be divisible by world size {world_size}"
        )
    per_process_batch = global_batch // world_size
    dataset = DvXrayDataset(
        resolve_split_path(config, split),
        img_size=int(config["data"]["image_size"]),
        train=train,
        data_root=config["data"]["root"],
        perturbation=perturbation,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=int(config["seed"]),
            drop_last=train,
        )
        if context and context.distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + rank)
    loader = DataLoader(
        dataset,
        batch_size=per_process_batch,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]) // world_size,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=(
            bool(config["data"].get("persistent_workers", True))
            and int(config["data"]["num_workers"]) // world_size > 0
        ),
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler


def forward_batch(model: nn.Module, batch: dict[str, Any], config: dict):
    model_name = config["model"]["name"]
    if model_name == "single_view":
        view = config["model"].get("view", "ol")
        output = model(batch[f"image_{view}"])
    else:
        output = model(batch["image_ol"], batch["image_sd"])
    if not isinstance(output, tuple):
        return output, None
    if len(output) < 2:
        return output[0], None
    return output[0], output[1]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("image_ol", "image_sd", "target"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler,
    scheduler,
    ema,
    device: torch.device,
    config: dict,
    context: DistributedContext,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    iterator = tqdm(
        loader,
        desc=f"train {epoch + 1}/{config['train']['epochs']}",
        disable=not context.is_main,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, aux_logits = forward_batch(model, batch, config)
            loss = criterion(
                logits,
                batch["target"],
                aux_logits=aux_logits,
                alpha=float(config["loss"]["aux_alpha"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_clip = float(config["train"]["grad_clip"])
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        scheduler.step_batch()
        batch_size = batch["target"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        iterator.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )
    scheduler.step_epoch()
    local_average = total_loss / max(total_samples, 1)
    return reduce_mean(local_average, context)


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    description: str = "evaluate",
) -> PredictionResult:
    model.eval()
    all_logits, all_targets, all_indices = [], [], []
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc=description, dynamic_ncols=True):
        batch = _move_batch(batch, device)
        logits, aux_logits = forward_batch(model, batch, config)
        loss = criterion(
            logits,
            batch["target"],
            aux_logits=aux_logits,
            alpha=float(config["loss"]["aux_alpha"]),
        )
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["target"].float().cpu())
        all_indices.append(batch["index"].long().cpu())
        batch_size = batch["target"].shape[0]
        total_loss += float(loss) * batch_size
        total_samples += batch_size
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    indices = torch.cat(all_indices)
    order = torch.argsort(indices)
    logits, targets, indices = logits[order], targets[order], indices[order]
    probabilities = torch.sigmoid(logits)
    metrics = compute_metrics(
        probabilities,
        targets,
        threshold=float(config["evaluation"]["threshold"]),
    )
    calibration = compute_calibration_metrics(
        probabilities,
        targets,
        num_bins=int(config["evaluation"]["calibration_bins"]),
    )
    return PredictionResult(
        logits=logits,
        probabilities=probabilities,
        targets=targets,
        indices=indices,
        metrics=metrics,
        calibration=calibration,
        loss=total_loss / max(total_samples, 1),
    )


def save_predictions(result: PredictionResult, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=result.logits.numpy(),
        probabilities=result.probabilities.numpy(),
        targets=result.targets.numpy().astype(np.int8),
        indices=result.indices.numpy(),
        metadata=np.array([metadata], dtype=object),
    )
