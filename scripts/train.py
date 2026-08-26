"""Train one paper configuration on one seed.

Use ``torchrun --standalone --nproc_per_node=2 scripts/train.py ...`` for the
paper's two-GPU protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from hasnet.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    sha256_file,
    write_json,
)
from hasnet.config import build_model, config_hash, load_config, serializable_config
from hasnet.distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from hasnet.ema import ModelEMA
from hasnet.engine import evaluate, make_loader, resolve_split_path, seed_everything
from hasnet.engine import train_one_epoch
from hasnet.losses import AsymmetricLoss
from hasnet.schedule import PaperV1Schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser.parse_args()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    try:
        config = load_config(args.config)
        if args.data_root is not None:
            config["data"]["root"] = args.data_root
        if args.seed is not None:
            config["seed"] = args.seed
        seed = int(config["seed"])
        seed_everything(seed, deterministic=bool(config["train"]["deterministic"]))

        output_dir = Path(args.output_dir) / config["name"] / f"seed_{seed}"
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(serializable_config(config), sort_keys=False),
                encoding="utf-8",
            )

        # Rank 0 downloads/caches pretrained weights first.
        if context.distributed and not context.is_main:
            barrier(context)
            model = build_model(config)
        else:
            model = build_model(config)
            if context.distributed:
                barrier(context)
        model = model.to(context.device)

        resume_state = load_checkpoint(args.resume, "cpu") if args.resume else None
        if resume_state:
            if (
                resume_state.get("config_sha256") != config_hash(config)
                and not args.allow_config_mismatch
            ):
                raise ValueError(
                    "Resume checkpoint/config mismatch. Use the original resolved config, "
                    "or pass --allow-config-mismatch only for an intentional fine-tune."
                )
            load_model_weights(model, resume_state, weights="raw")

        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                output_device=context.local_rank if context.device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        train_loader, train_sampler = make_loader(
            config, "train", context=context, train=True
        )
        val_loader = (
            make_loader(config, "val", context=None, train=False)[0]
            if context.is_main
            else None
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["train"]["learning_rate"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        scheduler = PaperV1Schedule(
            optimizer,
            epochs=int(config["train"]["epochs"]),
            steps_per_epoch=len(train_loader),
            warmup_epochs=int(config["train"]["scheduler"]["warmup_epochs"]),
            warmup_updates=config["train"]["scheduler"].get("warmup_updates"),
            eta_min_ratio=float(config["train"]["scheduler"]["eta_min_ratio"]),
        )
        amp_enabled = bool(config["train"]["amp"]) and context.device.type == "cuda"
        scaler = make_scaler(amp_enabled)
        ema_config = config["train"]["ema"]
        ema = (
            ModelEMA(
                model,
                decay=float(ema_config["decay"]),
                warmup_updates=int(ema_config["warmup_updates"]),
            )
            if bool(ema_config["enabled"])
            else None
        )
        criterion = AsymmetricLoss(
            gamma_neg=float(config["loss"]["gamma_neg"]),
            gamma_pos=float(config["loss"]["gamma_pos"]),
            clip=float(config["loss"]["clip"]),
        )

        start_epoch = 0
        best_map = float("-inf")
        if resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            if resume_state.get("scaler"):
                scaler.load_state_dict(resume_state["scaler"])
            if ema is not None and resume_state.get("ema"):
                ema.load_state_dict(resume_state["ema"])
            start_epoch = int(resume_state["epoch"]) + 1
            best_map = float(resume_state["best_metric"])

        split_hashes = {
            split: sha256_file(resolve_split_path(config, split))
            for split in ("train", "val", "test")
        }
        epochs = int(config["train"]["epochs"])
        selection_start_epoch = int(
            config["evaluation"].get("selection_start_epoch", 1)
        )
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                ema=ema,
                device=context.device,
                config=config,
                context=context,
                epoch=epoch,
            )
            barrier(context)
            if context.is_main:
                evaluation_model = ema.module if ema is not None else model
                val_result = evaluate(
                    model=evaluation_model,
                    loader=val_loader,
                    criterion=criterion,
                    device=context.device,
                    config=config,
                    description=f"val {epoch + 1}/{epochs}",
                )
                current_map = float(val_result.metrics["mAP"])
                eligible_for_selection = epoch + 1 >= selection_start_epoch
                improved = eligible_for_selection and current_map > best_map
                if eligible_for_selection:
                    best_map = max(best_map, current_map)
                state = build_checkpoint(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_map,
                    config=config,
                    split_hashes=split_hashes,
                )
                save_checkpoint(state, output_dir / "last.pt")
                if improved:
                    save_checkpoint(state, output_dir / "best.pt")
                row = {
                    "epoch": epoch + 1,
                    "seed": seed,
                    "train_loss": train_loss,
                    "val_loss": val_result.loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    **val_result.metrics,
                    **val_result.calibration,
                }
                append_jsonl(output_dir / "history.jsonl", row)
                print(
                    f"epoch={epoch + 1} train_loss={train_loss:.5f} "
                    f"val_mAP={current_map:.5f} best={best_map:.5f}"
                )
            barrier(context)

        if context.is_main:
            best_state = load_checkpoint(output_dir / "best.pt", "cpu")
            final_model = build_model(config, pretrained=False).to(context.device)
            load_model_weights(
                final_model,
                best_state,
                weights=str(config["evaluation"]["checkpoint_weights"]),
            )
            test_loader, _ = make_loader(config, "test", context=None, train=False)
            test_result = evaluate(
                model=final_model,
                loader=test_loader,
                criterion=criterion,
                device=context.device,
                config=config,
                description="test best",
            )
            summary = {
                "experiment": config["name"],
                "seed": seed,
                "best_epoch": int(best_state["epoch"]) + 1,
                "best_val_mAP": float(best_state["best_metric"]),
                "test_loss": test_result.loss,
                "test": test_result.metrics,
                "test_calibration": test_result.calibration,
                "config_sha256": config_hash(config),
                "split_sha256": split_hashes,
                "checkpoint": str((output_dir / "best.pt").resolve()),
                "checkpoint_sha256": sha256_file(output_dir / "best.pt"),
            }
            write_json(output_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2))
        barrier(context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
