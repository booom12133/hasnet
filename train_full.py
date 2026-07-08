"""Train or evaluate the proposed full model only.

Example:
    python train_full.py --data_root . --epochs 60
    python train_full.py --eval --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DvXrayDataset
from loss import AsymmetricLoss
from metrics import compute_metrics
from models import FullModel


CLASS_NAMES = [
    "Gun", "Knife", "Wrench", "Pliers", "Scissors", "Lighter", "Battery", "Bat",
    "Razor_blade", "Saw_blade", "Fireworks", "Hammer", "Screwdriver", "Dart", "Pressure_vessel",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the proposed full dual-view X-ray recognition model.")
    parser.add_argument("--data_root", type=str, default=".", help="Repository/dataset root. Split file image paths are resolved relative to this root.")
    parser.add_argument("--train_file", type=str, default="data/DvXray_train.txt")
    parser.add_argument("--val_file", type=str, default="data/DvXray_val.txt")
    parser.add_argument("--test_file", type=str, default="data/DvXray_test.txt")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation on the test split.")
    parser.add_argument("--backbone", type=str, default="convnext", choices=["convnext", "resnet50", "resnext50", "regnetx_3_2gf"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--aux_alpha", type=float, default=0.7)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split_file: str, args, train: bool):
    ds = DvXrayDataset(split_file, img_size=args.img_size, train=train, data_root=args.data_root)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers, pin_memory=True, drop_last=train)


@torch.no_grad()
def evaluate(model, loader, device, criterion, args, split_name: str = "val"):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    for image_ol, image_sd, targets in tqdm(loader, desc=f"Evaluate {split_name}"):
        image_ol = image_ol.to(device, non_blocking=True)
        image_sd = image_sd.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, aux_logits = model(image_ol, image_sd)
        loss = criterion(logits, targets, aux_logits=aux_logits, alpha=args.aux_alpha)
        total_loss += float(loss.item()) * targets.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(targets.cpu())
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(probs, targets, num_labels=len(CLASS_NAMES))
    metrics["loss"] = total_loss / len(loader.dataset)
    print(f"[{split_name}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, args, epoch: int):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for image_ol, image_sd, targets in pbar:
        image_ol = image_ol.to(device, non_blocking=True)
        image_sd = image_sd.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits, aux_logits = model(image_ol, image_sd)
            loss = criterion(logits, targets, aux_logits=aux_logits, alpha=args.aux_alpha)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item()) * targets.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(num_classes=len(CLASS_NAMES), backbone=args.backbone, pretrained=args.pretrained).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval:
        test_loader = make_loader(args.test_file, args, train=False)
        evaluate(model, test_loader, device, criterion, args, split_name="test")
        return

    train_loader = make_loader(args.train_file, args, train=True)
    val_loader = make_loader(args.val_file, args, train=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 45, 65], gamma=0.1)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, args, epoch)
        print(f"Train loss: {train_loss:.4f}")
        val_metrics = evaluate(model, val_loader, device, criterion, args, split_name="val")
        scheduler.step()
        is_best = val_metrics["mAP"] > best_map
        if is_best:
            best_map = val_metrics["mAP"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "best_mAP": best_map, "args": vars(args)}, out_dir / "best.pth")
            print(f"Saved best checkpoint: {out_dir / 'best.pth'}")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "best_mAP": best_map, "args": vars(args)}, out_dir / "last.pth")

    if Path(args.test_file).exists():
        test_loader = make_loader(args.test_file, args, train=False)
        print("Evaluating best checkpoint on test split...")
        ckpt = torch.load(out_dir / "best.pth", map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        evaluate(model, test_loader, device, criterion, args, split_name="test")


if __name__ == "__main__":
    main()
