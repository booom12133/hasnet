"""Profile Params/FLOPs for the full model."""

import argparse
import torch
from thop import clever_format, profile

from models import FullModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="convnext", choices=["convnext", "resnet50", "resnext50", "regnetx_3_2gf"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(backbone=args.backbone, pretrained=args.pretrained).to(device).eval()
    x1 = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    x2 = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Params: {params}")
    print(f"FLOPs: {flops}")


if __name__ == "__main__":
    main()
