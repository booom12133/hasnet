"""Profile the exact model selected by a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from thop import profile

from hasnet.config import build_model, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/full_convnext.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config, pretrained=False).to(device).eval()
    size = int(config["data"]["image_size"])
    image_ol = torch.randn(1, 3, size, size, device=device)
    image_sd = torch.randn(1, 3, size, size, device=device)
    if config["model"]["name"] == "single_view":
        flops, thop_params = profile(model, inputs=(image_ol,), verbose=False)
    else:
        flops, thop_params = profile(model, inputs=(image_ol, image_sd), verbose=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    payload = {
        "experiment": config["name"],
        "parameters": int(params),
        "parameters_M": params / 1e6,
        "trainable_parameters": int(trainable_params),
        "thop_module_parameters": int(thop_params),
        "FLOPs": int(flops),
        "FLOPs_G": flops / 1e9,
        "image_size": size,
        "batch_size": 1,
        "counts_both_views": config["model"]["name"] != "single_view",
        "note": "FLOPs use THOP; parameter totals use model.parameters() so standalone nn.Parameter values are included.",
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
