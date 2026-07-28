import torch

from hasnet.config import build_model, load_config
from hasnet.models.hasnet import ViewStatCalibrator


def test_full_forward_and_debug_contract():
    config = load_config("configs/paper/full_convnext.yaml")
    model = build_model(config, pretrained=False).eval()
    image_ol = torch.randn(2, 3, 64, 64)
    image_sd = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        logits, auxiliary, debug = model(image_ol, image_sd, return_debug=True)
    assert logits.shape == (2, 15)
    assert auxiliary[0].shape == (2, 15)
    assert auxiliary[1].shape == (2, 15)
    assert torch.allclose(debug["w_ol"] + debug["w_sd"], torch.ones_like(debug["w_ol"]))
    assert torch.equal(debug["final_logits"], logits)


def test_main_only_has_no_auxiliary_output():
    config = load_config("configs/paper/ablations/src/main_only.yaml")
    model = build_model(config, pretrained=False).eval()
    with torch.inference_mode():
        logits, auxiliary = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 15)
    assert auxiliary is None


def test_vsc_is_identity_at_initialization():
    module = ViewStatCalibrator(8)
    tensor = torch.randn(2, 8, 5, 5)
    assert torch.equal(module(tensor), tensor)
