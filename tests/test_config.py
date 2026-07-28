from hasnet.config import build_model, load_config


def test_full_config_and_inheritance():
    full = load_config("configs/paper/full_convnext.yaml")
    no_vsc = load_config("configs/paper/ablations/core/no_vsc.yaml")
    assert full["model"]["use_vsc"] is True
    assert no_vsc["model"]["use_vsc"] is False
    assert no_vsc["train"] == full["train"]


def test_full_parameter_count_matches_manuscript_rounding():
    config = load_config("configs/paper/full_convnext.yaml")
    model = build_model(config, pretrained=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 30_970_211
    assert round(parameters / 1e6, 2) == 30.97
