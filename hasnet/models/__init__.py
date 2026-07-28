"""Model registry."""

from .baselines import FeatureFusion, PlainDual, SingleView
from .hasnet import FullModel, HASNet

__all__ = ["HASNet", "FullModel", "SingleView", "PlainDual", "FeatureFusion"]
