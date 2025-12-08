"""Utilities for running YOLO models as backbone-only feature extractors."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from models.experimental import Ensemble, attempt_load
from models.yolo import Model


class BackboneModel(nn.Module):
    """Loads a YOLO model and forwards only through its backbone.

    This wrapper keeps the full model (including the detection head) available so that
    the original weights stay intact, but the forward pass short-circuits before the
    head and returns the final backbone activation.
    """

    def __init__(self, weights: Union[str, list[str]], map_location: Optional[torch.device] = None):
        super().__init__()
        loaded = attempt_load(weights, map_location=map_location)
        if isinstance(loaded, Ensemble):
            if len(loaded) != 1:
                raise ValueError('BackboneModel expects a single-model checkpoint, not an ensemble.')
            loaded = loaded[0]

        if not isinstance(loaded, Model):
            raise TypeError(f'Expected YOLO Model, received {type(loaded)}.')

        self.model: Model = loaded
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        self._feature_dim = self._infer_feature_dim()

    def forward(self, x, profile: bool = False):
        return self.model.forward_backbone(x, profile=profile)

    def full_model(self) -> Model:
        """Returns the underlying full YOLO model."""

        return self.model

    def get_model(self) -> Model:
        """Returns the unwrapped YOLO model."""

        return self.model

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass to extract features from images.

        Args:
            x: Input tensor shaped ``(B, 3, H, W)``.

        Returns:
            A dict containing the final backbone features under the ``"features"`` key.
        """

        features = self.model.forward_backbone(x)
        return {"features": features}

    def forward_pool(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Pools backbone features using adaptive average pooling."""

        pooled_features = self._pool(x["features"])
        return {"pooled_features": pooled_features}

    def feature_dim(self) -> int:
        """Return the dimension of the backbone output features."""

        return self._feature_dim

    def _infer_feature_dim(self) -> int:
        """Infer the feature dimension by running a small dummy input through the backbone."""

        with torch.no_grad():
            device = next(self.model.parameters()).device
            dummy = torch.zeros(1, 3, 64, 64, device=device)
            features = self.model.forward_backbone(dummy)
        return features.shape[1]

