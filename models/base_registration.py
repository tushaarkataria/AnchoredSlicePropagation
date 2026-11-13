"""
Abstract base class for all registration backends.

Every backend (Sli2Vol, TransMorph, GradICON) implements this interface
so that training, evaluation, and MORPH-LER can work with any of them.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseRegistrationNet(ABC, nn.Module):
    """
    Abstract interface for slice-to-slice registration.

    Subclasses must implement:
        - forward(source, target) -> dict with at least 'deformation' and 'warped_source'
        - warp(image, deformation_field) -> warped image

    The Sli2Vol backend is special: it uses correlation-based reconstruction
    rather than a deformation field. Its 'deformation' key may be None, and
    warping is done through affinity matrices instead of grid_sample.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, source, target):
        """
        Register source to target.

        Args:
            source: (B, C, H, W) source/moving image
            target: (B, C, H, W) target/fixed image

        Returns:
            dict with keys:
                'warped_source': (B, C, H, W) source warped toward target
                'deformation':   (B, 2, H, W) displacement field, or None for Sli2Vol
        """
        raise NotImplementedError

    @abstractmethod
    def warp(self, image, deformation_field):
        """
        Apply a deformation field to warp an image or mask.

        Args:
            image:             (B, C, H, W) image to warp
            deformation_field: (B, 2, H, W) displacement field

        Returns:
            warped: (B, C, H, W) warped image
        """
        raise NotImplementedError
