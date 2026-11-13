"""
Spatial transformer utilities for 2D registration.

Converts displacement fields to sampling grids and warps images/masks
using torch.nn.functional.grid_sample.
"""

import torch
import torch.nn.functional as F


class SpatialTransformer(torch.nn.Module):
    """
    Warps a 2D image by a displacement field using grid_sample.

    The displacement field is in pixel units (not normalized).
    """

    def __init__(self, size, mode='bilinear', padding_mode='border'):
        """
        Args:
            size: (H, W) spatial dimensions
            mode: interpolation mode ('bilinear' for images, 'nearest' for masks)
            padding_mode: 'border', 'zeros', or 'reflection'
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

        # Build identity grid: (1, H, W, 2) in pixel coordinates
        H, W = size
        vectors = [torch.arange(0, s, dtype=torch.float32) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')  # H, W
        grid = torch.stack(grids, dim=-1)  # (H, W, 2)  [y, x]
        grid = grid.unsqueeze(0)  # (1, H, W, 2)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        """
        Args:
            src:  (B, C, H, W) image to warp
            flow: (B, 2, H, W) displacement field (dy, dx) in pixel units

        Returns:
            warped: (B, C, H, W)
        """
        # flow: (B, 2, H, W) -> (B, H, W, 2)
        flow = flow.permute(0, 2, 3, 1)

        # Add flow to identity grid: new_grid[..., 0] = y + dy, new_grid[..., 1] = x + dx
        new_grid = self.grid + flow

        # Normalize to [-1, 1] for grid_sample
        H, W = src.shape[2], src.shape[3]
        new_grid_normalized = new_grid.clone()
        # grid_sample expects (x, y) order, so swap
        new_grid_normalized[..., 0] = 2.0 * new_grid[..., 1] / (W - 1) - 1.0  # x
        new_grid_normalized[..., 1] = 2.0 * new_grid[..., 0] / (H - 1) - 1.0  # y

        return F.grid_sample(
            src, new_grid_normalized,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )


def warp_image(src, flow, mode='bilinear', padding_mode='border'):
    """
    Functional version: warp src by flow without pre-built grid.

    Args:
        src:  (B, C, H, W) image to warp
        flow: (B, 2, H, W) displacement field (dy, dx) in pixel units
        mode: 'bilinear' for images, 'nearest' for masks
        padding_mode: 'border', 'zeros', or 'reflection'

    Returns:
        warped: (B, C, H, W)
    """
    B, _, H, W = flow.shape

    # Build identity grid
    gy = torch.arange(H, dtype=flow.dtype, device=flow.device)
    gx = torch.arange(W, dtype=flow.dtype, device=flow.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')

    # Add displacement
    new_y = grid_y.unsqueeze(0) + flow[:, 0]  # (B, H, W)
    new_x = grid_x.unsqueeze(0) + flow[:, 1]  # (B, H, W)

    # Normalize to [-1, 1]
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    # grid_sample expects (B, H, W, 2) with (x, y)
    grid = torch.stack([new_x, new_y], dim=-1)

    return F.grid_sample(
        src, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
