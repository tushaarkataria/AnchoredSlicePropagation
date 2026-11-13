"""
Polygon <-> mask tools for contour propagation.

Coordinates are (row, col) floats in pixel units, matching numpy indexing of
the (A, B) slice. Polygons are closed implicitly (last vertex connects to
first) and stored as (N, 2) arrays.

numpy side (extraction / hard rasterisation / checks):
    mask_to_polygons(mask, n_points)      -> list of (N, 2) polygons
    resample_closed(poly, n_points)       -> (n_points, 2) equal arc-length
    polygon_to_mask(poly, shape)          -> bool (A, B)
    polygon_area(poly)                    -> float (positive, shoelace)
    has_self_intersection(poly)           -> bool (O(N^2), for tests / QC)

torch side (differentiable):
    soft_rasterize(poly, shape, sigma)    -> (A, B) in [0, 1]
        sigmoid(-sdf / sigma) with sdf the signed distance to the polygon;
        differentiable w.r.t. the vertices through the distance magnitude
        (the inside/outside sign is piecewise constant, as in SoftRas-style
        rasterisers).
    chamfer_distance(p, q)                -> scalar, symmetric mean nearest-
                                             neighbour distance
    soft_dice(soft_mask, target)          -> scalar in [0, 1]
"""

from typing import List, Tuple

import numpy as np
import torch
from skimage import measure
from skimage.draw import polygon as _sk_polygon


# ----------------------------------------------------------------------------
# numpy: extraction, resampling, hard rasterisation
# ----------------------------------------------------------------------------

def resample_closed(poly: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed polygon to n_points equally spaced in arc length,
    starting from the original first vertex."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.repeat(poly, n_points, axis=0)
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(poly[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, 2))
    for d in range(2):
        out[:, d] = np.interp(targets, cum, closed[:, d])
    return out.astype(np.float32)


def mask_to_polygons(mask: np.ndarray, n_points: int = 64,
                     min_area: float = 4.0) -> List[np.ndarray]:
    """Closed iso-contours of a binary mask at level 0.5, each resampled to
    n_points vertices. Small contours (area < min_area px) and holes
    (negative orientation relative to the outer contour) are dropped: we
    track outer boundaries only."""
    m = (np.asarray(mask) > 0).astype(np.float32)
    if m.sum() == 0:
        return []
    # pad so contours touching the border close properly
    mp = np.pad(m, 1, mode="constant")
    contours = measure.find_contours(mp, 0.5)
    polys = []
    for c in contours:
        c = c - 1.0  # undo padding
        if c.shape[0] < 4:
            continue
        if not np.allclose(c[0], c[-1]):
            continue  # open contour (should not happen after padding)
        c = c[:-1]
        if polygon_area(c) < min_area:
            continue
        # outer boundary vs hole: the region enclosed by an outer boundary is
        # mostly foreground, the region enclosed by a hole mostly background.
        inside = polygon_to_mask(c, m.shape)
        if inside.sum() == 0 or m[inside].mean() < 0.5:
            continue
        polys.append(resample_closed(c, n_points))
    return polys


def _signed_area(poly: np.ndarray) -> float:
    r, c = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(r, np.roll(c, -1)) - np.dot(c, np.roll(r, -1)))


def polygon_area(poly: np.ndarray) -> float:
    return abs(_signed_area(np.asarray(poly, dtype=np.float64)))


def polygon_to_mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Hard rasterisation (even-odd fill of one polygon)."""
    poly = np.asarray(poly, dtype=np.float64)
    out = np.zeros(shape, dtype=bool)
    if poly.shape[0] < 3:
        return out
    rr, cc = _sk_polygon(poly[:, 0], poly[:, 1], shape=shape)
    out[rr, cc] = True
    return out


def polygons_to_mask(polys: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for p in polys:
        out |= polygon_to_mask(p, shape)
    return out


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])
    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def has_self_intersection(poly: np.ndarray) -> bool:
    """O(N^2) proper-crossing test between non-adjacent edges."""
    poly = np.asarray(poly, dtype=np.float64)
    n = poly.shape[0]
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through the closure
            b1, b2 = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


# ----------------------------------------------------------------------------
# torch: differentiable rasterisation and losses
# ----------------------------------------------------------------------------

def _pixel_grid(shape: Tuple[int, int], device, dtype):
    A, B = shape
    rr = torch.arange(A, device=device, dtype=dtype)
    cc = torch.arange(B, device=device, dtype=dtype)
    gr, gc = torch.meshgrid(rr, cc, indexing="ij")
    return torch.stack([gr, gc], dim=-1).reshape(-1, 2)  # (A*B, 2)


def point_to_polygon_distance(points: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Unsigned distance from each of P points to the closed polygon's
    boundary. points: (P, 2), poly: (N, 2). Returns (P,)."""
    a = poly                               # (N, 2)
    b = torch.roll(poly, shifts=-1, dims=0)
    ab = b - a                             # (N, 2)
    ab2 = (ab * ab).sum(-1).clamp_min(1e-12)  # (N,)
    ap = points[:, None, :] - a[None, :, :]   # (P, N, 2)
    t = ((ap * ab[None]).sum(-1) / ab2[None]).clamp(0.0, 1.0)  # (P, N)
    proj = a[None] + t[..., None] * ab[None]                   # (P, N, 2)
    d2 = ((points[:, None, :] - proj) ** 2).sum(-1)            # (P, N)
    return d2.min(dim=1).values.clamp_min(1e-12).sqrt()


def points_inside_polygon(points: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Even-odd (crossing number) inside test. Non-differentiable (bool)."""
    a = poly
    b = torch.roll(poly, shifts=-1, dims=0)
    pr, pc = points[:, 0:1], points[:, 1:2]            # (P,1)
    ar, ac = a[None, :, 0], a[None, :, 1]              # (1,N)
    br, bc = b[None, :, 0], b[None, :, 1]
    cond = (ar > pr) != (br > pr)                      # edge straddles the row
    # col of the edge at this row
    denom = (br - ar)
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    x_int = ac + (pr - ar) * (bc - ac) / denom
    cross = cond & (pc < x_int)
    return (cross.sum(dim=1) % 2) == 1


def signed_distance(poly: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Signed distance field (A, B): negative inside, positive outside."""
    pts = _pixel_grid(shape, poly.device, poly.dtype)
    d = point_to_polygon_distance(pts, poly)
    inside = points_inside_polygon(pts.detach(), poly.detach())
    sdf = torch.where(inside, -d, d)
    return sdf.reshape(shape)


def soft_rasterize(poly: torch.Tensor, shape: Tuple[int, int],
                   sigma: float = 1.0) -> torch.Tensor:
    """Soft mask in [0, 1]: sigmoid(-sdf / sigma). Differentiable w.r.t. poly."""
    sdf = signed_distance(poly, shape)
    return torch.sigmoid(-sdf / sigma)


def soft_rasterize_many(polys: List[torch.Tensor], shape: Tuple[int, int],
                        sigma: float = 1.0) -> torch.Tensor:
    """Union of several soft masks: 1 - prod(1 - m_i)."""
    if len(polys) == 0:
        return torch.zeros(shape)
    out = torch.ones(shape, device=polys[0].device, dtype=polys[0].dtype)
    for p in polys:
        out = out * (1.0 - soft_rasterize(p, shape, sigma))
    return 1.0 - out


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric mean nearest-neighbour distance between point sets (P,2),(Q,2)."""
    d = torch.cdist(p, q)  # (P, Q)
    return 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def soft_dice(soft_mask: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    t = target.to(soft_mask.dtype)
    inter = (soft_mask * t).sum()
    return (2.0 * inter + eps) / (soft_mask.sum() + t.sum() + eps)
