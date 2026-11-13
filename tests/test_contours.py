"""Unit tests for contours/anchors.py and contours/raster.py on synthetic masks.
CPU-only, no data. Run:  python -m pytest tests/test_contours.py -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from contours import anchors as A
from contours import raster as R


# ---------------------------------------------------------------- fixtures
def ellipsoid(shape=(40, 64, 64), center=(20, 32, 30), radii=(14, 22, 16)):
    Z, H, W = shape
    z, y, x = np.ogrid[:Z, :H, :W]
    cz, cy, cx = center
    rz, ry, rx = radii
    return ((((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0)


def disk(shape=(64, 64), center=(32, 30), radius=15):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return ((y - center[0]) ** 2 + (x - center[1]) ** 2) <= radius ** 2


def dice_np(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


# ---------------------------------------------------------------- anchors
def test_known_volume_matches_gt_on_seed_planes():
    gt = ellipsoid()
    seeds = A.seeds_from_gt(gt.astype(np.int8))
    assert set(seeds) == {0, 1, 2}
    known = A.build_known_volume(gt.shape, seeds)
    for ax, (idx, m) in seeds.items():
        kp = A.permute_to_axis(known, ax)
        gp = A.permute_to_axis(gt.astype(np.int8), ax)
        assert idx == int(np.argmax(gp.reshape(gp.shape[0], -1).sum(1)))
        assert np.array_equal(kp[idx], gp[idx])          # fully known seed plane
    # everywhere else: exactly the two lines are known in an axial slice
    s_ax, s_cor, s_sag = seeds[0][0], seeds[1][0], seeds[2][0]
    z = s_ax + 5
    sl = known[z]
    assert (sl[s_cor, :] != A.UNKNOWN).all() and (sl[:, s_sag] != A.UNKNOWN).all()
    n_known = (sl != A.UNKNOWN).sum()
    assert n_known == sl.shape[0] + sl.shape[1] - 1


def test_axial_anchor_points_lie_on_gt_boundary():
    gt = ellipsoid()
    seeds = A.seeds_from_gt(gt.astype(np.int8))
    known = A.build_known_volume(gt.shape, seeds)
    per_slice = A.anchors_for_axis(known, axis=0, exclude_seed_slice=seeds[0][0])
    z_fg = [z for z in range(gt.shape[0]) if gt[z].any()]
    checked = 0
    for z in z_fg:
        sa = per_slice[z]
        if sa is None:
            continue
        gt_slice = gt[z].astype(np.float32)
        for (r, c), d in zip(sa.points, sa.line_dirs):
            # crossing is midway between two pixels of different label
            if d == 0:   # row line -> c is half-integer
                lo, hi = int(np.floor(c)), int(np.ceil(c))
                assert gt_slice[int(r), lo] != gt_slice[int(r), hi]
            else:
                lo, hi = int(np.floor(r)), int(np.ceil(r))
                assert gt_slice[lo, int(c)] != gt_slice[hi, int(c)]
            checked += 1
        # a convex object: exactly 2 crossings per line that hits it
        for d in (0, 1):
            n = int((sa.line_dirs == d).sum())
            assert n in (0, 2), (z, d, n)
        assert sa.n_fg_segments <= 2
    assert checked > 50


def test_coronal_and_sagittal_anchors_consistent():
    gt = ellipsoid()
    seeds = A.seeds_from_gt(gt.astype(np.int8))
    known = A.build_known_volume(gt.shape, seeds)
    for ax in (1, 2):
        gp = A.permute_to_axis(gt.astype(np.int8), ax)
        per_slice = A.anchors_for_axis(known, axis=ax, exclude_seed_slice=seeds[ax][0])
        n_pts = 0
        for k, sa in enumerate(per_slice):
            if sa is None:
                continue
            # every known label equals the GT in the permuted frame
            kn = sa.label_map != A.UNKNOWN
            assert np.array_equal(sa.label_map[kn], gp[k][kn])
            n_pts += sa.n_points
        assert n_pts > 20


def test_extent_is_lower_bound_of_true_extent():
    gt = ellipsoid()
    seeds = A.seeds_from_gt(gt.astype(np.int8))
    known = A.build_known_volume(gt.shape, seeds)
    for ax in (0, 1, 2):
        ext = A.fg_extent_along_axis(known, ax)
        gp = A.permute_to_axis(gt, ax)
        true_idx = np.flatnonzero(gp.reshape(gp.shape[0], -1).any(1))
        assert ext is not None
        assert true_idx[0] <= ext[0] and ext[1] <= true_idx[-1]


def test_anchor_agreement_rewards_correct_prediction():
    gt = ellipsoid()
    seeds = A.seeds_from_gt(gt.astype(np.int8))
    known = A.build_known_volume(gt.shape, seeds)
    z = seeds[0][0] + 4
    lm = known[z]
    assert A.anchor_agreement(gt[z], lm) == 1.0
    shifted = np.roll(gt[z], 8, axis=1)
    assert A.anchor_agreement(shifted, lm) < 1.0
    assert A.anchor_agreement(gt[z], np.full_like(lm, A.UNKNOWN)) is None


def test_two_components_give_more_fg_segments():
    m = np.zeros((64, 64), np.int8)
    m[20:30, 10:20] = 1
    m[20:30, 40:50] = 1
    lm = np.full((64, 64), A.UNKNOWN, np.int8)
    lm[25, :] = m[25, :]          # one row line crossing both blobs
    sa = A.extract_slice_anchors(lm)
    assert sa.n_fg_segments == 2
    assert sa.n_points == 4


def test_crossing_angle_filter():
    poly = R.resample_closed(np.array([[10, 10], [10, 50], [50, 50], [50, 10]], float), 64)
    # on the top edge (row 10) the contour runs along cols -> tangent to a row line
    assert not A.crossing_angle_ok(poly, np.array([10.0, 30.0]), line_dir=0)
    # same point vs a column line: perpendicular -> ok
    assert A.crossing_angle_ok(poly, np.array([10.0, 30.0]), line_dir=1)


# ---------------------------------------------------------------- raster (numpy)
def test_mask_polygon_roundtrip():
    m = disk()
    polys = R.mask_to_polygons(m, n_points=64)
    assert len(polys) == 1
    p = polys[0]
    assert p.shape == (64, 2)
    back = R.polygon_to_mask(p, m.shape)
    assert dice_np(back, m) > 0.97
    assert abs(R.polygon_area(p) - m.sum()) / m.sum() < 0.05
    assert not R.has_self_intersection(p)


def test_two_blobs_two_polygons():
    m = np.zeros((64, 64), bool)
    m |= disk(center=(20, 16), radius=8)
    m |= disk(center=(44, 48), radius=10)
    polys = R.mask_to_polygons(m, n_points=32)
    assert len(polys) == 2
    assert dice_np(R.polygons_to_mask(polys, m.shape), m) > 0.95


def test_hole_is_ignored():
    m = disk(radius=20) & ~disk(radius=8)
    polys = R.mask_to_polygons(m, n_points=64)
    assert len(polys) == 1              # outer boundary only


def test_resample_equal_spacing():
    sq = np.array([[0, 0], [0, 40], [40, 40], [40, 0]], float)
    p = R.resample_closed(sq, 80)
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    assert np.allclose(seg, 2.0, atol=1e-4)


def test_self_intersection_detected():
    bow = np.array([[0, 0], [40, 40], [40, 0], [0, 40]], float)
    assert R.has_self_intersection(bow)


# ---------------------------------------------------------------- raster (torch)
def test_soft_rasterize_matches_hard():
    m = disk()
    p = torch.tensor(R.mask_to_polygons(m, 64)[0])
    soft = R.soft_rasterize(p, m.shape, sigma=0.5)
    assert soft.shape == m.shape
    assert dice_np((soft > 0.5).numpy(), m) > 0.97
    # inside is ~1, outside ~0
    assert soft[32, 30] > 0.95 and soft[2, 2] < 0.05


def test_signed_distance_sign_and_magnitude():
    sq = torch.tensor([[10., 10.], [10., 50.], [50., 50.], [50., 10.]])
    sdf = R.signed_distance(sq, (64, 64))
    assert sdf[30, 30] < 0 and abs(sdf[30, 30].item() + 20.0) < 1e-4   # centre, 20 px inside
    assert abs(sdf[30, 55].item() - 5.0) < 1e-4                         # 5 px outside right edge
    assert abs(sdf[10, 30].item()) < 1e-4                               # on the boundary


def test_soft_rasterize_is_differentiable_and_dice_improves():
    m = torch.tensor(disk(), dtype=torch.float32)
    # start from a shifted / shrunk polygon, optimise soft Dice to the target
    p0 = R.mask_to_polygons(disk(center=(28, 36), radius=11), 48)[0]
    p = torch.tensor(p0, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.5)
    d_first = None
    for it in range(60):
        soft = R.soft_rasterize(p, m.shape, sigma=1.0)
        loss = 1.0 - R.soft_dice(soft, m)
        if d_first is None:
            d_first = 1.0 - loss.item()
        opt.zero_grad(); loss.backward()
        assert torch.isfinite(p.grad).all()
        opt.step()
    d_last = R.soft_dice(R.soft_rasterize(p.detach(), m.shape, 1.0), m).item()
    assert d_last > d_first + 0.15, (d_first, d_last)


def test_chamfer_zero_for_identical_and_grows_with_shift():
    p = torch.tensor(R.mask_to_polygons(disk(), 64)[0])
    assert R.chamfer_distance(p, p).item() < 1e-2  # float32 cdist round-off
    q = p + torch.tensor([3.0, 0.0])
    d = R.chamfer_distance(p, q).item()
    assert 0.5 < d <= 3.0 + 1e-4   # <= shift; smaller along the tangent direction
