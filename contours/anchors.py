"""
Anchors from orthogonal seed slices.

The three seed masks (one per axis) intersect every slice of every other axis.
In canonical (Z, H, W) space:

    axial seed    at z = s_ax  -> label known on the whole plane  [s_ax, :, :]
    coronal seed  at y = s_cor -> label known on the plane        [:, s_cor, :]
    sagittal seed at x = s_sag -> label known on the plane        [:, :, s_sag]

So in an axial slice z != s_ax the label is known exactly on one row (y = s_cor)
and one column (x = s_sag). The same holds, with the roles permuted, for every
coronal and sagittal slice. We represent all of this with ONE sparse volume,
`known`, holding 1 (foreground), 0 (background) or -1 (unknown), and read it
out per slice in the propagation axis's permuted frame. This mirrors
`_permute_to_axis` in eval_triplanar.py, so the per-slice anchors line up with
the per-slice images the propagator sees.

Per slice we extract (SliceAnchors):
    points     : (P, 2) float boundary crossings in (row, col) pixel coords of
                 the permuted slice, sub-pixel (midway between the two pixels
                 whose labels differ)
    line_dirs  : (P,) int, 0 if the crossing was found on a row line
                 (direction along cols), 1 if on a column line
    label_map  : (A, B) int8 of {-1, 0, 1}, the known labels in this slice
    n_fg_segments : number of contiguous foreground runs over both lines
                    (lower bound on the number of connected components)
    has_fg     : whether any known foreground exists in this slice

Only the propagation axis's OWN seed slice is fully known; every other slice
has at most two lines. The seed slice itself is excluded from anchor
extraction by default (the propagator starts from it anyway).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

UNKNOWN, BG, FG = -1, 0, 1

# Same permutations as eval_triplanar.py so slice k here is slice k there.
_AXIS_PERMS = {
    0: ((0, 1, 2), (0, 1, 2)),
    1: ((1, 0, 2), (1, 0, 2)),
    2: ((2, 0, 1), (1, 2, 0)),
}
AXIS_NAMES = {0: "axial", 1: "coronal", 2: "sagittal"}
AXIS_INDEX = {v: k for k, v in AXIS_NAMES.items()}


def permute_to_axis(vol_zhw: np.ndarray, axis: int) -> np.ndarray:
    fwd, _ = _AXIS_PERMS[axis]
    return np.ascontiguousarray(np.transpose(vol_zhw, fwd))


def permute_from_axis(vol_perm: np.ndarray, axis: int) -> np.ndarray:
    _, inv = _AXIS_PERMS[axis]
    return np.ascontiguousarray(np.transpose(vol_perm, inv))


# ----------------------------------------------------------------------------
# Building the sparse known-label volume
# ----------------------------------------------------------------------------

def build_known_volume(shape_zhw: Tuple[int, int, int],
                       seeds: Dict[int, Tuple[int, np.ndarray]]) -> np.ndarray:
    """
    shape_zhw : (Z, H, W)
    seeds     : {axis: (index, mask2d)} in canonical space, mask2d being the
                seed slice as the propagator sees it for that axis:
                  axis 0 -> mask2d is (H, W)   at z = index
                  axis 1 -> mask2d is (Z, W)   at y = index
                  axis 2 -> mask2d is (Z, H)   at x = index
                (these are exactly gt_perm[seed_idx] from eval_triplanar.py).
    Returns int8 volume (Z, H, W) with values in {-1, 0, 1}.
    """
    Z, H, W = shape_zhw
    known = np.full((Z, H, W), UNKNOWN, dtype=np.int8)
    for axis, (idx, m) in seeds.items():
        m = (np.asarray(m) > 0).astype(np.int8)
        if axis == 0:
            assert m.shape == (H, W), (m.shape, (H, W))
            known[idx, :, :] = m
        elif axis == 1:
            assert m.shape == (Z, W), (m.shape, (Z, W))
            known[:, idx, :] = m
        elif axis == 2:
            assert m.shape == (Z, H), (m.shape, (Z, H))
            known[:, :, idx] = m
        else:
            raise ValueError(f"bad axis {axis}")
    return known


def seeds_from_gt(gt_zhw: np.ndarray, axes=(0, 1, 2),
                  select_fn=None) -> Dict[int, Tuple[int, np.ndarray]]:
    """Pick one seed per axis from a GT volume with the paper's rule
    (argmax cross-sectional area). Returns {axis: (index, mask2d)}."""
    if select_fn is None:
        def select_fn(gt_perm):
            areas = gt_perm.reshape(gt_perm.shape[0], -1).sum(axis=1)
            return None if areas.max() == 0 else int(np.argmax(areas))
    seeds = {}
    for ax in axes:
        gp = permute_to_axis(gt_zhw, ax)
        idx = select_fn(gp)
        if idx is not None:
            seeds[ax] = (idx, gp[idx].astype(np.int8))
    return seeds


# ----------------------------------------------------------------------------
# Per-slice extraction
# ----------------------------------------------------------------------------

@dataclass
class SliceAnchors:
    points: np.ndarray                      # (P, 2) float32, (row, col)
    line_dirs: np.ndarray                   # (P,) int8, 0=row line, 1=col line
    label_map: np.ndarray                   # (A, B) int8 in {-1,0,1}
    n_fg_segments: int
    has_fg: bool
    fg_segments: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # each fg segment as (line_dir, line_index, start, end_inclusive)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])


def _runs(labels_1d: np.ndarray):
    """Contiguous runs of a 1-D int array -> list of (value, start, end_incl)."""
    if labels_1d.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels_1d)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [labels_1d.size - 1]])
    return [(int(labels_1d[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _crossings_on_line(labels_1d: np.ndarray) -> List[float]:
    """Sub-pixel positions where a known label flips 0<->1 along a line.
    Unknown (-1) entries break the line: no crossing is reported across them."""
    out = []
    for i in range(labels_1d.size - 1):
        a, b = labels_1d[i], labels_1d[i + 1]
        if a == UNKNOWN or b == UNKNOWN or a == b:
            continue
        out.append(i + 0.5)
    return out


def extract_slice_anchors(label_map: np.ndarray,
                          min_known_fraction: float = 0.0) -> SliceAnchors:
    """
    label_map: (A, B) int8 in {-1, 0, 1} for one slice in the propagation
    axis's permuted frame. Lines are any row / column that is (mostly) known.
    """
    A, B = label_map.shape
    known = label_map != UNKNOWN
    row_known = known.mean(axis=1)   # fraction of known pixels per row
    col_known = known.mean(axis=0)

    pts, dirs, fg_segs = [], [], []

    # Rows that are known lines (coronal/sagittal seed intersections give
    # fully-known lines; threshold lets partially known lines through).
    for r in np.flatnonzero(row_known > max(min_known_fraction, 0.5)):
        line = label_map[r]
        for c in _crossings_on_line(line):
            pts.append((float(r), c)); dirs.append(0)
        for val, s, e in _runs(line):
            if val == FG:
                fg_segs.append((0, int(r), s, e))
    for c in np.flatnonzero(col_known > max(min_known_fraction, 0.5)):
        line = label_map[:, c]
        for r in _crossings_on_line(line):
            pts.append((r, float(c))); dirs.append(1)
        for val, s, e in _runs(line):
            if val == FG:
                fg_segs.append((1, int(c), s, e))

    points = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    line_dirs = np.asarray(dirs, dtype=np.int8)
    has_fg = bool((label_map == FG).any())
    return SliceAnchors(points=points, line_dirs=line_dirs, label_map=label_map,
                        n_fg_segments=len(fg_segs), has_fg=has_fg,
                        fg_segments=fg_segs)


def anchors_for_axis(known_zhw: np.ndarray, axis: int,
                     exclude_seed_slice: Optional[int] = None
                     ) -> List[Optional[SliceAnchors]]:
    """
    Per-slice anchors along `axis`. Returns a list of length K (slices along
    the axis); entries are SliceAnchors, or None for the excluded seed slice.
    Fully-known slices (the axis's own seed) carry no useful *anchor* lines,
    so they are returned with the full label_map but, if excluded, as None.
    """
    kp = permute_to_axis(known_zhw, axis)
    out = []
    for k in range(kp.shape[0]):
        if exclude_seed_slice is not None and k == exclude_seed_slice:
            out.append(None)
            continue
        out.append(extract_slice_anchors(kp[k]))
    return out


# ----------------------------------------------------------------------------
# Whole-volume summaries that the propagator can use
# ----------------------------------------------------------------------------

def fg_extent_along_axis(known_zhw: np.ndarray, axis: int) -> Optional[Tuple[int, int]]:
    """Lower bound on the organ's extent along `axis`: first and last slice
    (in the permuted frame) with any known foreground. None if there is none."""
    kp = permute_to_axis(known_zhw, axis)
    has = (kp == FG).reshape(kp.shape[0], -1).any(axis=1)
    idx = np.flatnonzero(has)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def anchor_agreement(pred_slice: np.ndarray, label_map: np.ndarray) -> Optional[float]:
    """Fraction of KNOWN pixels in this slice that a binary prediction gets
    right. This is the label-free per-chain reliability signal r_c(k).
    Returns None if nothing is known in the slice."""
    known = label_map != UNKNOWN
    n = int(known.sum())
    if n == 0:
        return None
    pred = (np.asarray(pred_slice) > 0.5).astype(np.int8)
    return float((pred[known] == label_map[known]).mean())


def crossing_angle_ok(polygon_rc: np.ndarray, point_rc: np.ndarray,
                      line_dir: int, min_deg: float = 15.0) -> bool:
    """
    Reject an anchor if the contour crosses the anchor line at a grazing
    angle (tangent), where the crossing position is unreliable. Angle is
    measured between the contour tangent at the nearest polygon vertex and
    the anchor line direction. polygon_rc: (N, 2) closed polygon (row, col).
    """
    if polygon_rc.shape[0] < 3:
        return True
    d = np.linalg.norm(polygon_rc - point_rc[None], axis=1)
    i = int(np.argmin(d))
    tangent = polygon_rc[(i + 1) % len(polygon_rc)] - polygon_rc[i - 1]
    tn = np.linalg.norm(tangent)
    if tn < 1e-6:
        return True
    tangent = tangent / tn
    line = np.array([0.0, 1.0]) if line_dir == 0 else np.array([1.0, 0.0])
    cosang = abs(float(np.dot(tangent, line)))
    ang = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    return ang >= min_deg
