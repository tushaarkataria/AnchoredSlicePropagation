"""
No-retraining transport operators on top of a trained Sli2Vol affinity.

Sli2Vol's correlation layer gives, for every TARGET pixel (y, x), a softmax
distribution over SOURCE pixels in a (2R+1)^2 window:

    corr[d, y, x],  d = ky*D + kx,  D = 2R+1,
    source pixel   = (y + ky - R, x + kx - R).

Three ways to move a mask from slice k to k+1 with that same affinity:

  'mask'    : dense soft copy of the binary mask, threshold 0.5. This is what
              Sli2Vol itself does (one-hot copy + argmax); the reference row.
  'sdf'     : dense soft copy of a truncated signed-distance field, threshold
              at 0, then RE-INITIALISE the SDF from the new mask. Averaging
              distances is better behaved than averaging indicators (the zero
              level set is all that matters, and |sdf| is a confidence), and
              re-initialisation stops the blur from accumulating.
  'ensemble': all three on the SAME affinity (which is the expensive part),
              each raw result turned into a truncated SDF, confidence-weighted
              mean, threshold at 0 = the consensus. With consensus_reinit every
              operator restarts from the consensus mask each step, so no single
              operator's bias (contour shrink, dense leak) can compound. The
              per-slice agreement of the three raw masks (mean pairwise Dice)
              is a LABEL-FREE chain reliability (self_agreement) that needs no
              anchors.
  'contour' : sparse. Read the affinity only at contour points, transposed
              into a FORWARD distribution per point (which target pixels point
              back at this source point), take its soft-argmax as the point's
              displacement and its peakiness as confidence, then fit a
              confidence-weighted affine + smooth residual motion along the
              contour (the shape constraint), rasterise.

Anchors from orthogonal seeds (contours.anchors) enter all three:
  dense variants : clamp the transported field on the known lines
                   (known FG -> inside, known BG -> outside) before threshold.
  contour        : snap the polygon's crossing with each anchor line to the
                   anchor point, with a Gaussian falloff along the contour.
The chain reliability r_c(k) is the agreement of the RAW transported mask
with the known labels, measured BEFORE the anchors are applied (so it is not
inflated by the correction it will receive).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_dilation

from contours import anchors as AN
from contours import raster as RA


# ----------------------------------------------------------------------------
# affinity access
# ----------------------------------------------------------------------------

@torch.no_grad()
def get_corr(model, src_hw: torch.Tensor, tgt_hw: torch.Tensor) -> torch.Tensor:
    """corr (D^2, H, W) for moving SOURCE -> TARGET. Inputs (H, W) tensors on
    the model's device, intensities as the test loader provides them."""
    out = model(src_hw[None, None], tgt_hw[None, None], return_corr=True)
    corr = out['corr_weights'][0]                      # (D^2, H', W')
    H, W = src_hw.shape
    if corr.shape[-2:] != (H, W):                      # safety; Sli2Vol is stride 1
        corr = F.interpolate(corr[None], (H, W), mode='bilinear', align_corners=False)[0]
        corr = corr / corr.sum(0, keepdim=True).clamp_min(1e-8)
    return corr


@torch.no_grad()
def get_corr_extra(model, src_hw: torch.Tensor, tgt_hw: torch.Tensor, n_scales: int) -> List[torch.Tensor]:
    """Coarser affinities for a multi-scale readout: the same model run on the
    slices block-averaged by 2, 4, ... (n_scales - 1 of them; [] for n_scales <= 1)."""
    out = []
    s, t = src_hw[None, None], tgt_hw[None, None]
    for _ in range(max(0, n_scales - 1)):
        s = F.avg_pool2d(s, 2); t = F.avg_pool2d(t, 2)
        out.append(get_corr(model, s[0, 0], t[0, 0]))
    return out


def _radius(corr: torch.Tensor) -> int:
    D = int(round(corr.shape[0] ** 0.5))
    assert D * D == corr.shape[0], corr.shape
    return (D - 1) // 2


@torch.no_grad()
def transport_dense(corr: torch.Tensor, field_hw: torch.Tensor) -> torch.Tensor:
    """tgt(y,x) = sum_d corr[d,y,x] * src(y+ky-R, x+kx-R). field_hw: (H, W)."""
    R = _radius(corr); D = 2 * R + 1
    H, W = field_hw.shape
    padded = F.pad(field_hw[None, None], [R] * 4, mode='replicate')
    patches = F.unfold(padded, kernel_size=D)          # (1, D^2, H*W), row-major (ky,kx)
    patches = patches.reshape(D * D, H, W)
    return (corr * patches).sum(0)


@torch.no_grad()
def forward_distribution_torch(corr: torch.Tensor, points_rc: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch version used by the trainable readout: returns (w (N, D^2)
    normalised forward distribution, offs (D^2, 2)). Gather positions are the
    rounded points (no gradient through the index); w itself is detached from
    the backbone (frozen)."""
    R = _radius(corr); D = 2 * R + 1
    _, H, W = corr.shape
    dev = corr.device
    p = torch.round(points_rc.detach()).long()
    p = torch.stack([p[:, 0].clamp(0, H - 1), p[:, 1].clamp(0, W - 1)], 1)
    oy, ox = torch.meshgrid(torch.arange(-R, R + 1, device=dev),
                            torch.arange(-R, R + 1, device=dev), indexing='ij')
    offs = torch.stack([oy.reshape(-1), ox.reshape(-1)], -1)
    d_idx = (R - offs[:, 0]) * D + (R - offs[:, 1])
    q = p[:, None, :] + offs[None, :, :]
    valid = (q[..., 0] >= 0) & (q[..., 0] < H) & (q[..., 1] >= 0) & (q[..., 1] < W)
    qr = q[..., 0].clamp(0, H - 1); qc = q[..., 1].clamp(0, W - 1)
    w = corr[d_idx[None, :].expand_as(qr), qr, qc]
    w = torch.where(valid, w, torch.zeros_like(w))
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return w, offs.to(w.dtype)


def point_forward_distribution(corr: torch.Tensor, points_rc: np.ndarray,
                               mode: str = 'soft') -> Tuple[np.ndarray, np.ndarray]:
    """
    For each SOURCE point p, the forward distribution over target offsets o:
        w(o) ~ corr[(R-oy)*D + (R-ox), p+o]   (target q=p+o assigns mass to p)
    Returns displacement (N,2) and confidence (N,) in [0,1]
    = 1 - normalised entropy of w.
    mode='soft'  : displacement = soft-argmax (expectation) over o. On a
                   curved boundary the expectation of an arc lies INSIDE the
                   arc, i.e. convex shapes get an inward bias of ~R^2/(2 r)
                   px per step (r = radius of curvature) that compounds.
    mode='argmax': displacement = mode of w (no curvature bias, noisier; the
                   residual smoothing in contour_step handles the noise).
    """
    R = _radius(corr); D = 2 * R + 1
    _, H, W = corr.shape
    dev = corr.device
    p = torch.as_tensor(np.rint(points_rc), dtype=torch.long, device=dev)   # (N,2)
    p[:, 0].clamp_(0, H - 1); p[:, 1].clamp_(0, W - 1)
    oy, ox = torch.meshgrid(torch.arange(-R, R + 1, device=dev),
                            torch.arange(-R, R + 1, device=dev), indexing='ij')
    offs = torch.stack([oy.reshape(-1), ox.reshape(-1)], -1)                 # (D^2,2)
    d_idx = (R - offs[:, 0]) * D + (R - offs[:, 1])                          # (D^2,)
    q = p[:, None, :] + offs[None, :, :]                                     # (N,D^2,2)
    valid = (q[..., 0] >= 0) & (q[..., 0] < H) & (q[..., 1] >= 0) & (q[..., 1] < W)
    qr = q[..., 0].clamp(0, H - 1); qc = q[..., 1].clamp(0, W - 1)
    w = corr[d_idx[None, :].expand_as(qr), qr, qc]                           # (N,D^2)
    w = torch.where(valid, w, torch.zeros_like(w))
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    if mode == 'argmax':
        disp = offs[w.argmax(1)].to(w.dtype)                                 # (N,2)
    else:
        disp = (w[..., None] * offs[None].to(w.dtype)).sum(1)                # (N,2)
    ent = -(w * (w + 1e-12).log()).sum(1) / np.log(D * D)
    conf = (1.0 - ent).clamp(0.0, 1.0)
    return disp.cpu().numpy().astype(np.float32), conf.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# sdf helpers
# ----------------------------------------------------------------------------

def sdf_from_mask(mask_hw: np.ndarray, trunc: float = 10.0) -> np.ndarray:
    """Signed distance (negative inside), truncated to [-trunc, trunc]."""
    m = mask_hw > 0
    if not m.any():
        return np.full(mask_hw.shape, trunc, np.float32)
    if m.all():
        return np.full(mask_hw.shape, -trunc, np.float32)
    inside = distance_transform_edt(m)
    outside = distance_transform_edt(~m)
    return np.clip(outside - inside, -trunc, trunc).astype(np.float32)


def soft_from_sdf(sdf_hw: np.ndarray, sigma: float) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(np.clip(sdf_hw / max(sigma, 1e-3), -30, 30)))).astype(np.float32)


# ----------------------------------------------------------------------------
# anchors applied to a dense field / a polygon
# ----------------------------------------------------------------------------

def clamp_field_with_anchors(field_hw: np.ndarray, label_map: Optional[np.ndarray],
                             inside_value: float, outside_value: float) -> np.ndarray:
    """Force known-FG pixels to be inside and known-BG pixels to be outside.
    For a probability field pass (1.0, 0.0); for an SDF pass (-0.5, +0.5)."""
    if label_map is None:
        return field_hw
    out = field_hw.copy()
    fg = label_map == AN.FG; bg = label_map == AN.BG
    if inside_value < outside_value:       # sdf-like: inside is smaller
        out[fg] = np.minimum(out[fg], inside_value)
        out[bg] = np.maximum(out[bg], outside_value)
    else:                                  # probability-like: inside is larger
        out[fg] = np.maximum(out[fg], inside_value)
        out[bg] = np.minimum(out[bg], outside_value)
    return out


def _line_crossings(poly: np.ndarray, line_dir: int, line_index: float) -> List[Tuple[int, float]]:
    """Edges of the closed polygon crossing row (line_dir=0) or col (1) at
    line_index. Returns [(edge_start_vertex, crossing_coord_along_line)]."""
    n = poly.shape[0]
    a = poly[:, line_dir]; b = np.roll(poly, -1, axis=0)[:, line_dir]
    other = 1 - line_dir
    oa = poly[:, other]; ob = np.roll(poly, -1, axis=0)[:, other]
    out = []
    for i in range(n):
        # half-open test so a vertex lying exactly on the line is counted once
        if (a[i] <= line_index < b[i]) or (b[i] <= line_index < a[i]):
            t = (line_index - a[i]) / (b[i] - a[i])
            out.append((i, float(oa[i] + t * (ob[i] - oa[i]))))
    return out


def snap_polygon_to_anchors(poly: np.ndarray, sa: Optional[AN.SliceAnchors],
                            max_snap: float = 12.0, falloff_pts: float = 3.0,
                            min_angle_deg: float = 15.0) -> Tuple[np.ndarray, int]:
    """Move the polygon so that it passes through each anchor point found on
    a line it crosses. Correction decays with a Gaussian in vertex index.
    Returns (new polygon, number of anchors used)."""
    if sa is None or sa.n_points == 0 or poly.shape[0] < 4:
        return poly, 0
    poly = poly.astype(np.float64).copy()
    n = poly.shape[0]
    used = 0
    idx = np.arange(n)
    for (ar, ac), d in zip(sa.points, sa.line_dirs):
        anchor = np.array([ar, ac])
        line_index = ar if d == 0 else ac
        along = 1 - d                       # coordinate that varies along the line
        target_along = anchor[along]
        crossings = _line_crossings(poly, d, line_index)
        if not crossings:
            continue
        i, c = min(crossings, key=lambda ic: abs(ic[1] - target_along))
        delta = target_along - c
        if abs(delta) > max_snap:
            continue
        if not AN.crossing_angle_ok(poly, anchor, int(d), min_angle_deg):
            continue
        # centre of the correction: the edge (i, i+1); circular index distance
        centre = i + 0.5
        dist = np.minimum(np.abs(idx - centre), n - np.abs(idx - centre))
        wgt = np.exp(-0.5 * (dist / falloff_pts) ** 2)
        poly[:, along] += wgt * delta
        used += 1
    return poly.astype(np.float32), used


def polygon_from_fg_segments(sa: AN.SliceAnchors, shape: Tuple[int, int],
                             n_points: int, grow: int = 2) -> List[np.ndarray]:
    """Re-seed: a small region around the known foreground segments."""
    m = np.zeros(shape, bool)
    for d, li, s, e in sa.fg_segments:
        if d == 0: m[li, s:e + 1] = True
        else:      m[s:e + 1, li] = True
    if not m.any():
        return []
    m = binary_dilation(m, iterations=grow)
    return RA.mask_to_polygons(m, n_points)


# ----------------------------------------------------------------------------
# one contour step
# ----------------------------------------------------------------------------

def _weighted_affine(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares affine A (2x3) with per-point weights; falls back to a
    weighted translation if the system is degenerate."""
    n = src.shape[0]
    X = np.hstack([src, np.ones((n, 1))])              # (n,3)
    sw = np.sqrt(np.clip(w, 1e-4, None))[:, None]
    try:
        A, *_ = np.linalg.lstsq(X * sw, dst * sw, rcond=None)   # (3,2)
        A = A.T
        if not np.all(np.isfinite(A)) or abs(np.linalg.det(A[:, :2])) < 0.2 \
                or abs(np.linalg.det(A[:, :2])) > 5.0:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        t = ((dst - src) * w[:, None]).sum(0) / max(w.sum(), 1e-6)
        A = np.array([[1, 0, t[0]], [0, 1, t[1]]], np.float64)
    return A


def _circular_smooth(values: np.ndarray, weights: np.ndarray, sigma_pts: float) -> np.ndarray:
    n = values.shape[0]
    i = np.arange(n)
    dist = np.abs(i[:, None] - i[None, :]); dist = np.minimum(dist, n - dist)
    K = np.exp(-0.5 * (dist / max(sigma_pts, 1e-3)) ** 2) * weights[None, :]
    K = K / K.sum(1, keepdims=True).clip(1e-8)
    return K @ values


def _rescale_about_centroid(poly: np.ndarray, s: float) -> np.ndarray:
    c = poly.mean(0, keepdims=True)
    return (poly - c) * s + c


def contour_step(corr: torch.Tensor, polys: List[np.ndarray], shape: Tuple[int, int],
                 n_points: int = 64, residual_weight: float = 1.0,
                 smooth_pts: Optional[float] = None, max_area_ratio: float = 1.6,
                 sa: Optional[AN.SliceAnchors] = None, use_anchors: bool = True,
                 disp_mode: str = 'soft', area_match: str = 'none',
                 area_clamp: Tuple[float, float] = (0.8, 1.25),
                 diag: Optional[list] = None
                 ) -> Tuple[List[np.ndarray], float, int]:
    """Move every polygon by the affinity. Returns (new polygons, mean conf,
    anchors used).
    disp_mode : 'soft' | 'argmax'  (see point_forward_distribution)
    area_match: 'none' | 'dense'. 'dense' transports each polygon's own mask
                densely with the same affinity (the paper's operator, which
                has no curvature bias: liver recall 0.97) and rescales the
                tracked polygon about its centroid so its area matches the
                dense area (per-step factor clamped to area_clamp). The
                contour keeps the shape/anchor machinery; the dense copy
                supplies the area. This is the diagnostic AND the fix for the
                systematic shrinkage of the soft-argmax tracker.
    diag      : if a list, one dict per polygon is appended with the step's
                area bookkeeping (a_prev, a_poly, a_dense, scale, det, conf).
    """
    new_polys, confs, used_total = [], [], 0
    for poly in polys:
        disp, conf = point_forward_distribution(corr, poly, mode=disp_mode)
        dst = poly + disp
        A = _weighted_affine(poly.astype(np.float64), dst.astype(np.float64), conf)
        aff = (A[:, :2] @ poly.T.astype(np.float64)).T + A[:, 2]
        resid = dst - aff
        s_pts = smooth_pts if smooth_pts is not None else max(2.0, poly.shape[0] / 16.0)
        resid_s = _circular_smooth(resid, conf, s_pts)
        new = aff + residual_weight * resid_s
        # area sanity: blend back toward the affine prediction if the step
        # changed the area implausibly
        a0, a1 = RA.polygon_area(poly), RA.polygon_area(new)
        if a0 > 0 and (a1 / a0 > max_area_ratio or a1 / a0 < 1.0 / max_area_ratio):
            new = aff
        a_dense, scale = float('nan'), 1.0
        if area_match == 'dense' or diag is not None:
            pm = torch.from_numpy(RA.polygon_to_mask(poly, shape).astype(np.float32)).to(corr.device)
            a_dense = float((transport_dense(corr, pm) > 0.5).sum().item())
        if area_match == 'dense':
            a_new = RA.polygon_area(new)
            if a_new > 0 and a_dense > 0:
                scale = float(np.clip(np.sqrt(a_dense / a_new), area_clamp[0], area_clamp[1]))
                new = _rescale_about_centroid(new, scale)
        if diag is not None:
            diag.append({'a_prev': float(a0), 'a_poly': float(RA.polygon_area(aff + residual_weight * resid_s)),
                         'a_dense': a_dense, 'scale': scale,
                         'det': float(np.linalg.det(A[:, :2])), 'conf': float(conf.mean())})
        if use_anchors:
            new, used = snap_polygon_to_anchors(new.astype(np.float32), sa)
            used_total += used
        new_polys.append(new.astype(np.float32))
        confs.append(float(conf.mean()))
    return new_polys, (float(np.mean(confs)) if confs else 0.0), used_total


def _readout_step(corr: torch.Tensor, poly: np.ndarray, readout, disp_mode: str, residual_weight: float,
                  corr_extra: Optional[List[torch.Tensor]] = None):
    """One learned step for one polygon. Returns (new polygon np (N,2),
    mean conf) or None if the readout says the object vanished."""
    from contours.readout import contour_step_torch
    with torch.no_grad():
        o = contour_step_torch(corr, torch.from_numpy(np.asarray(poly, np.float32)).to(corr.device),
                               readout, disp_mode=disp_mode, residual_weight=residual_weight,
                               corr_extra=corr_extra)
    if float(o['vanish']) > 0.0:
        return None
    return o['new'].cpu().numpy().astype(np.float32), float(o['conf'].mean())


# ----------------------------------------------------------------------------
# the propagator (same contract as propagate_mask_sli2vol, plus anchors)
# ----------------------------------------------------------------------------

@torch.no_grad()
def propagate_mask_affinity(model, img_zhw: np.ndarray, seed_idx: int, seed_mask: np.ndarray,
                            transport: str = 'contour',
                            known_perm: Optional[np.ndarray] = None,
                            use_anchors: bool = True,
                            n_points: int = 64,
                            sdf_trunc: float = 10.0,
                            soft_sigma: float = 1.0,
                            residual_weight: float = 1.0,
                            reseed_from_anchors: bool = False,
                            use_verification: bool = False,
                            device: str = 'cuda',
                            return_soft: bool = True,
                            return_reliability: bool = True,
                            disp_mode: str = 'soft',
                            area_match: str = 'none',
                            step_log: Optional[list] = None,
                            consensus_reinit: bool = True,
                            ens_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                            self_agreement: Optional[np.ndarray] = None,
                            readout=None):
    """
    img_zhw   : (K, A, B) float32 volume in the propagation axis's frame
    seed_idx  : slice index of the seed; seed_mask (A, B) int
    transport : 'mask' | 'sdf' | 'contour' | 'ensemble'
    ensemble  : consensus_reinit=True restarts mask/sdf/contour from the
                consensus every step (mutual regularisation); False lets each
                operator carry its own state and only the output is fused.
                ens_weights = (w_mask, w_sdf, w_contour). self_agreement: a
                (K,) float array filled IN PLACE with the mean pairwise Dice
                of the three raw masks per slice (nan at the seed / untouched).
    known_perm: (K, A, B) int8 {-1,0,1} known labels in the SAME frame
                (contours.anchors.permute_to_axis(known_zhw, axis)); None = no anchors
    disp_mode / area_match: see contour_step (contour transport only).
    readout   : trained contours.readout.ContourReadout; if given, the contour
                step is the torch step (hand-crafted + learned correction) and
                area_match is ignored. Polygons whose vanish logit > 0 are dropped.
    step_log  : if a list, per-step diagnostics are appended as dicts
                {'k': target slice, 'dir': +-1, 'a_gt_free': ..., per-polygon
                 entries from contour_step} (contour transport only).
    Returns pred (K,A,B) int32 [, soft (K,A,B) float32] [, rel (K,) float32 with
    nan where nothing is known].
    """
    assert transport in ('mask', 'sdf', 'contour', 'ensemble'), transport
    ens_w = np.asarray(ens_weights, np.float32); ens_w = ens_w / max(ens_w.sum(), 1e-6)
    K, A, B = img_zhw.shape
    pred = np.zeros((K, A, B), np.int32)
    soft = np.zeros((K, A, B), np.float32)
    rel = np.full((K,), np.nan, np.float32)
    pred[seed_idx] = (seed_mask > 0).astype(np.int32)
    soft[seed_idx] = pred[seed_idx].astype(np.float32)
    img = torch.from_numpy(np.ascontiguousarray(img_zhw)).float().to(device)

    if use_verification:
        from core_utils import verification_module

    def anchors_at(k) -> Optional[AN.SliceAnchors]:
        if known_perm is None or not use_anchors:
            return None
        return AN.extract_slice_anchors(known_perm[k])

    def label_map_at(k):
        return None if known_perm is None else known_perm[k]

    for direction in (-1, +1):
        rng = range(seed_idx, 0, -1) if direction < 0 else range(seed_idx, K - 1)
        cur_mask = pred[seed_idx].copy()
        polys = RA.mask_to_polygons(cur_mask, n_points) if transport in ('contour', 'ensemble') else None
        cur_sdf = sdf_from_mask(cur_mask, sdf_trunc) if transport in ('sdf', 'ensemble') else None
        own_mask = cur_mask.copy()                       # ensemble without re-init: each operator's own state
        template, mask_coll, img_coll = [], pred[seed_idx:seed_idx + 1].copy(), img_zhw[seed_idx:seed_idx + 1][:, None].copy()

        for i in rng:
            j = i + direction
            corr = get_corr(model, img[i], img[j])
            corr_extra = (get_corr_extra(model, img[i], img[j], readout.n_scales)
                          if readout is not None and getattr(readout, 'n_scales', 1) > 1
                          and transport in ('contour', 'ensemble') else None)
            sa = anchors_at(j); lm = label_map_at(j)

            if transport == 'mask':
                fld = transport_dense(corr, torch.from_numpy(cur_mask.astype(np.float32)).to(device)).cpu().numpy()
                raw = (fld > 0.5).astype(np.int32)
                if lm is not None: rel[j] = AN.anchor_agreement(raw, lm) or np.nan
                if use_anchors: fld = clamp_field_with_anchors(fld, lm, 1.0, 0.0)
                new_mask = (fld > 0.5).astype(np.int32)
                new_soft = np.clip(fld, 0, 1).astype(np.float32)

            elif transport == 'sdf':
                fld = transport_dense(corr, torch.from_numpy(cur_sdf).to(device)).cpu().numpy()
                raw = (fld < 0).astype(np.int32)
                if lm is not None: rel[j] = AN.anchor_agreement(raw, lm) or np.nan
                if use_anchors: fld = clamp_field_with_anchors(fld, lm, -0.5, 0.5)
                new_mask = (fld < 0).astype(np.int32)
                new_soft = soft_from_sdf(fld, soft_sigma)

            elif transport == 'ensemble':
                # 1) three raw transports on the shared affinity
                fld_m = transport_dense(corr, torch.from_numpy(own_mask.astype(np.float32)).to(device)).cpu().numpy()
                raw_m = (fld_m > 0.5).astype(np.int32)
                fld_s = transport_dense(corr, torch.from_numpy(cur_sdf).to(device)).cpu().numpy()
                raw_s = (fld_s < 0).astype(np.int32)
                if polys and readout is not None:                    # trained readout, same as the contour branch
                    new_polys = []
                    for p_ in polys:
                        o = _readout_step(corr, p_, readout, disp_mode, residual_weight, corr_extra)
                        if o is not None: new_polys.append(o[0])
                    raw_c = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32) if new_polys else np.zeros((A, B), np.int32)
                elif polys:
                    dg = [] if step_log is not None else None
                    new_polys, mconf, _ = contour_step(corr, polys, (A, B), n_points,
                                                       residual_weight=residual_weight,
                                                       sa=None, use_anchors=False,
                                                       disp_mode=disp_mode, area_match=area_match, diag=dg)
                    raw_c = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32)
                    if step_log is not None:
                        step_log.append({'k': int(j), 'dir': int(direction), 'polys': dg})
                else:
                    new_polys, raw_c = [], np.zeros((A, B), np.int32)
                raws = [raw_m, raw_s, raw_c]
                # 2) label-free agreement of the three raw masks (mean pairwise Dice)
                pair = []
                for a_ in range(3):
                    for b_ in range(a_ + 1, 3):
                        den = raws[a_].sum() + raws[b_].sum()
                        pair.append(2.0 * (raws[a_] & raws[b_]).sum() / den if den > 0 else 1.0)
                if self_agreement is not None: self_agreement[j] = float(np.mean(pair))
                # 3) consensus = weighted mean of truncated SDFs, threshold at 0
                d_cons = sum(w * sdf_from_mask(r_, sdf_trunc) for w, r_ in zip(ens_w, raws)).astype(np.float32)
                raw = (d_cons < 0).astype(np.int32)
                if lm is not None: rel[j] = AN.anchor_agreement(raw, lm) or np.nan
                if use_anchors: d_cons = clamp_field_with_anchors(d_cons, lm, -0.5, 0.5)
                new_mask = (d_cons < 0).astype(np.int32)
                new_soft = soft_from_sdf(d_cons, soft_sigma)
                if not consensus_reinit:
                    # each operator keeps its own (anchor-corrected) state
                    fm = clamp_field_with_anchors(fld_m, lm, 1.0, 0.0) if use_anchors else fld_m
                    own_mask = (fm > 0.5).astype(np.int32)
                    fs = clamp_field_with_anchors(fld_s, lm, -0.5, 0.5) if use_anchors else fld_s
                    cur_sdf = sdf_from_mask((fs < 0).astype(np.int32), sdf_trunc)
                    if use_anchors and sa is not None:
                        new_polys = [snap_polygon_to_anchors(p_, sa)[0] for p_ in new_polys]
                    polys = RA.mask_to_polygons(RA.polygons_to_mask(new_polys, (A, B)), n_points) if new_polys else []

            else:  # contour
                if polys and readout is not None:
                    new_polys, confs = [], []
                    for p_ in polys:
                        o = _readout_step(corr, p_, readout, disp_mode, residual_weight, corr_extra)
                        if o is None: continue
                        new_polys.append(o[0]); confs.append(o[1])
                    mconf = float(np.mean(confs)) if confs else 0.0
                    raw = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32)
                    if lm is not None: rel[j] = AN.anchor_agreement(raw, lm) or np.nan
                    if use_anchors and sa is not None:
                        new_polys = [snap_polygon_to_anchors(p_, sa)[0] for p_ in new_polys]
                    new_mask = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32)
                    sig = soft_sigma * (0.75 + 2.5 * (1.0 - mconf))
                elif polys:
                    dg = [] if step_log is not None else None
                    new_polys, mconf, _ = contour_step(corr, polys, (A, B), n_points,
                                                       residual_weight=residual_weight,
                                                       sa=None, use_anchors=False,
                                                       disp_mode=disp_mode, area_match=area_match,
                                                       diag=dg)
                    if step_log is not None:
                        step_log.append({'k': int(j), 'dir': int(direction), 'polys': dg})
                    raw = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32)
                    if lm is not None: rel[j] = AN.anchor_agreement(raw, lm) or np.nan
                    if use_anchors and sa is not None:
                        new_polys = [snap_polygon_to_anchors(p, sa)[0] for p in new_polys]
                    new_mask = RA.polygons_to_mask(new_polys, (A, B)).astype(np.int32)
                    sig = soft_sigma * (0.75 + 2.5 * (1.0 - mconf))
                else:
                    new_mask = np.zeros((A, B), np.int32); sig = soft_sigma
                new_soft = soft_from_sdf(sdf_from_mask(new_mask, sdf_trunc), sig)

            if use_verification and new_mask.any():
                new_mask, template = verification_module(new_mask, mask_coll, img_zhw[j:j + 1], img_coll, template)
                new_mask = new_mask.astype(np.int32)

            if not new_mask.any() and reseed_from_anchors and sa is not None and sa.n_fg_segments > 0:
                rp = polygon_from_fg_segments(sa, (A, B), n_points)
                new_mask = RA.polygons_to_mask(rp, (A, B)).astype(np.int32)
                new_soft = soft_from_sdf(sdf_from_mask(new_mask, sdf_trunc), soft_sigma)

            pred[j] = new_mask; soft[j] = new_soft
            mask_coll = np.concatenate([mask_coll, new_mask[None]], 0)
            img_coll = np.concatenate([img_coll, img_zhw[j:j + 1][:, None]], 0)

            if not new_mask.any():
                break
            cur_mask = new_mask
            if transport == 'contour':
                polys = RA.mask_to_polygons(cur_mask, n_points)   # re-extract: handles merge/split
            elif transport == 'sdf':
                cur_sdf = sdf_from_mask(cur_mask, sdf_trunc)       # re-initialise
            elif transport == 'ensemble' and consensus_reinit:
                own_mask = cur_mask.copy()
                cur_sdf = sdf_from_mask(cur_mask, sdf_trunc)
                polys = RA.mask_to_polygons(cur_mask, n_points)

    outs = [pred]
    if return_soft: outs.append(soft)
    if return_reliability: outs.append(rel)
    return tuple(outs) if len(outs) > 1 else pred
