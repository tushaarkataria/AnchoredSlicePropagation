"""
Trainable contour READOUT on top of the frozen Sli2Vol affinity (the
two-week-gate version of the design doc's tracker).

Per step, for every polygon point we already have (transport.py):
    w_i   (D^2,)  forward distribution over target offsets
    disp  = soft-argmax(w_i),  conf = 1 - entropy
and the hand-crafted step  new_i = affine(poly)_i + residual_smooth_i.

The readout sees the SAME inputs plus local geometry and predicts a
CORRECTION delta_i added to the hand-crafted position, a log-variance
(confidence) per point and a 'vanished' logit per polygon. At initialisation
delta = 0, so an untrained readout reproduces the hand-crafted tracker
exactly; training can only move away from it where the K-step unrolled loss
says so (fail-safe).

Everything here is torch and differentiable w.r.t. the polygon so the loss
can be unrolled over K steps (gradient flows through positions across steps;
the gather index is the only non-differentiable part).
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contours.transport import forward_distribution_torch, _radius


# ----------------------------------------------------------------------------
# hand-crafted step, torch version (matches contour_step's numpy path)
# ----------------------------------------------------------------------------

def soft_argmax_conf(w: torch.Tensor, offs: torch.Tensor, mode: str = 'soft'
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    D2 = w.shape[1]
    if mode == 'argmax':
        disp = offs[w.argmax(1)]
    else:
        disp = w @ offs                                            # (N,2)
    ent = -(w * (w + 1e-12).log()).sum(1) / np.log(D2)
    return disp, (1.0 - ent).clamp(0.0, 1.0)


def weighted_affine_torch(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(2,3) affine by weighted least squares; falls back to translation when
    degenerate (same rule as transport._weighted_affine)."""
    n = src.shape[0]
    X = torch.cat([src, torch.ones(n, 1, device=src.device, dtype=src.dtype)], 1)   # (n,3)
    sw = w.clamp_min(1e-4).sqrt()[:, None]
    try:
        A = torch.linalg.lstsq(X * sw, dst * sw).solution.T                           # (2,3)
        det = torch.linalg.det(A[:, :2])
        ok = torch.isfinite(A).all() and (det.abs() > 0.2) and (det.abs() < 5.0)
    except Exception:  # noqa
        ok = False
    if not ok:
        t = ((dst - src) * w[:, None]).sum(0) / w.sum().clamp_min(1e-6)
        A = torch.cat([torch.eye(2, device=src.device, dtype=src.dtype), t[:, None]], 1)
    return A


def circular_smooth_torch(values: torch.Tensor, weights: torch.Tensor, sigma_pts: float) -> torch.Tensor:
    n = values.shape[0]
    i = torch.arange(n, device=values.device, dtype=values.dtype)
    dist = (i[:, None] - i[None, :]).abs(); dist = torch.minimum(dist, n - dist)
    K = torch.exp(-0.5 * (dist / max(sigma_pts, 1e-3)) ** 2) * weights[None, :]
    K = K / K.sum(1, keepdim=True).clamp_min(1e-8)
    return K @ values


def handcrafted_step_torch(poly: torch.Tensor, disp: torch.Tensor, conf: torch.Tensor,
                           residual_weight: float = 1.0, smooth_pts: Optional[float] = None,
                           max_area_ratio: float = 1.6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (new, aff, resid_s): affine + smoothed residual, with the same
    area sanity rule as the numpy path (fall back to the affine prediction)."""
    dst = poly + disp
    A = weighted_affine_torch(poly, dst, conf)
    aff = poly @ A[:, :2].T + A[:, 2]
    resid = dst - aff
    s_pts = smooth_pts if smooth_pts is not None else max(2.0, poly.shape[0] / 16.0)
    resid_s = circular_smooth_torch(resid, conf, s_pts)
    new = aff + residual_weight * resid_s
    a0, a1 = polygon_area_torch(poly).detach(), polygon_area_torch(new).detach()
    if a0 > 0 and (a1 / a0 > max_area_ratio or a1 / a0 < 1.0 / max_area_ratio):
        new = aff
    return new, aff, resid_s


def polygon_area_torch(poly: torch.Tensor) -> torch.Tensor:
    r, c = poly[:, 0], poly[:, 1]
    return 0.5 * (r * torch.roll(c, -1) - c * torch.roll(r, -1)).sum().abs()


def polygon_geometry(poly: torch.Tensor) -> torch.Tensor:
    """Per-point geometric descriptor (N, 7): unit tangent (2), outward normal
    (2), curvature proxy (1), normalised arc position (1), local spacing /
    mean spacing (1). Orientation-independent of the polygon's winding."""
    n = poly.shape[0]
    prev, nxt = torch.roll(poly, 1, 0), torch.roll(poly, -1, 0)
    tang = nxt - prev
    tang = tang / tang.norm(dim=1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack([tang[:, 1], -tang[:, 0]], 1)                 # rotate -90 deg
    # make normals point outward: away from the centroid on average
    c = poly.mean(0, keepdim=True)
    if ((poly - c) * normal).sum() < 0:
        normal = -normal
    curv = (nxt - 2 * poly + prev).norm(dim=1) / (nxt - prev).norm(dim=1).clamp_min(1e-6)
    seg = (nxt - poly).norm(dim=1)
    arc = torch.cumsum(seg, 0) / seg.sum().clamp_min(1e-6)
    spacing = seg / seg.mean().clamp_min(1e-6)
    return torch.cat([tang, normal, curv[:, None], arc[:, None], spacing[:, None]], 1)


# ----------------------------------------------------------------------------
# the readout network
# ----------------------------------------------------------------------------

class ContourReadout(nn.Module):
    """
    Inputs per point: w (D^2), geometry (7), hand-crafted disp (2), conf (1),
    residual (2) [all in the polygon's local frame = image pixels].
    A point MLP, then a circular 1-D conv stack along the contour (the shape
    constraint: every point sees its neighbours' evidence), then heads.
    """

    def __init__(self, D2: int = 169, hidden: int = 96, n_conv: int = 4, kernel: int = 5, n_scales: int = 1):
        super().__init__()
        self.D2, self.n_scales = D2, n_scales
        # n_scales > 1: the point also sees the forward distribution of the SAME
        # window on 2x, 4x, ... downsampled slices (context: +-R*2^s native px).
        self.point_mlp = nn.Sequential(nn.Linear(D2 * n_scales + 7 + 5, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        convs = []
        for i in range(n_conv):
            d = [1, 2, 4, 1][i % 4]
            convs.append(nn.Conv1d(hidden, hidden, kernel, padding=0, dilation=d))
        self.convs = nn.ModuleList(convs)
        self.dil = [[1, 2, 4, 1][i % 4] for i in range(n_conv)]
        self.kernel = kernel
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(n_conv)])
        self.head_delta = nn.Linear(hidden, 2)
        self.head_logvar = nn.Linear(hidden, 1)
        self.head_vanish = nn.Linear(hidden, 1)
        # fail-safe init: zero correction, unit variance, 'not vanished'
        nn.init.zeros_(self.head_delta.weight); nn.init.zeros_(self.head_delta.bias)
        nn.init.zeros_(self.head_logvar.weight); nn.init.zeros_(self.head_logvar.bias)
        nn.init.zeros_(self.head_vanish.weight); nn.init.constant_(self.head_vanish.bias, -3.0)
        self.max_delta = 6.0                                                 # px, tanh-bounded

    def _circ_conv(self, x: torch.Tensor, conv: nn.Conv1d, dil: int) -> torch.Tensor:
        pad = dil * (self.kernel - 1) // 2
        x = torch.cat([x[..., -pad:], x, x[..., :pad]], -1)                 # circular padding
        return conv(x)

    def forward(self, w: torch.Tensor, geom: torch.Tensor, disp: torch.Tensor,
                conf: torch.Tensor, resid: torch.Tensor):
        x = torch.cat([w, geom, disp / 6.0, conf[:, None], resid / 6.0], 1)  # (N, F)
        h = self.point_mlp(x).T[None]                                        # (1, C, N)
        for conv, norm, d in zip(self.convs, self.norms, self.dil):
            h = h + F.gelu(norm(self._circ_conv(h, conv, d)))
        hp = h[0].T                                                          # (N, C)
        delta = self.max_delta * torch.tanh(self.head_delta(hp))             # (N, 2)
        logvar = self.head_logvar(hp)[:, 0].clamp(-6.0, 6.0)                 # (N,)
        vanish = self.head_vanish(hp.mean(0, keepdim=True))[0, 0]            # ()
        return delta, logvar, vanish


# ----------------------------------------------------------------------------
# one differentiable step (hand-crafted + optional learned correction)
# ----------------------------------------------------------------------------

def multiscale_w(w: torch.Tensor, poly: torch.Tensor, corr_extra: Optional[List[torch.Tensor]],
                 n_scales: int) -> torch.Tensor:
    """Concatenate the fine forward distribution w (N, D^2) with the window read
    from each coarser affinity (scale s = 2^(k+1) downsampled slices; the point
    is divided by the scale). corr_extra[k] is (D^2, H/2^(k+1), W/2^(k+1))."""
    if n_scales <= 1:
        return w
    assert corr_extra is not None and len(corr_extra) >= n_scales - 1, \
        f'readout with n_scales={n_scales} needs {n_scales - 1} coarser affinities'
    ws = [w]
    for k in range(n_scales - 1):
        f = 2.0 ** (k + 1)
        ws.append(forward_distribution_torch(corr_extra[k], poly / f)[0])
    return torch.cat(ws, 1)


def contour_step_torch(corr: torch.Tensor, poly: torch.Tensor, readout: Optional[ContourReadout] = None,
                       disp_mode: str = 'soft', residual_weight: float = 1.0,
                       smooth_pts: Optional[float] = None, max_area_ratio: float = 1.6,
                       corr_extra: Optional[List[torch.Tensor]] = None):
    """poly (N,2) float tensor on corr.device. Returns dict with 'new' (N,2),
    'logvar' (N,), 'vanish' (), 'conf' (N,), 'base' (N,2) hand-crafted.
    corr_extra: coarser affinities (see multiscale_w), only read when the
    readout was built with n_scales > 1."""
    w, offs = forward_distribution_torch(corr, poly)
    disp, conf = soft_argmax_conf(w, offs, disp_mode)
    base, aff, resid_s = handcrafted_step_torch(poly, disp, conf, residual_weight, smooth_pts, max_area_ratio)
    out = {'base': base, 'conf': conf}
    if readout is None:
        out.update({'new': base, 'logvar': torch.zeros_like(conf), 'vanish': torch.tensor(-3.0, device=poly.device)})
        return out
    geom = polygon_geometry(poly.detach())
    w_in = multiscale_w(w, poly.detach(), corr_extra, getattr(readout, 'n_scales', 1))
    delta, logvar, vanish = readout(w_in, geom, disp.detach(), conf.detach(), (disp - (aff - poly)).detach())
    out.update({'new': base + delta, 'logvar': logvar, 'vanish': vanish, 'delta': delta})
    return out


def load_readout(path: str, device: str = 'cpu') -> ContourReadout:
    ck = torch.load(path, map_location=device)
    cfg = ck.get('config', {})
    m = ContourReadout(D2=cfg.get('D2', 169), hidden=cfg.get('hidden', 96),
                       n_conv=cfg.get('n_conv', 4), kernel=cfg.get('kernel', 5),
                       n_scales=cfg.get('n_scales', 1))
    m.load_state_dict(ck['state_dict']); m.to(device).eval()
    return m
