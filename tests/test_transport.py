"""Transport tests with a SYNTHETIC affinity (no model, no data, CPU)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from contours import transport as T, raster as R, anchors as A


def shift_corr(H, W, R_, dy, dx, peak=1.0):
    """Affinity where every target pixel (y,x) copies source (y+dy, x+dx),
    i.e. the mask moves by (-dy, -dx). Optional leakage to make it soft."""
    D = 2 * R_ + 1
    corr = torch.full((D * D, H, W), (1.0 - peak) / (D * D - 1) if peak < 1 else 0.0)
    d = (dy + R_) * D + (dx + R_)
    corr[d] = peak
    return corr


def disk(shape=(64, 64), c=(32, 32), r=12):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return ((y - c[0]) ** 2 + (x - c[1]) ** 2) <= r * r


def test_dense_transport_shifts_mask():
    m = disk().astype(np.float32)
    corr = shift_corr(64, 64, 6, dy=2, dx=-3)           # mask moves by (-2, +3)
    out = T.transport_dense(corr, torch.from_numpy(m)).numpy()
    expect = np.roll(np.roll(m, -2, 0), 3, 1)
    assert np.abs(out - expect)[4:-4, 4:-4].max() < 1e-5


def test_point_forward_matches_shift_and_confidence():
    corr = shift_corr(64, 64, 6, dy=2, dx=-3, peak=1.0)
    pts = np.array([[30.0, 30.0], [20.0, 40.0]])
    disp, conf = T.point_forward_distribution(corr, pts)
    assert np.allclose(disp, [[-2, 3], [-2, 3]], atol=1e-4)
    assert (conf > 0.99).all()
    soft = shift_corr(64, 64, 6, dy=2, dx=-3, peak=0.3)
    _, conf2 = T.point_forward_distribution(soft, pts)
    assert (conf2 < conf).all() and (conf2 > 0).all()


def test_contour_step_tracks_shift():
    corr = shift_corr(64, 64, 6, dy=1, dx=-2)
    poly = R.mask_to_polygons(disk(), 48)[0]
    new, mconf, _ = T.contour_step(corr, [poly], (64, 64), 48, use_anchors=False)
    assert np.allclose(new[0] - poly, [-1, 2], atol=0.15)
    assert mconf > 0.99


def test_sdf_roundtrip_and_soft():
    m = disk()
    s = T.sdf_from_mask(m, 10.0)
    assert (s[m] < 0).all() and (s[~m] > 0).all()
    assert abs(s[32, 32] + 10.0) < 1e-6                # truncated at centre
    soft = T.soft_from_sdf(s, 1.0)
    assert soft[32, 32] > 0.99 and soft[0, 0] < 0.01


def test_clamp_with_anchors():
    lm = np.full((8, 8), A.UNKNOWN, np.int8); lm[4, :4] = A.FG; lm[4, 4:] = A.BG
    prob = np.full((8, 8), 0.5, np.float32)
    out = T.clamp_field_with_anchors(prob, lm, 1.0, 0.0)
    assert (out[4, :4] == 1.0).all() and (out[4, 4:] == 0.0).all() and out[0, 0] == 0.5
    sdf = np.zeros((8, 8), np.float32)
    out = T.clamp_field_with_anchors(sdf, lm, -0.5, 0.5)
    assert (out[4, :4] == -0.5).all() and (out[4, 4:] == 0.5).all()


def test_snap_moves_crossing_onto_anchor():
    poly = R.mask_to_polygons(disk(c=(32, 32), r=12), 64)[0]
    # anchor says: on row 32 the boundary is at col 16.5 and 47.5 (true: 20/44) -> radius 16 along that row
    lm = np.full((64, 64), A.UNKNOWN, np.int8)
    lm[32, :] = 0; lm[32, 17:48] = 1
    sa = A.extract_slice_anchors(lm)
    assert sa.n_points == 2
    new, used = T.snap_polygon_to_anchors(poly, sa, max_snap=12.0)
    assert used == 2
    cr = T._line_crossings(new, 0, 32.0)
    cols = sorted(c for _, c in cr)
    assert abs(cols[0] - 16.5) < 0.6 and abs(cols[-1] - 47.5) < 0.6
    # far from the anchor row the polygon barely moved
    far = np.abs(new - poly)[np.abs(poly[:, 0] - 32) > 9].max()
    assert far < 1.0


def test_propagator_end_to_end_with_fake_model():
    """A fake model whose affinity is a constant shift; check all three
    transports move the seed consistently and anchors fix a wrong shift."""
    class Fake:
        def __init__(self, dy, dx): self.dy, self.dx = dy, dx
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, self.dy, self.dx)[None]}
    K = 7
    gt = np.zeros((K, 64, 64), np.int32)
    for k in range(K):
        gt[k] = disk(c=(32 - k, 32 + 2 * k), r=10)       # true motion per slice: (-1, +2)
    img = np.random.RandomState(0).rand(K, 64, 64).astype(np.float32)
    model = Fake(dy=1, dx=-2)                             # affinity encodes exactly the true motion
    for tr in ('mask', 'sdf', 'contour'):
        pred, soft, rel = T.propagate_mask_affinity(model, img, 0, gt[0], transport=tr,
                                                    known_perm=None, device='cpu')
        d = [2 * (pred[k] & gt[k]).sum() / (pred[k].sum() + gt[k].sum()) for k in range(K)]
        assert min(d) > 0.9, (tr, d)
    # wrong affinity (no motion) + anchors along the true boundary: anchors must help
    wrong = Fake(dy=0, dx=0)
    known = np.full_like(gt, A.UNKNOWN, dtype=np.int8)
    known[:, 32, :] = gt[:, 32, :]                       # a coronal-like known line per slice (row 32)
    known[:, :, 40] = gt[:, :, 40]                       # a sagittal-like known line (col 40)
    res = {}
    for ua in (False, True):
        pred, _, rel = T.propagate_mask_affinity(wrong, img, 0, gt[0], transport='contour',
                                                 known_perm=known, use_anchors=ua, device='cpu')
        res[ua] = np.mean([2 * (pred[k] & gt[k]).sum() / (pred[k].sum() + gt[k].sum()) for k in range(1, K)])
    assert res[True] > res[False] + 0.05, res
    assert np.isfinite(rel[1:]).all() and (rel[1:] <= 1).all()


def test_interface_with_real_sli2vol_class():
    """Random-weight Sli2VolRegistration on random slices: shapes + ranges of
    corr, and the propagator runs end to end for all three transports."""
    from models.sli2vol_original import Sli2VolRegistration
    torch.manual_seed(0)
    model = Sli2VolRegistration(R=6, is_training=False, use_edge_profile=True,
                                edge_radius=3, edge_interval=1, temperature=0.7).eval()
    img = torch.rand(5, 48, 48)
    corr = T.get_corr(model, img[0], img[1])
    assert corr.shape == (13 * 13, 48, 48)
    assert torch.allclose(corr.sum(0), torch.ones(48, 48), atol=1e-4)
    seed = disk((48, 48), (24, 24), 9).astype(np.int32)
    for tr in ('mask', 'sdf', 'contour', 'ensemble'):
        pred, soft, rel = T.propagate_mask_affinity(model, img.numpy(), 2, seed, transport=tr,
                                                    known_perm=None, device='cpu')
        assert pred.shape == (5, 48, 48) and soft.shape == (5, 48, 48) and rel.shape == (5,)
        assert pred.dtype == np.int32 and 0.0 <= soft.min() and soft.max() <= 1.0
        assert (pred[2] == seed).all()


def test_ensemble_tracks_shift_and_reports_agreement():
    """Ensemble consensus follows a constant shift like the single operators,
    fills self_agreement with high values (operators agree on a clean
    affinity), and runs with consensus_reinit off as well."""
    class Fake:
        def __init__(self, dy, dx): self.dy, self.dx = dy, dx
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, self.dy, self.dx)[None]}
    K = 7
    gt = np.zeros((K, 64, 64), np.int32)
    for k in range(K):
        gt[k] = disk(c=(32 - k, 32 + 2 * k), r=10)
    img = np.random.RandomState(0).rand(K, 64, 64).astype(np.float32)
    model = Fake(dy=1, dx=-2)
    for reinit in (True, False):
        agree = np.full((K,), np.nan, np.float32)
        pred, soft, rel = T.propagate_mask_affinity(model, img, 0, gt[0], transport='ensemble',
                                                    known_perm=None, device='cpu',
                                                    consensus_reinit=reinit, self_agreement=agree)
        d = [2 * (pred[k] & gt[k]).sum() / (pred[k].sum() + gt[k].sum()) for k in range(K)]
        assert min(d) > 0.9, (reinit, d)
        assert np.isnan(agree[0]) and np.isfinite(agree[1:]).all() and (agree[1:] > 0.9).all(), agree
        assert 0.0 <= soft.min() and soft.max() <= 1.0 and pred.dtype == np.int32
    # a leaky affinity lowers the agreement (the label-free reliability reacts)
    class Leaky(Fake):
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, self.dy, self.dx, peak=0.2)[None]}
    agree2 = np.full((K,), np.nan, np.float32)
    T.propagate_mask_affinity(Leaky(1, -2), img, 0, gt[0], transport='ensemble', known_perm=None,
                              device='cpu', self_agreement=agree2)
    assert np.nanmean(agree2) < np.nanmean(agree), (np.nanmean(agree2), np.nanmean(agree))


def test_ensemble_anchors_help_and_weights():
    class Fake:
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, 0, 0)[None]}   # no motion
    K = 7
    gt = np.zeros((K, 64, 64), np.int32)
    for k in range(K):
        gt[k] = disk(c=(32 - k, 32 + 2 * k), r=10)
    img = np.random.RandomState(1).rand(K, 64, 64).astype(np.float32)
    known = np.full_like(gt, A.UNKNOWN, dtype=np.int8)
    known[:, 32, :] = gt[:, 32, :]; known[:, :, 40] = gt[:, :, 40]
    res = {}
    for ua in (False, True):
        pred, _, rel = T.propagate_mask_affinity(Fake(), img, 0, gt[0], transport='ensemble',
                                                 known_perm=known, use_anchors=ua, device='cpu',
                                                 ens_weights=(1.0, 2.0, 1.0))
        res[ua] = np.mean([2 * (pred[k] & gt[k]).sum() / (pred[k].sum() + gt[k].sum()) for k in range(1, K)])
    assert res[True] > res[False] + 0.02, res
    assert np.isfinite(rel[1:]).all()


def test_readout_torch_step_matches_numpy_and_is_failsafe():
    """Torch hand-crafted step == numpy contour_step; an untrained readout
    changes nothing; the propagator accepts a readout."""
    from contours import readout as RO
    H = W = 80
    corr = shift_corr(H, W, 6, dy=2, dx=-1, peak=0.4)
    m = disk((H, W), (40, 38), 14).astype(np.int32)
    poly = R.mask_to_polygons(m, 48)[0]
    new_np, _, _ = T.contour_step(corr, [poly], (H, W), 48, use_anchors=False)
    out = RO.contour_step_torch(corr, torch.from_numpy(poly))
    assert float((out['new'] - torch.from_numpy(new_np[0])).abs().max()) < 1e-3
    ro = RO.ContourReadout()
    out2 = RO.contour_step_torch(corr, torch.from_numpy(poly), ro)
    assert float((out2['new'] - out['new']).abs().max()) < 1e-4 and float(out2['vanish']) < 0
    # propagator with the (untrained) readout == propagator without it
    class Fake:
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, 1, -2)[None]}
    K = 5
    gt = np.zeros((K, 64, 64), np.int32)
    for k in range(K): gt[k] = disk(c=(32 - k, 32 + 2 * k), r=10)
    img = np.random.RandomState(0).rand(K, 64, 64).astype(np.float32)
    a, _, _ = T.propagate_mask_affinity(Fake(), img, 0, gt[0], transport='contour', known_perm=None, device='cpu')
    b, _, _ = T.propagate_mask_affinity(Fake(), img, 0, gt[0], transport='contour', known_perm=None, device='cpu', readout=ro)
    assert (a == b).mean() > 0.995
    d = [2 * (b[k] & gt[k]).sum() / (b[k].sum() + gt[k].sum()) for k in range(K)]
    assert min(d) > 0.9, d


def test_readout_unrolled_loss_learns_on_synthetic_shift():
    """A few optimiser steps on a constant-shift affinity with a deliberately
    WRONG hand-crafted base (residual_weight=0, shift under-estimated by a
    soft peak) must reduce the unrolled loss: the readout can learn."""
    from contours import readout as RO
    import train_readout as TR
    H = W = 64
    gt = np.zeros((4, H, W), np.int32)
    for k in range(4): gt[k] = disk((H, W), (32 - 2 * k, 30 + 3 * k), 11)
    corr = shift_corr(H, W, 6, dy=2, dx=-3, peak=0.05)        # very diffuse: soft-argmax under-shoots
    class Fake:
        def __call__(self, s, t, return_corr=True): return {'corr_weights': corr[None]}
    sample = {'img': np.random.RandomState(0).rand(4, H, W).astype(np.float32), 'targets': gt.astype(bool),
              'poly0': R.mask_to_polygons(gt[0], 40)[0].astype(np.float32), 'axis': 0, 'case': 'x'}
    ro = RO.ContourReadout(hidden=32, n_conv=2)
    opt = torch.optim.Adam(ro.parameters(), lr=3e-3)
    l0, d0, _ = TR.unroll(Fake(), ro, sample, 3, 'cpu')
    for _ in range(25):
        opt.zero_grad(); l, d, _ = TR.unroll(Fake(), ro, sample, 3, 'cpu'); l.backward(); opt.step()
    l1, d1, _ = TR.unroll(Fake(), ro, sample, 3, 'cpu')
    assert float(l1) < float(l0) and np.mean(d1) >= np.mean(d0) - 1e-3, (float(l0), float(l1), d0, d1)


def test_ensemble_uses_readout_and_zero_init_matches_handcrafted():
    """With a zero-initialised readout the ensemble must reproduce the
    hand-crafted ensemble (the readout path is wired in, and is identity at init)."""
    import torch
    from contours.readout import ContourReadout
    class Fake:
        def __call__(self, s, t, return_corr=True):
            return {'corr_weights': shift_corr(s.shape[-2], s.shape[-1], 6, 1, -2)[None]}
    K = 6
    gt = np.zeros((K, 64, 64), np.int32)
    for k in range(K):
        gt[k] = disk(c=(32 - k, 32 + 2 * k), r=10)
    img = np.random.RandomState(0).rand(K, 64, 64).astype(np.float32)
    ro = ContourReadout().eval()
    with torch.no_grad():
        for p in ro.parameters(): p.zero_()
        ro.head_vanish.bias.fill_(-3.0)
    p0, _, _ = T.propagate_mask_affinity(Fake(), img, 0, gt[0], transport='ensemble', known_perm=None, device='cpu')
    p1, _, _ = T.propagate_mask_affinity(Fake(), img, 0, gt[0], transport='ensemble', known_perm=None, device='cpu', readout=ro)
    d = [2 * (p0[k] & p1[k]).sum() / max(p0[k].sum() + p1[k].sum(), 1) for k in range(K)]
    assert min(d) > 0.97, d
    d_gt = [2 * (p1[k] & gt[k]).sum() / (p1[k].sum() + gt[k].sum()) for k in range(K)]
    assert min(d_gt) > 0.9, d_gt
