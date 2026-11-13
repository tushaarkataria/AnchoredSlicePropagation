"""
Installation check without data or GPU: load both released checkpoints and run
every paper method end to end on a small synthetic CT-like volume.

    python -m pytest tests/test_pipeline_synthetic.py -q
"""
import hashlib
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import evaluate as EV  # noqa: E402

SHA256 = {
    'sli2vol_backbone.pth': '6a070d8c4e306f09fb34a6cae3800d0896b9ef0f78e684c67771257a48f18d34',
    'readout_learned.pth': 'fd455def381f6d7a668abe335cb1be89895932ec24b12bfd8af143dd52a976f5',
}


def _args(method):
    old = sys.argv
    sys.argv = ['evaluate.py', '--method', method, '--data', 'x=/nonexistent', '--out_root', '/tmp/unused', '--device', 'cpu']
    try:
        return EV.parse_args()
    finally:
        sys.argv = old


def _phantom(Z=24, H=56, W=56, seed=0):
    """An ellipsoid 'organ' brighter than a textured background, intensities in [0, 1] like the loader's."""
    rng = np.random.RandomState(seed)
    z, y, x = np.mgrid[:Z, :H, :W].astype(np.float32)
    gt = (((z - Z / 2) / (Z / 2.6)) ** 2 + ((y - H / 2) / (H / 4.0)) ** 2 + ((x - W / 2) / (W / 3.2)) ** 2) <= 1.0
    img = 0.35 + 0.05 * rng.randn(Z, H, W).astype(np.float32)
    img[gt] += 0.30
    return np.clip(img, 0, 1).astype(np.float32), gt.astype(np.int32)


def test_checkpoints_are_the_released_ones():
    for name, ref in SHA256.items():
        h = hashlib.sha256(open(os.path.join(REPO, 'checkpoints', name), 'rb').read()).hexdigest()
        assert h == ref, f'{name}: checksum {h} differs from the released file'


@pytest.fixture(scope='module')
def model():
    a = _args('orthogonal')
    m = EV.build_eval_model(a, 'cpu')
    m.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    return m.eval()


@pytest.mark.parametrize('method', sorted(EV.METHODS))
def test_every_method_runs_and_segments_the_phantom(model, method):
    a = _args(method)
    a._readout = None
    if a.readout_ckpt:
        from contours.readout import load_readout
        a._readout = load_readout(a.readout_ckpt, 'cpu')
    img, gt = _phantom()
    with torch.no_grad():
        fused, per_pred, info = EV.run_case(model, img, gt, a, 'cpu')
    assert fused.shape == gt.shape
    expected_chains = {'axial'} if method == 'sli2vol_1seed' else {'axial', 'coronal', 'sagittal'}
    assert set(per_pred) == expected_chains
    d = EV.vol_dice(fused, gt)
    assert d > 0.8, f'{method}: Dice {d:.3f} on an easy phantom'


def test_learned_readout_starts_from_the_rule_based_tracker():
    """A freshly built readout has a zero displacement head (the learned tracker starts as the
    rule-based one); the released readout has a trained, non-zero head and is the checkpoint
    selected at step 2800 with pseudo-chain validation Dice 0.762 (paper, Sec. 2.4)."""
    from contours.readout import load_readout, ContourReadout
    fresh = ContourReadout()
    assert float(fresh.head_delta.weight.abs().sum()) == 0.0 and float(fresh.head_delta.bias.abs().sum()) == 0.0
    ro = load_readout(os.path.join(REPO, 'checkpoints', 'readout_learned.pth'), 'cpu')
    assert float(ro.head_delta.weight.abs().sum()) > 0.0
    ck = torch.load(os.path.join(REPO, 'checkpoints', 'readout_learned.pth'), map_location='cpu')
    assert ck['step'] == 2800 and abs(ck['val']['dice_mean'] - 0.7618) < 1e-3
