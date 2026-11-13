"""Object-mix knobs of the readout training sampler (no GPU, no data)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_readout as TR


def _ellipsoid(shape=(20, 32, 32), radii=(6, 10, 8)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [s // 2 for s in shape]
    return ((z - c[0]) / radii[0]) ** 2 + ((y - c[1]) / radii[1]) ** 2 + ((x - c[2]) / radii[2]) ** 2 <= 1.0


def test_object_stats_volume_and_fill():
    obj = np.zeros((10, 16, 16), bool); obj[2:6, 4:8, 4:12] = True          # 4 x 4 x 8 box
    ml, fill = TR.PseudoChainSampler.object_stats(obj, factor=2, spacing_zhw=(2.5, 0.8, 0.8))
    # 128 low-res voxels, each 2.5 mm x (2*0.8) mm x (2*0.8) mm
    assert abs(ml - 128 * 2.5 * 1.6 * 1.6 / 1000.0) < 1e-9
    assert fill == 1.0
    ml_e, fill_e = TR.PseudoChainSampler.object_stats(_ellipsoid(), factor=1, spacing_zhw=None)
    assert 0.35 < fill_e < 0.6                                              # ellipsoid fills ~pi/6 of its box (less once discretised)


def test_accept_object_filters():
    rng = np.random.RandomState(0)
    box = np.zeros((10, 16, 16), bool); box[2:6, 4:8, 4:12] = True
    s = TR.PseudoChainSampler([], K=2, rng=rng)                              # defaults: everything accepted
    assert s.accept_object(box, 1, None)
    s = TR.PseudoChainSampler([], K=2, rng=rng, obj_ml=(1.0, 0.0))           # >= 1 ml; box is 0.128 ml
    assert not s.accept_object(box, 1, None)
    assert s.accept_object(box, 4, None)                                     # x16 in-plane -> 2.05 ml
    s = TR.PseudoChainSampler([], K=2, rng=rng, obj_ml=(0.0, 0.1))           # <= 0.1 ml
    assert not s.accept_object(box, 1, None)
    s = TR.PseudoChainSampler([], K=2, rng=rng, obj_fill_min=0.9)
    assert s.accept_object(box, 1, None) and not s.accept_object(_ellipsoid(), 1, None)


def test_sample_counts_rejections_and_returns_none_when_everything_is_filtered(monkeypatch, tmp_path):
    class FakeSV:
        factor = 1
        def sample(self, rng, scale=None, n_merge_max=5, min_voxels=200, **kw):
            obj = np.zeros((10, 16, 16), bool); obj[2:6, 4:8, 4:12] = True; return obj
    case = {'img_npy': str(tmp_path / 'x.npy'), 'sv_npz': 'k', 'case_id': 'c', 'spacing_zhw': [1, 1, 1]}
    s = TR.PseudoChainSampler([case], K=2, rng=np.random.RandomState(0), obj_ml=(5.0, 0.0))
    s.cache['k'] = FakeSV()
    assert s.sample(tries=7) is None and s.n_rejected == 7
