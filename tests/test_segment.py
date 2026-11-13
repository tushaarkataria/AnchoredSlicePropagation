"""
segment.py (inference from three annotated slices) against evaluate.py (the evaluated code path).

    python -m pytest tests/test_segment.py -q        # CPU, about 4 minutes
"""
import os
import subprocess
import sys

import numpy as np
import nibabel as nib
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import evaluate as EV          # noqa: E402
import segment as SG           # noqa: E402
from contours import anchors as AN  # noqa: E402


def _phantom(Z=24, H=56, W=60, seed=0):
    """Off-centre, asymmetric 'organ' (an ellipsoid plus a lobe) so that any flip or axis swap
    of the output would be detected; intensities in [0, 1] like the loader's."""
    rng = np.random.RandomState(seed)
    z, y, x = np.mgrid[:Z, :H, :W].astype(np.float32)
    gt = (((z - 11) / 8.0) ** 2 + ((y - 22) / 12.0) ** 2 + ((x - 34) / 16.0) ** 2) <= 1.0
    gt |= (((z - 13) / 5.0) ** 2 + ((y - 34) / 6.0) ** 2 + ((x - 22) / 6.0) ** 2) <= 1.0
    img = 0.35 + 0.05 * rng.randn(Z, H, W).astype(np.float32)
    img[gt] += 0.30
    return np.clip(img, 0, 1).astype(np.float32), gt.astype(np.int32)


@pytest.fixture(scope='module')
def built():
    return {m: SG.build(m, 'cpu') for m in EV.METHODS}


def test_detect_planes_recovers_the_paper_seeds():
    _, gt = _phantom()
    ann = SG.planes_from_label(gt)
    ref = AN.seeds_from_gt(gt, axes=(0, 1, 2))
    got = SG.detect_planes(ann)
    assert set(got) == {0, 1, 2}
    for ax in ref:
        assert got[ax][0] == ref[ax][0]
        assert np.array_equal(got[ax][1], ref[ax][1])


def test_detect_planes_rejects_two_slices_in_one_direction():
    _, gt = _phantom()
    ann = SG.planes_from_label(gt)
    k = AN.seeds_from_gt(gt, axes=(0,))[0][0]
    ann[k + 3] = gt[k + 3]
    with pytest.raises(ValueError):
        SG.detect_planes(ann)


def test_detect_planes_accepts_a_single_direction():
    _, gt = _phantom()
    ann = np.zeros_like(gt, np.uint8)
    k = AN.seeds_from_gt(gt, axes=(0,))[0][0]
    ann[k] = gt[k]
    assert set(SG.detect_planes(ann)) == {0}


@pytest.mark.parametrize('method', sorted(EV.METHODS))
def test_segment_equals_evaluate_exactly(built, method):
    """Three planes of the label as the only input give bit-identical masks to evaluate.py,
    which reads the full label; so segment.py inherits the verified numbers."""
    model, args, dev = built[method]
    img, gt = _phantom()
    with torch.no_grad():
        ref, ref_chains, _ = EV.run_case(model, img, gt, args, dev)
    fused, chains = SG.segment(model, args, dev, img, SG.detect_planes(SG.planes_from_label(gt)), gt.shape)
    assert np.array_equal(fused, ref)
    assert set(chains) == set(ref_chains) and all(np.array_equal(chains[k], ref_chains[k]) for k in chains)


def _write_case(tmp_path, affine):
    """The phantom as a scan + label in a NIfTI whose array is NOT stored in RAS order."""
    img, gt = _phantom()
    hu = img * 700.0 - 300.0                                    # the loader maps [-300, 400] HU back to [0, 1]
    canon_img = nib.Nifti1Image(np.moveaxis(hu, 0, -1).astype(np.float32), np.diag([1.0, 1.0, 2.5, 1.0]))
    canon_lab = nib.Nifti1Image(np.moveaxis(gt, 0, -1).astype(np.uint8), canon_img.affine)
    to_file = nib.orientations.ornt_transform(nib.io_orientation(canon_img.affine), nib.io_orientation(affine))
    paths = {}
    for name, im in (('scan', canon_img), ('label', canon_lab)):
        out = im.as_reoriented(to_file)
        paths[name] = str(tmp_path / f'{name}.nii.gz')
        nib.save(out, paths[name])
    return paths


@pytest.mark.parametrize('codes', ['LPS', 'RAS'])
def test_end_to_end_cli_writes_the_scan_geometry(tmp_path, codes):
    flips = {'LPS': np.diag([-1.0, -1.0, 2.5, 1.0]), 'RAS': np.diag([1.0, 1.0, 2.5, 1.0])}[codes]
    paths = _write_case(tmp_path, flips)
    out = str(tmp_path / 'new_dir' / 'mask.nii.gz')     # output folder does not exist yet
    r = subprocess.run([sys.executable, os.path.join(REPO, 'segment.py'), '--image', paths['scan'],
                        '--simulate_from_label', paths['label'], '--out', out, '--device', 'cpu'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    scan, lab, pred = nib.load(paths['scan']), nib.load(paths['label']), nib.load(out)
    assert ''.join(nib.aff2axcodes(scan.affine)) == codes
    assert pred.shape == scan.shape and np.allclose(pred.affine, scan.affine)
    p = np.asanyarray(pred.dataobj) > 0
    g = np.asanyarray(lab.dataobj) > 0
    dice = 2 * (p & g).sum() / (p.sum() + g.sum())
    assert dice > 0.8, f'Dice {dice:.3f} in the scan geometry'
    # the saved annotation is the three planes of the label, in the scan geometry
    ann = np.asanyarray(nib.load(out.replace('.nii.gz', '') + '_annotation.nii.gz').dataobj) > 0
    assert ann.any() and not (ann & ~g).any()
    # feeding that annotation file back in gives the same mask
    out2 = str(tmp_path / 'mask2.nii.gz')
    r = subprocess.run([sys.executable, os.path.join(REPO, 'segment.py'), '--image', paths['scan'],
                        '--annotation', out.replace('.nii.gz', '') + '_annotation.nii.gz', '--out', out2,
                        '--device', 'cpu'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert np.array_equal(np.asanyarray(nib.load(out2).dataobj), np.asanyarray(pred.dataobj))
