"""
Segment a new CT scan from three annotated slices (inference only, no ground truth needed).

Draw the object on ONE slice in each of the three orthogonal directions (axial, coronal,
sagittal) in any viewer (ITK-SNAP, 3D Slicer, ...), and save the three slices together as
ONE label volume on the scan's grid: non-zero on the drawn slices, zero everywhere else.
For the best results annotate the slice where the object is largest along each direction,
as in the paper.

    python segment.py --image scan.nii.gz --annotation three_slices.nii.gz --out mask.nii.gz

    # mask + anchors instead of the learned contour tracker (better for branching structures
    # such as vessels, see the paper's Table 3)
    python segment.py --image scan.nii.gz --annotation three_slices.nii.gz --out mask.nii.gz --method mask_anchors

    # try it on a scan with a full label: use the label's three largest cross-sections as the
    # annotation, exactly as the paper's evaluation does (the annotation is saved too)
    python segment.py --image scan.nii.gz --simulate_from_label label.nii.gz --out mask.nii.gz

The output mask has the scan's shape, orientation and affine, so it overlays on the scan
in any viewer. The computation is the one evaluated in the paper (evaluate.py run_case).
A GPU is strongly recommended: about 1-2 minutes per scan on a GPU, much longer on CPU.
"""
import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as EV                         # noqa: E402
from contours import anchors as AN            # noqa: E402
from core_utils import require_checkpoint     # noqa: E402
from data_loader import BASE_IMG_LAB_TF       # noqa: E402

# processing frame (Z, H, W): axis 0 = canonical (RAS) axis 2, axis 1 = canonical 0, axis 2 = canonical 1
_FRAME_TO_CANON = {0: 2, 1: 0, 2: 1}


# ----------------------------------------------------------------------------
# annotation <-> seeds
# ----------------------------------------------------------------------------
def _explained_by_lines(sl):
    """True if every foreground pixel of a 2D slice lies on one row plus one column, i.e. the
    slice only contains the intersections with annotated planes of the other two directions."""
    rr, cc = np.nonzero(sl)
    if rr.size == 0 or np.unique(cc).size <= 1 or np.unique(rr).size <= 1:
        return True
    for r in np.unique(rr):                  # remove one candidate row; the rest must be one column
        if np.unique(cc[rr != r]).size <= 1:
            return True
    return False


def detect_planes(ann_zhw):
    """{axis: (index, 2D mask)} of the annotated slices of a (Z, H, W) annotation volume.
    Along each axis exactly one slice may carry a drawn cross-section; slices that only hold the
    intersection lines of the other planes are ignored. Raises ValueError on ambiguous input."""
    ann = np.asarray(ann_zhw) > 0
    if not ann.any():
        raise ValueError('the annotation is empty')
    seeds = {}
    for ax in (0, 1, 2):
        perm = AN.permute_to_axis(ann, ax)
        drawn = [k for k in np.nonzero(perm.reshape(perm.shape[0], -1).any(1))[0] if not _explained_by_lines(perm[k])]
        if len(drawn) > 1:
            raise ValueError(f'more than one annotated slice along processing axis {ax}: {drawn[:6]}; '
                             'annotate exactly one slice per direction')
        if drawn:
            seeds[ax] = (int(drawn[0]), perm[drawn[0]].astype(np.int8))
    if not seeds:
        raise ValueError('no annotated slice found (each drawn slice needs more than a single line)')
    return seeds


def planes_from_label(label_zhw):
    """The paper's simulated annotation: the largest cross-section of a full label along each axis,
    returned as a (Z, H, W) volume that is non-zero only on those three slices."""
    seeds = AN.seeds_from_gt(np.asarray(label_zhw) > 0, axes=(0, 1, 2))
    known = AN.build_known_volume(label_zhw.shape, seeds)
    return (known == 1).astype(np.uint8)


# ----------------------------------------------------------------------------
# geometry: scan file <-> processing frame
# ----------------------------------------------------------------------------
def load_scan(image_path, annotation_path):
    """Image and annotation in the processing frame, exactly as evaluate.py sees a test case."""
    im, an = nib.load(image_path), nib.load(annotation_path)
    if im.shape[:3] != an.shape[:3] or not np.allclose(im.affine, an.affine, atol=1e-3):
        raise ValueError(f'annotation grid {an.shape[:3]} / affine does not match the image grid {im.shape[:3]}; '
                         'save the annotation on the scan itself')
    d = BASE_IMG_LAB_TF({'image': image_path, 'label_full': annotation_path})
    img = np.moveaxis(d['image'].squeeze(0).cpu().numpy().astype(np.float32), -1, 0)
    ann = np.moveaxis((d['label_full'].squeeze(0).cpu().numpy() > 0).astype(np.uint8), -1, 0)
    return img, ann


def to_scan_geometry(vol_zhw, image_path, dtype=np.uint8):
    """Processing-frame volume -> NIfTI with the scan's shape, orientation and affine."""
    orig = nib.load(image_path)
    canon = nib.as_closest_canonical(orig)
    out = nib.Nifti1Image(np.moveaxis(np.asarray(vol_zhw), 0, -1).astype(dtype), canon.affine)
    to_orig = nib.orientations.ornt_transform(nib.io_orientation(canon.affine), nib.io_orientation(orig.affine))
    out = out.as_reoriented(to_orig)
    out.set_qform(orig.affine, code=int(orig.header['qform_code']) or 1)
    out.set_sform(orig.affine, code=int(orig.header['sform_code']) or 1)
    return out


def describe_planes(seeds, image_path, shape_zhw):
    """Annotated slices as (file array axis, index), so they can be checked in a viewer."""
    ornt = nib.io_orientation(nib.load(image_path).affine)
    out = []
    for ax, (k, m) in sorted(seeds.items()):
        c = _FRAME_TO_CANON[ax]
        i = int(np.nonzero(ornt[:, 0] == c)[0][0])
        n = shape_zhw[ax]
        j = n - 1 - k if ornt[i, 1] < 0 else k
        out.append(f'array axis {i}, slice {j} ({int(m.sum())} voxels)')
    return out


# ----------------------------------------------------------------------------
def build(method, device, ckpt=None, readout_ckpt=None):
    argv = ['--method', method, '--data', 'unused=.', '--out_root', '.', '--device', device]
    old = sys.argv
    sys.argv = ['segment.py'] + argv
    try:
        args = EV.parse_args()
    finally:
        sys.argv = old
    if ckpt:
        args.ckpt = ckpt
    if readout_ckpt:
        args.readout_ckpt = readout_ckpt
    dev = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    require_checkpoint(args.ckpt)
    require_checkpoint(args.readout_ckpt)
    model = EV.build_eval_model(args, dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    args._readout = None
    if args.readout_ckpt:
        from contours.readout import load_readout
        args._readout = load_readout(args.readout_ckpt, dev)
    return model, args, dev


def segment(model, args, device, img_zhw, seeds, shape_zhw):
    """Fused mask and per-chain masks in the processing frame."""
    with torch.no_grad():
        fused, per_pred, _ = EV.run_case(model, img_zhw, np.zeros(shape_zhw, np.int32), args, device, seeds=seeds)
    return fused, per_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='CT scan (NIfTI, Hounsfield units)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--annotation', help='label volume on the scan grid, non-zero only on the three annotated slices')
    g.add_argument('--simulate_from_label', help='full label: annotate its three largest cross-sections (paper protocol)')
    ap.add_argument('--out', required=True, help='output mask (.nii.gz)')
    ap.add_argument('--method', default='contour_learned', choices=sorted(EV.METHODS),
                    help='contour_learned (default, best on most organs) or mask_anchors (branching structures); '
                         'the others reproduce the paper baselines')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ckpt', default=None, help='backbone checkpoint (default: checkpoints/sli2vol_backbone.pth)')
    ap.add_argument('--readout_ckpt', default=None, help='readout checkpoint for contour_learned (default: the released one)')
    ap.add_argument('--save_chains', action='store_true', help='also write the three single-direction masks before fusion')
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.simulate_from_label:
        img, lab = load_scan(a.image, a.simulate_from_label)
        ann = planes_from_label(lab)
        ann_path = a.out.replace('.nii.gz', '') + '_annotation.nii.gz'
        nib.save(to_scan_geometry(ann, a.image), ann_path)
        print(f'[annotation] three largest cross-sections of {a.simulate_from_label} -> {ann_path}')
    else:
        img, ann = load_scan(a.image, a.annotation)
    seeds = detect_planes(ann)
    if a.method == 'sli2vol_1seed' and 0 not in seeds:
        raise SystemExit('sli2vol_1seed needs an annotated slice along processing axis 0 (the axial direction)')
    for line in describe_planes(seeds, a.image, img.shape):
        print(f'[annotation] {line}')
    if len(seeds) < 3:
        print(f'[annotation] WARNING: {len(seeds)} annotated direction(s) found; the method expects three')

    model, args, dev = build(a.method, a.device, a.ckpt, a.readout_ckpt)
    print(f'[model] {a.method} on {dev}')
    fused, per_pred = segment(model, args, dev, img, seeds, img.shape)

    nib.save(to_scan_geometry(fused, a.image), a.out)
    if a.save_chains:
        for name, m in per_pred.items():
            nib.save(to_scan_geometry(m, a.image), a.out.replace('.nii.gz', '') + f'_chain_{name}.nii.gz')
    vox = np.prod(nib.load(a.image).header.get_zooms()[:3])
    print(f'[done] {a.out}: {int(fused.sum())} voxels, {fused.sum() * vox / 1000:.1f} ml, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
