"""
True-mm rescoring of saved predictions: volumetric Dice, normalised surface
Dice at 1 / 2 / 5 mm (NSD) and average symmetric surface distance in mm
(ASSD, stored as 'ahd_mm'). These are the NSD and ASSD columns of Table 1.

    python rescore.py --root SLiver07=/data/SLiver07 --root MSD_Pancreas=/data/Task07_Pancreas \
        --result contour_learned=results/grid/contour_learned --out results/grid/contour_learned/scores_mm.json

--root names must match the --data names used with evaluate.py. Saved NIfTIs
carry identity affines; the native voxel spacing is read from
<root>/imagesTr/<case>.nii.gz after canonical (RAS) reorientation, i.e. in the
same (Z, H, W) order as the saved arrays. Metric code unchanged from the paper.
"""
import os, json, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion

TAUS_MM = [1.0, 2.0, 5.0]
_ROOTS = {}


def surface_mask(mask_bool):
    """A voxel is on the surface iff it is FG and at least one 6-neighbour is BG."""
    eroded = binary_erosion(mask_bool, iterations=1, border_value=0)
    return mask_bool & ~eroded


def case_metrics(pred, gt, spacing_mm):
    """Compute (dice, ahd_mm, nsd_at_each_tau) for one case. Returns dict."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    out = {'dice': np.nan, 'ahd_mm': np.nan}
    for tau in TAUS_MM:
        out[f'nsd_{tau:g}mm'] = np.nan

    inter = (pred & gt).sum()
    den = pred.sum() + gt.sum()
    if den == 0:
        out['dice'] = 1.0
    else:
        out['dice'] = float((2.0 * inter + 1e-8) / (den + 1e-8))

    pred_s = surface_mask(pred)
    gt_s = surface_mask(gt)
    if not pred_s.any() or not gt_s.any():
        return out

    dt_gt = distance_transform_edt(~gt_s, sampling=spacing_mm)
    dt_pr = distance_transform_edt(~pred_s, sampling=spacing_mm)

    d_pg = dt_gt[pred_s]
    d_gp = dt_pr[gt_s]

    out['ahd_mm'] = float(0.5 * (d_pg.mean() + d_gp.mean()))

    n_p = d_pg.size
    n_g = d_gp.size
    for tau in TAUS_MM:
        n_p_in = (d_pg <= tau).sum()
        n_g_in = (d_gp <= tau).sum()
        out[f'nsd_{tau:g}mm'] = float((n_p_in + n_g_in) / (n_p + n_g))
    return out


def spacing_zhw(source, case_id):
    root = _ROOTS.get(source)
    if root:
        p = os.path.join(root, 'imagesTr', f'{case_id}.nii.gz')
        if os.path.exists(p):
            zr = nib.as_closest_canonical(nib.load(p)).header.get_zooms()[:3]
            return (float(zr[2]), float(zr[0]), float(zr[1]))
    print(f'[warn] no source image for {source}/{case_id}; using 1mm isotropic', flush=True)
    return (1.0, 1.0, 1.0)


def score_one(job):
    source, case_id, pred_p, gt_p = job
    # dataobj + uint8: avoids nibabel's float64 materialisation (a 500-slice
    # volume is 260 MB in float64; the two 3D EDTs inside case_metrics then
    # take ~1-3 GB per worker, which is what OOM-killed the 16-worker run).
    pred = (np.asarray(nib.load(pred_p).dataobj) > 0.5).astype(np.uint8)
    gt = (np.asarray(nib.load(gt_p).dataobj) > 0.5).astype(np.uint8)
    Z = min(pred.shape[0], gt.shape[0])
    m = case_metrics(pred[:Z], gt[:Z], spacing_zhw(source, case_id))
    m['case_id'] = case_id
    return source, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True, help='Source=/dataset/root (repeatable)')
    ap.add_argument('--result', action='append', required=True, help='name=/result/dir (repeatable)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    for spec in a.root:
        k, v = spec.split('=', 1); _ROOTS[k] = v

    results = {}
    for spec in a.result:
        name, rdir = spec.split('=', 1)
        jobs = []
        for src in sorted(os.listdir(os.path.join(rdir, 'niftis'))):
            d = os.path.join(rdir, 'niftis', src)
            for f in sorted(os.listdir(d)):
                if f.endswith('_pred.nii.gz'):
                    cid = f[:-len('_pred.nii.gz')]
                    g = os.path.join(d, f'{cid}_gt.nii.gz')
                    if os.path.exists(g):
                        jobs.append((src, cid, os.path.join(d, f), g))
        per_src = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for src, m in ex.map(score_one, jobs):
                per_src.setdefault(src, []).append(m)
        results[name] = per_src
        print(f'[{name}] scored {len(jobs)} cases', flush=True)

    keys = ['dice', 'nsd_2mm', 'ahd_mm']
    print(f'\n{"variant":<44s} {"source":<12s} {"n":>4s}  ' + '  '.join(f'{k:>14s}' for k in keys))
    for name, per_src in results.items():
        allrows = [r for rows in per_src.values() for r in rows]
        for src, rows in list(sorted(per_src.items())) + [('POOLED', allrows)]:
            line = f'{name:<44s} {src:<12s} {len(rows):>4d}  '
            for k in keys:
                v = np.array([r[k] for r in rows], float); v = v[np.isfinite(v)]
                line += f'  {v.mean():>6.3f} ± {v.std():<5.3f}' if v.size else f'  {"nan":>14s}'
            print(line)
    with open(a.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nper-case scores -> {a.out}')


if __name__ == '__main__':
    main()
