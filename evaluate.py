"""
Anchored slice propagation: evaluation of every method in the paper.

One run = one method on one or more test sets. Paper rows (``--method``):

    sli2vol_1seed    Sli2Vol, 1 seed            mask transport, axial seed only, no anchors
    orthogonal       Orthogonal seeds, mask     3 seeds, distance-weighted soft fusion
    mask_anchors     Mask + anchors             + anchors, reliability-weighted fusion
    sdf_anchors      SDF + anchors              signed-distance transport
    contour_rule     Contour tracking, rule-based   (training-free)
    contour_learned  Contour tracking, learned  (+ checkpoints/readout_learned.pth)

Example (two SLiver07 cases):
    python evaluate.py --method contour_learned --data SLiver07=/data/SLiver07 \
        --limit_cases 2 --out_root results/grid/contour_learned

Outputs in --out_root: per_case.json (volumetric Dice per case and per chain),
aggregate_stats.json, run_config.json and niftis/<source>/<case>_{gt,pred}.nii.gz
(arrays in the loader's (Z,H,W) order with identity affine; rescore.py reads the
native spacing from the dataset root for NSD / ASSD).

The per-case computation is unchanged from the code that produced the paper.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_utils import (
    dice_multiclass_slice, symmetric_hausdorff, symmetric_mean_surface_distance, build_eval_model,
    fuse_soft, _permute_to_axis, _permute_from_axis, _select_seed, _AXIS_NAMES, save_nifti, save_overlap_nifti,
    require_checkpoint,
)
from data_loader import iterate_cases
from contours import anchors as AN
from contours.transport import propagate_mask_affinity


# ----------------------------------------------------------------------------
# reliability-weighted soft_dist fusion
# ----------------------------------------------------------------------------

def fuse_soft_dist_rel(soft_per_axis, rel_per_axis, seeds_canonical, shape,
                       sigma=30.0, rel_floor=0.05, rel_smooth=2):
    """soft_dist weights multiplied by each chain's per-slice anchor agreement
    r_c(k) (nan -> 1). rel_per_axis[axis] is a 1-D array indexed by the
    chain's slice coordinate (Z for axial, H for coronal, W for sagittal)."""
    Z, H, W = shape
    grids = {'axial': np.arange(Z)[:, None, None],
             'coronal': np.arange(H)[None, :, None],
             'sagittal': np.arange(W)[None, None, :]}
    num = np.zeros(shape, np.float32); den = np.zeros(shape, np.float32)
    for ax, soft in soft_per_axis.items():
        seed = seeds_canonical[ax]
        w = np.exp(-np.abs(grids[ax] - float(seed)) / sigma).astype(np.float32)
        rel = np.asarray(rel_per_axis[ax], np.float32).copy()
        rel[~np.isfinite(rel)] = 1.0
        if rel_smooth > 0:                                   # light smoothing along the chain
            k = np.ones(2 * rel_smooth + 1) / (2 * rel_smooth + 1)
            rel = np.convolve(np.pad(rel, rel_smooth, mode='edge'), k, mode='valid')
        rel = np.clip(rel, rel_floor, 1.0)
        shp = [1, 1, 1]; shp[{'axial': 0, 'coronal': 1, 'sagittal': 2}[ax]] = -1
        w = w * rel.reshape(shp)
        num += w * soft; den += w
    fused = num / np.maximum(den, 1e-8)
    return (fused > 0.5).astype(np.int32)


# ----------------------------------------------------------------------------
# per-case propagation
# ----------------------------------------------------------------------------

def _seed_selector(args):
    """Seed-position rule. 'largest' = the paper's argmax-area slice (default).
    A number f in [0,1] = the slice at fraction f of the organ's extent along
    that axis (0.5 = middle). 'random' = uniform over the slices that contain
    the organ, seeded per case by --seed."""
    pos = getattr(args, 'seed_pos', 'largest')
    if pos == 'largest':
        return None
    rng = np.random.RandomState(args.seed)
    def select_fn(gt_perm):
        areas = gt_perm.reshape(gt_perm.shape[0], -1).sum(axis=1)
        nz = np.nonzero(areas)[0]
        if nz.size == 0: return None
        if pos == 'random':
            return int(rng.choice(nz))
        f = float(pos); idx = int(round(nz[0] + f * (nz[-1] - nz[0])))
        idx = int(nz[np.argmin(np.abs(nz - idx))])          # nearest slice that actually contains the organ
        return idx
    return select_fn


def run_case(model, img_zhw, gt_zhw, args, device, step_logs=None, seeds=None):
    """seeds: optional {axis: (slice index, 2D mask)} of the annotated slices (segment.py). By default
    they are taken from the full label gt_zhw with the paper's rule (largest cross-section per axis)."""
    axes = (0,) if args.seeds == 'axial1' else (0, 1, 2)
    if seeds is None:
        seeds = AN.seeds_from_gt(gt_zhw, axes=(0, 1, 2), select_fn=_seed_selector(args))   # all three seeds are known annotations
    known = AN.build_known_volume(gt_zhw.shape, seeds) if args.anchors == 'on' else None
    # For the 1-axial arm the anchors come from the SAME three annotations the
    # orthogonal arm uses -- so 'axial1 + anchors on' is a 3-annotation config
    # with a single chain. It is reported as such.
    per_pred, per_soft, per_rel, per_self, info = {}, {}, {}, {}, []
    for ax in axes:
        if ax not in seeds:
            info.append({'axis': _AXIS_NAMES[ax], 'seed_idx': None, 'skipped': True}); continue
        seed_idx, seed_mask = seeds[ax]
        img_p = _permute_to_axis(img_zhw, ax).astype(np.float32, copy=False)
        known_p = AN.permute_to_axis(known, ax) if known is not None else None
        self_p = np.full((img_p.shape[0],), np.nan, np.float32)
        pred_p, soft_p, rel_p = propagate_mask_affinity(
            model, img_p, seed_idx, seed_mask, transport=args.transport,
            known_perm=known_p, use_anchors=(args.anchors == 'on'),
            n_points=args.n_points, sdf_trunc=args.sdf_trunc, soft_sigma=args.soft_sigma,
            residual_weight=args.residual_weight, reseed_from_anchors=args.reseed_from_anchors,
            use_verification=not args.no_verification, device=device,
            disp_mode=args.disp_mode, area_match=args.area_match,
            step_log=(step_logs.setdefault(_AXIS_NAMES[ax], []) if step_logs is not None else None),
            consensus_reinit=(args.consensus_reinit == 'on'),
            ens_weights=tuple(float(v) for v in args.ens_weights.split(',')),
            self_agreement=self_p, readout=getattr(args, '_readout', None))
        name = _AXIS_NAMES[ax]
        per_self[name] = self_p
        per_pred[name] = _permute_from_axis(pred_p, ax).astype(np.int32)
        per_soft[name] = _permute_from_axis(soft_p, ax).astype(np.float32)
        per_rel[name] = rel_p
        info.append({'axis': name, 'seed_idx': int(seed_idx), 'skipped': False,
                     'mean_rel': float(np.nanmean(rel_p)) if np.isfinite(rel_p).any() else None,
                     'mean_self_agree': float(np.nanmean(self_p)) if np.isfinite(self_p).any() else None})
    if not per_pred:
        return np.zeros_like(gt_zhw, np.int32), per_pred, info
    seeds_canon = {i['axis']: i['seed_idx'] for i in info if not i['skipped']}
    if len(per_pred) == 1:
        fused = next(iter(per_pred.values()))
    elif args.fusion == 'soft_dist':
        fused = fuse_soft(per_soft, mode='soft_dist', sigma=args.fusion_sigma,
                          seeds_canonical=seeds_canon, shape=gt_zhw.shape)
    elif args.fusion == 'soft_dist_rel':
        fused = fuse_soft_dist_rel(per_soft, per_rel, seeds_canon, gt_zhw.shape, sigma=args.fusion_sigma)
    else:  # soft_dist_self: label-free reliability from operator agreement (ensemble only; else all-ones)
        fused = fuse_soft_dist_rel(per_soft, per_self, seeds_canon, gt_zhw.shape, sigma=args.fusion_sigma)
    return fused.astype(np.int32), per_pred, info


def vol_dice(a, b):
    a = a > 0; b = b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.merge:
        merge_shards(args.out_root); return
    if not args.data:
        raise SystemExit('--data NAME=ROOT is required (repeatable)')
    device = args.device if torch.cuda.is_available() else 'cpu'
    source_roots = [tuple(spec.split('=', 1)) for spec in args.data]
    only = [c for c in args.cases.split(',') if c] if args.cases else None
    test_loader = iterate_cases(source_roots, only=only)

    require_checkpoint(args.ckpt)
    require_checkpoint(args.readout_ckpt)
    model = build_eval_model(args, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f'[backbone] loaded {args.ckpt} on {device}', flush=True)
    model.eval()
    args._readout = None
    if args.readout_ckpt:
        from contours.readout import load_readout
        args._readout = load_readout(args.readout_ckpt, device)
        print(f'[readout] loaded {args.readout_ckpt}', flush=True)

    os.makedirs(args.out_root, exist_ok=True)
    nifti_dir = os.path.join(args.out_root, 'niftis'); os.makedirs(nifti_dir, exist_ok=True)
    with open(os.path.join(args.out_root, 'run_config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items() if not k.startswith('_')}, f, indent=2)

    summary, all_dice, by_src = {}, [], {}
    shard_k, shard_n = (int(args.shard.split('/')[0]), int(args.shard.split('/')[1])) if args.shard else (0, 0)
    per_case_name = f'per_case_{shard_k}of{shard_n}.json' if shard_n else 'per_case.json'
    t0 = time.time()
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            if batch is None: continue
            if args.limit_cases and bidx >= args.limit_cases: break
            if shard_n and (bidx % shard_n) != shard_k: continue
            img_vol, lab_full, _, _, z_start, z_end, case_id, source = batch
            case_id = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
            source = source[0] if isinstance(source, (list, tuple)) else source
            z0 = int(z_start.item()) if torch.is_tensor(z_start) else int(z_start)
            z1 = int(z_end.item()) if torch.is_tensor(z_end) else int(z_end)
            img_zhw = img_vol[0].cpu().numpy().astype(np.float32)
            gt_zhw = lab_full[0].cpu().numpy().astype(np.int32)
            Z = min(img_zhw.shape[0], gt_zhw.shape[0]); img_zhw, gt_zhw = img_zhw[:Z], gt_zhw[:Z]
            z0, z1 = min(z0, Z - 1), min(z1, Z - 1)

            tc = time.time()
            step_logs = {} if (args.log_steps and args.transport in ('contour', 'ensemble')) else None
            fused, per_pred, info = run_case(model, img_zhw, gt_zhw, args, device, step_logs=step_logs)
            dt = time.time() - tc
            if step_logs:
                # ground-truth area per slice along each chain, for the shrinkage diagnostic
                gt_area = {'axial': gt_zhw.sum((1, 2)), 'coronal': gt_zhw.sum((0, 2)), 'sagittal': gt_zhw.sum((0, 1))}
                diag_dir = os.path.join(args.out_root, 'diag'); os.makedirs(diag_dir, exist_ok=True)
                with open(os.path.join(diag_dir, f'{source}_{case_id}.json'), 'w') as f:
                    json.dump({ax: {'gt_area': [int(v) for v in gt_area[ax]], 'steps': steps}
                               for ax, steps in step_logs.items()}, f)

            zs = np.arange(z0, z1 + 1)
            dice_vals = np.array([dice_multiclass_slice(fused[z], gt_zhw[z]) for z in zs])
            sd_vals = np.array([symmetric_mean_surface_distance(fused[z], gt_zhw[z]) for z in zs])
            hd_vals = np.array([symmetric_hausdorff(fused[z], gt_zhw[z]) for z in zs])
            vdice = vol_dice(fused, gt_zhw)
            tag = f'{source}_{case_id}'
            summary[tag] = {
                'source': str(source), 'case_id': str(case_id), 'z_start': z0, 'z_end': z1,
                'vol_dice': vdice, 'mean_dice': float(np.nanmean(dice_vals)),
                'mean_sd_px': float(np.nanmean(sd_vals)), 'mean_hd_px': float(np.nanmean(hd_vals)),
                'per_axis_vol_dice': {k: vol_dice(v, gt_zhw) for k, v in per_pred.items()},
                'per_axis_info': info, 'seconds': dt,
                'transport': args.transport, 'anchors': args.anchors, 'seeds': args.seeds, 'fusion': args.fusion,
            }
            all_dice.append(vdice); by_src.setdefault(str(source), []).append(vdice)

            src_out = os.path.join(nifti_dir, str(source)); os.makedirs(src_out, exist_ok=True)
            save_nifti(gt_zhw, os.path.join(src_out, f'{case_id}_gt.nii.gz'))
            save_nifti(fused, os.path.join(src_out, f'{case_id}_pred.nii.gz'))
            save_overlap_nifti(gt_zhw, fused, os.path.join(src_out, f'{case_id}_overlap.nii.gz'))
            if args.save_per_axis:
                for k, v in per_pred.items():
                    save_nifti(v, os.path.join(src_out, f'{case_id}_pred_{k}.nii.gz'))

            ax_str = ', '.join(f"{k}={v:.3f}" for k, v in summary[tag]['per_axis_vol_dice'].items())
            print(f'[{bidx}] {tag} | volDice={vdice:.4f} sliceDice={summary[tag]["mean_dice"]:.4f} '
                  f'| {ax_str} | {dt:.1f}s', flush=True)
            with open(os.path.join(args.out_root, per_case_name), 'w') as f:
                json.dump(summary, f, indent=1)

    agg = {'n': len(all_dice), 'vol_dice_mean': float(np.mean(all_dice)) if all_dice else None,
           'vol_dice_std': float(np.std(all_dice)) if all_dice else None,
           'by_source': {s: {'n': len(v), 'mean': float(np.mean(v)), 'std': float(np.std(v))} for s, v in by_src.items()},
           'minutes': (time.time() - t0) / 60.0, 'config': {k: v for k, v in vars(args).items() if not k.startswith('_')}}
    agg_name = f'aggregate_stats_{shard_k}of{shard_n}.json' if shard_n else 'aggregate_stats.json'
    with open(os.path.join(args.out_root, agg_name), 'w') as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != 'config'}, indent=2))
    if shard_n:
        merge_shards(args.out_root)   # merges whatever shards have finished so far
    print(f'\nDone. Outputs at: {args.out_root}')


def merge_shards(out_root):
    """Combine per_case_*of*.json into per_case.json + aggregate_stats.json.
    Safe to call repeatedly; reports how many shards were found."""
    import glob
    files = sorted(glob.glob(os.path.join(out_root, 'per_case_*of*.json')))
    if not files:
        return
    merged = {}
    for f in files:
        merged.update(json.load(open(f)))
    n_expected = int(os.path.basename(files[0]).split('of')[1].split('.')[0])
    vd = [v['vol_dice'] for v in merged.values()]
    by_src = {}
    for v in merged.values():
        by_src.setdefault(v['source'], []).append(v['vol_dice'])
    agg = {'n': len(vd), 'shards_found': len(files), 'shards_expected': n_expected,
           'complete': len(files) == n_expected,
           'vol_dice_mean': float(np.mean(vd)) if vd else None, 'vol_dice_std': float(np.std(vd)) if vd else None,
           'by_source': {s: {'n': len(x), 'mean': float(np.mean(x)), 'std': float(np.std(x))} for s, x in by_src.items()}}
    with open(os.path.join(out_root, 'per_case.json'), 'w') as f:
        json.dump(merged, f, indent=1)
    with open(os.path.join(out_root, 'aggregate_stats.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'[merge] {len(files)}/{n_expected} shards, {len(vd)} cases, mean volDice '
          f'{agg["vol_dice_mean"]:.4f}' if vd else '[merge] nothing')


HERE = os.path.dirname(os.path.abspath(__file__))
# paper rows -> flags (everything else at its default, which is the paper setting)
METHODS = {
    'sli2vol_1seed':   dict(transport='mask', anchors='none', seeds='axial1', fusion='soft_dist', readout_ckpt=None),
    'orthogonal':      dict(transport='mask', anchors='none', seeds='orthogonal', fusion='soft_dist', readout_ckpt=None),
    'mask_anchors':    dict(transport='mask', anchors='on', seeds='orthogonal', fusion='soft_dist_rel', readout_ckpt=None),
    'sdf_anchors':     dict(transport='sdf', anchors='on', seeds='orthogonal', fusion='soft_dist_rel', readout_ckpt=None),
    'contour_rule':    dict(transport='contour', anchors='on', seeds='orthogonal', fusion='soft_dist_rel', readout_ckpt=None),
    'contour_learned': dict(transport='contour', anchors='on', seeds='orthogonal', fusion='soft_dist_rel',
                            readout_ckpt=os.path.join(HERE, 'checkpoints', 'readout_learned.pth')),
}


def parse_args():
    p = argparse.ArgumentParser('evaluate.py', description='Anchored slice propagation: evaluate one paper method (see module docstring).')
    p.add_argument('--method', choices=sorted(METHODS), default=None,
                   help='paper row; sets --transport/--anchors/--seeds/--fusion/--readout_ckpt (overrides them)')
    p.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints', 'sli2vol_backbone.pth'),
                   help='frozen Sli2Vol backbone')
    p.add_argument('--transport', choices=['mask', 'sdf', 'contour', 'ensemble'], default='contour')
    p.add_argument('--consensus_reinit', choices=['on', 'off'], default='on',
                   help='ensemble: restart all three operators from the consensus each step (on) or let each '
                        'carry its own state and only fuse the output (off)')
    p.add_argument('--ens_weights', default='1,1,1', help='ensemble: w_mask,w_sdf,w_contour')
    p.add_argument('--anchors', choices=['none', 'on'], default='on')
    p.add_argument('--seeds', choices=['axial1', 'orthogonal'], default='orthogonal')
    p.add_argument('--fusion', choices=['soft_dist', 'soft_dist_rel', 'soft_dist_self'], default='soft_dist',
                   help='soft_dist_rel = x anchor agreement (needs anchors); soft_dist_self = x label-free '
                        'self-agreement of the three operators (ensemble transport, no anchors needed)')
    p.add_argument('--fusion_sigma', type=float, default=30.0)
    p.add_argument('--n_points', type=int, default=64)
    p.add_argument('--sdf_trunc', type=float, default=10.0)
    p.add_argument('--seed_pos', default='largest', help="seed slice rule: 'largest' (paper), a fraction 0-1 of the organ extent (0.5 = middle), or 'random'")
    p.add_argument('--soft_sigma', type=float, default=1.0)
    p.add_argument('--residual_weight', type=float, default=1.0,
                   help='1 = affine + smoothed non-rigid residual; 0 = affine only')
    p.add_argument('--reseed_from_anchors', action='store_true')
    p.add_argument('--disp_mode', choices=['soft', 'argmax'], default='soft',
                   help='contour: per-point displacement = soft-argmax (expectation; inward bias on convex '
                        'boundaries) or argmax of the forward distribution')
    p.add_argument('--area_match', choices=['none', 'dense'], default='none',
                   help='contour: rescale each tracked polygon per step so its area matches the densely '
                        'transported area of its own mask (removes the systematic shrinkage)')
    p.add_argument('--readout_ckpt', default=None,
                   help='contour: trained contours.readout checkpoint (best.pth from train_readout.py)')
    p.add_argument('--log_steps', action='store_true',
                   help='contour: write per-step area diagnostics to <out_root>/diag/<Source>_<case>.json')
    p.add_argument('--no_verification', action='store_true',
                   help='Disable the intensity verification module (ON by default, as in the baseline)')
    p.add_argument('--save_per_axis', action='store_true')
    p.add_argument('--limit_cases', type=int, default=0, help='>0: stop after this many cases (smoke test)')
    p.add_argument('--shard', type=str, default=None,
                   help="'k/n': process only cases with index %% n == k (0-based); outputs go to the same "
                        "out_root with per_case_<k>of<n>.json; run merge_shards() / --merge afterwards")
    p.add_argument('--merge', action='store_true', help='Only merge existing per_case_*of*.json in out_root and exit')
    # data
    p.add_argument('--data', action='append', default=None,
                   help='NAME=ROOT, repeatable; ROOT holds imagesTr/ and labelsTr/ (see README, Data)')
    p.add_argument('--cases', default='', help='comma-separated case ids to run (default: all)')
    # model knobs (must match training; defaults = paper axial checkpoint)
    p.add_argument('--R', type=int, default=6)
    p.add_argument('--no_edge_profile', action='store_true')
    p.add_argument('--edge_radius', type=int, default=3)
    p.add_argument('--edge_interval', type=int, default=1)
    p.add_argument('--corr_temperature', type=float, default=0.7)
    p.add_argument('--out_root', required=True)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.method:
        for k, v in METHODS[args.method].items():
            setattr(args, k, v)
    return args


if __name__ == '__main__':
    main()
