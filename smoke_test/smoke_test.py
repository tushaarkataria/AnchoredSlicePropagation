"""
Smoke test: does this code, on your machine, reproduce the paper's numbers?

Runs evaluate.py (and rescore.py) on a few named development-grid cases and
compares every per-case number with the values behind Table 1 / Table 2
(reference_results/grid.json). It also runs segment.py, the inference-only
path, on one case and checks the written mask against the label. Needs a GPU and the three grid folders
prepared with prepare_data.py.

    # ~20 min on a TITAN V: learned contour tracker + orthogonal seeding, one case per organ
    python smoke_test/smoke_test.py --data_root $DATA --level quick

    # ~100 min: all six methods of Table 1, two cases per organ
    python smoke_test/smoke_test.py --data_root $DATA --level full

    # only re-compare an existing smoke output directory
    python smoke_test/smoke_test.py --data_root $DATA --level full --compare_only

Exit code 0 = every number within tolerance (PASS), 1 = at least one mismatch.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

GRID = {'SLiver07': ['sliver_0001', 'sliver_0002'],
        'MSD_Pancreas': ['pancreas_001', 'pancreas_004'],
        'MSD_Spleen': ['spleen_10', 'spleen_12']}
LEVELS = {
    'quick': (['orthogonal', 'contour_learned'], 1),
    'full': (['sli2vol_1seed', 'orthogonal', 'mask_anchors', 'sdf_anchors', 'contour_rule', 'contour_learned'], 2),
}
INFER = ('MSD_Pancreas', 'pancreas_001')     # segment.py (inference-only path) is checked on this case
TOL = {'dice': 0.005, 'chains': 0.02,      # a single chain before fusion drifts more across GPUs than the fused mask
       'nsd_2mm': 0.01, 'assd_mm': 0.10}


def run(cmd, log):
    print('  $ ' + ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f'  FAILED (exit {r.returncode}); last lines of {log}:')
        print(''.join(open(log).readlines()[-25:]))
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', required=True, help='folder holding SLiver07/, MSD_Pancreas/, MSD_Spleen/')
    ap.add_argument('--level', choices=sorted(LEVELS), default='quick')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'smoke'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--compare_only', action='store_true')
    ap.add_argument('--no_rescore', action='store_true', help='skip NSD / ASSD (Dice and chain Dice only)')
    a = ap.parse_args()

    methods, n = LEVELS[a.level]
    cases = {s: c[:n] for s, c in GRID.items()}
    ref = json.load(open(os.path.join(REPO, 'reference_results', 'grid.json')))
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, 'smoke_log.txt')

    for s in cases:
        for c in cases[s]:
            for sub in ('imagesTr', 'labelsTr'):
                p = os.path.join(a.data_root, s, sub, c + '.nii.gz')
                if not os.path.exists(p):
                    sys.exit(f'missing {p}: prepare the data first (README, Data)')

    if not a.compare_only:
        import torch
        if a.device == 'cuda' and not torch.cuda.is_available():
            print('WARNING: no GPU visible; this will be very slow on CPU.')
        data = [f'--data={s}={os.path.join(a.data_root, s)}' for s in cases]
        sel = ','.join(c for cs in cases.values() for c in cs)
        for m in methods:
            t0 = time.time()
            print(f'[{m}] running {sum(len(v) for v in cases.values())} cases', flush=True)
            run([sys.executable, 'evaluate.py', '--method', m, *data, '--cases', sel,
                 '--device', a.device, '--out_root', os.path.join(a.out, m)], log)
            if not a.no_rescore:
                run([sys.executable, 'rescore.py', *[f'--root={s}={os.path.join(a.data_root, s)}' for s in cases],
                     '--result', f'{m}={os.path.join(a.out, m)}', '--out', os.path.join(a.out, m, 'scores_mm.json'),
                     '--workers', '2'], log)
            print(f'[{m}] done in {(time.time() - t0) / 60:.1f} min', flush=True)
        # inference path: segment.py from the three largest cross-sections of one label, in the scan geometry
        src, case = INFER
        print(f'[segment.py] {case}', flush=True)
        run([sys.executable, 'segment.py', '--image', os.path.join(a.data_root, src, 'imagesTr', case + '.nii.gz'),
             '--simulate_from_label', os.path.join(a.data_root, src, 'labelsTr', case + '.nii.gz'),
             '--out', os.path.join(a.out, 'segment', case + '.nii.gz'), '--device', a.device], log)

    # ------------------------------------------------------------------ compare
    print(f'\n{"method":16s} {"case":14s} {"metric":10s} {"expected":>9s} {"got":>9s} {"|diff|":>8s}')
    n_bad = n_all = 0
    worst = {}
    for m in methods:
        pc_path = os.path.join(a.out, m, 'per_case.json')
        if not os.path.exists(pc_path):
            print(f'{m}: no output at {pc_path}'); n_bad += 1; continue
        got = {r['case_id']: r for r in json.load(open(pc_path)).values()}
        mm = {}
        mp = os.path.join(a.out, m, 'scores_mm.json')
        if os.path.exists(mp) and not a.no_rescore:
            for lst in next(iter(json.load(open(mp)).values())).values():
                for r in lst:
                    mm[r['case_id']] = r
        for s, cs in cases.items():
            for c in cs:
                e = ref[m][s][c]
                if c not in got:
                    print(f'{m:16s} {c:14s} MISSING'); n_bad += 1; continue
                pairs = [('dice', e['dice'], got[c]['vol_dice'])]
                for ax, v in e['chains'].items():
                    pairs.append((f'chain_{ax[:3]}', v, got[c].get('per_axis_vol_dice', {}).get(ax, np.nan)))
                if c in mm:
                    pairs.append(('nsd_2mm', e['nsd_2mm'], mm[c]['nsd_2mm']))
                    if e['assd_mm'] is not None:
                        pairs.append(('assd_mm', e['assd_mm'], mm[c]['ahd_mm']))
                for key, ev, gv in pairs:
                    tol = TOL['chains' if key.startswith('chain') else key]
                    d = abs(float(gv) - float(ev)) if gv == gv else float('inf')
                    n_all += 1
                    bad = d > tol
                    n_bad += bad
                    worst[key.split('_')[0]] = max(worst.get(key.split('_')[0], 0.0), d)
                    print(f'{m:16s} {c:14s} {key:10s} {ev:9.4f} {float(gv):9.4f} {d:8.4f}{"  <-- MISMATCH" if bad else ""}')
    # segment.py: Dice of the written mask against the label file, both in the scan geometry
    src, case = INFER
    sp = os.path.join(a.out, 'segment', case + '.nii.gz')
    if os.path.exists(sp):
        import nibabel as nib
        p_ = np.asanyarray(nib.load(sp).dataobj) > 0
        g_ = np.asanyarray(nib.load(os.path.join(a.data_root, src, 'labelsTr', case + '.nii.gz')).dataobj) > 0
        gv = 2 * (p_ & g_).sum() / max(p_.sum() + g_.sum(), 1)
        ev = ref['contour_learned'][src][case]['dice']
        d = abs(gv - ev); bad = d > TOL['dice']; n_all += 1; n_bad += bad
        worst['dice'] = max(worst.get('dice', 0.0), d)
        print(f'{"segment.py":16s} {case:14s} {"dice":10s} {ev:9.4f} {gv:9.4f} {d:8.4f}{"  <-- MISMATCH" if bad else ""}')
    else:
        print(f'segment.py: no output at {sp}'); n_bad += 1
    print('\nlargest |difference| per metric: ' + ', '.join(f'{k} {v:.4f}' for k, v in worst.items()))
    print(f'tolerances: Dice {TOL["dice"]}, single-chain Dice {TOL["chains"]}, NSD {TOL["nsd_2mm"]}, ASSD {TOL["assd_mm"]} mm')
    if n_bad == 0:
        print(f'PASS: all {n_all} numbers reproduce the paper within tolerance')
    else:
        print(f'FAIL: {n_bad} of {n_all} numbers outside tolerance; first run prepare_data.py check, then compare GPU and library versions')
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == '__main__':
    main()
