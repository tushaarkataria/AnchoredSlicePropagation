"""
Print Tables 1-3 of the paper from per-case results.

Either from the shipped reference files (what the paper reports):
    python make_tables.py --grid reference_results/grid.json --cohort reference_results/cohort.json

or from your own runs, laid out as scripts/run_grid.sh and scripts/run_cohort.sh write them:
    python make_tables.py --results_dir results          # reads results/grid/<method>/ and results/cohort/<structure>/<method>/
    python make_tables.py --results_dir results --save results/collected   # also writes grid.json / cohort.json there

and compare a run against the reference case by case:
    python make_tables.py --results_dir results --compare reference_results
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

METHODS = ['sli2vol_1seed', 'orthogonal', 'mask_anchors', 'sdf_anchors', 'contour_rule', 'contour_learned']
LABELS = {'sli2vol_1seed': 'Sli2Vol, 1 seed', 'orthogonal': 'Orthogonal seeds, mask', 'mask_anchors': 'Mask + anchors',
          'sdf_anchors': 'SDF + anchors', 'contour_rule': 'Contour tracking, rule-based',
          'contour_learned': 'Contour tracking, learned'}
GRID_ORGANS = [('SLiver07', 'Liver'), ('MSD_Pancreas', 'Pancreas'), ('MSD_Spleen', 'Spleen')]
COHORT = [('AMOSva_liver', 'Liv'), ('AMOSva_spleen', 'Spl'), ('AMOSva_left_kidney', 'LKid'), ('AMOSva_right_kidney', 'RKid'),
          ('AMOSva_stomach', 'Sto'), ('AMOSva_bladder', 'Bla'), ('AMOSva_prostate_uterus', 'Pro'), ('AMOSva_pancreas', 'Pan'),
          ('AMOSva_gall_bladder', 'GB'), ('AMOSva_duodenum', 'Duo'), ('AMOSva_esophagus', 'Eso'), ('AMOSva_arota', 'Aor'),
          ('AMOSva_postcava', 'PCv'), ('AMOSva_left_adrenal_gland', 'LAd'), ('AMOSva_right_adrenal_gland', 'RAd'),
          ('HV_tumour', 'HTu'), ('HV_vessel', 'HVe')]
REF = 'orthogonal'


# ----------------------------------------------------------------------------
# collecting a results tree into the reference format
# ----------------------------------------------------------------------------
def _per_case(d):
    rows = {}
    files = sorted(glob.glob(os.path.join(d, 'per_case_*of*.json'))) or [os.path.join(d, 'per_case.json')]
    for f in files:
        if os.path.exists(f):
            rows.update(json.load(open(f)))
    return rows


def collect(results_dir):
    grid, cohort = {}, {}
    for m in METHODS:
        d = os.path.join(results_dir, 'grid', m)
        rows = _per_case(d)
        if rows:
            mm = {}
            f = os.path.join(d, 'scores_mm.json')
            if os.path.exists(f):
                for src, lst in next(iter(json.load(open(f)).values())).items():
                    for r in lst:
                        mm[(src, r['case_id'])] = r
            g = {}
            for r in rows.values():
                x = mm.get((r['source'], r['case_id']), {})
                assd = x.get('ahd_mm')
                g.setdefault(r['source'], {})[r['case_id']] = {
                    'dice': r['vol_dice'], 'nsd_2mm': x.get('nsd_2mm'),
                    'assd_mm': None if assd is None or assd != assd else assd,
                    'chains': r.get('per_axis_vol_dice', {})}
            grid[m] = g
        for sd in sorted(glob.glob(os.path.join(results_dir, 'cohort', '*', m))):
            rows = _per_case(sd)
            if rows:
                cohort.setdefault(m, {})[os.path.basename(os.path.dirname(sd))] = {
                    r['case_id']: r['vol_dice'] for r in rows.values()}
    return grid, cohort


def _load(path):
    if not path or not os.path.exists(path):
        return {}
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith('_')}


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def _mean(v):
    v = np.array([x for x in v if x is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def table1(grid):
    if REF not in grid:
        print('Table 1: orthogonal-seeding results missing'); return
    print('\nTable 1 -- development grid (Dice / NSD@2mm / ASSD mm per organ; paired difference vs orthogonal seeding, wins)')
    hdr = f'{"Method":30s}' + ''.join(f'{o:>24s}' for _, o in GRID_ORGANS) + f'{"dDice (wins)":>18s}{"dNSD (wins)":>17s}{"dASSD (wins)":>17s}'
    print(hdr); print('-' * len(hdr))
    for m in METHODS:
        if m not in grid:
            continue
        line = f'{LABELS[m]:30s}'
        for src, _ in GRID_ORGANS:
            rows = grid[m].get(src, {}).values()
            line += f'   {_mean([r["dice"] for r in rows]):.3f} {_mean([r["nsd_2mm"] for r in rows]):.3f} {_mean([r["assd_mm"] for r in rows]):5.2f}  ' if rows else f'{"-":>24s}'
        if m == REF:
            line += f'{"reference":>18s}'
        else:
            for key, lower in (('dice', False), ('nsd_2mm', False), ('assd_mm', True)):
                a, b = [], []
                for src, _ in GRID_ORGANS:
                    for c, r in grid[m].get(src, {}).items():
                        q = grid[REF].get(src, {}).get(c)
                        if q and r[key] is not None and q[key] is not None:
                            a.append(r[key]); b.append(q[key])
                if not a:
                    line += f'{"-":>17s}'; continue
                d = np.array(a) - np.array(b)
                wins = (d < 0).mean() if lower else (d > 0).mean()
                line += f'{d.mean():>+10.3f} ({100 * wins:3.0f}%)'
        print(line)


def table2(grid):
    print('\nTable 2 -- chain ablation, pooled Dice over the grid (each chain before fusion; fused = final)')
    print(f'{"Method":30s}{"axial":>9s}{"coronal":>9s}{"sagittal":>9s}{"fused":>9s}')
    for m in METHODS:
        if m not in grid:
            continue
        rows = [r for src in grid[m].values() for r in src.values()]
        ch = {a: _mean([r['chains'].get(a) for r in rows]) for a in ('axial', 'coronal', 'sagittal')}
        f = lambda x: f'{x:9.3f}' if np.isfinite(x) else f'{"--":>9s}'
        print(f'{LABELS[m]:30s}{f(ch["axial"])}{f(ch["coronal"])}{f(ch["sagittal"])}{f(_mean([r["dice"] for r in rows]))}')


def table3(cohort):
    if REF not in cohort:
        print('\nTable 3: orthogonal-seeding cohort results missing'); return
    print('\nTable 3 -- held-out cohort, volumetric Dice. Marks vs orthogonal seeding (paired Wilcoxon, p<0.05):')
    print('          no mark = significantly better, ° = no significant difference, † = significantly worse')
    present = [(s, a) for s, a in COHORT if any(s in cohort.get(m, {}) for m in METHODS)]
    print(f'{"Method":30s}' + ''.join(f'{a:>7s}' for _, a in present) + f'{"Mean":>8s}')
    ns = [len(cohort[REF].get(s, {})) for s, _ in present]
    print(f'{"n":30s}' + ''.join(f'{n:>7d}' for n in ns))
    for m in METHODS:
        if m not in cohort:
            continue
        line, means = f'{LABELS[m]:30s}', []
        for s, _ in present:
            cur, ref = cohort[m].get(s, {}), cohort[REF].get(s, {})
            if not cur:
                line += f'{"-":>7s}'; continue
            mu = float(np.mean(list(cur.values()))); means.append(mu)
            mark = ''
            if m != REF:
                keys = [k for k in cur if k in ref]
                x = np.array([cur[k] for k in keys]); y = np.array([ref[k] for k in keys])
                p = 1.0 if np.allclose(x, y) else wilcoxon(x, y).pvalue
                mark = '°' if p >= 0.05 else ('' if x.mean() > y.mean() else '†')
            line += f'{mu:>6.3f}{mark or " "}'
        line += f'{np.mean(means):>8.3f}' if len(means) == len(present) else f'{"-":>8s}'
        print(line)


def compare(grid, cohort, ref_dir, tol):
    rg, rc = _load(os.path.join(ref_dir, 'grid.json')), _load(os.path.join(ref_dir, 'cohort.json'))
    print(f'\nCase-by-case comparison with {ref_dir} (|volumetric Dice difference|, tolerance {tol})')
    bad = 0
    for name, mine, ref, get in (('grid', grid, rg, lambda r: r['dice']), ('cohort', cohort, rc, lambda r: r)):
        for m in METHODS:
            for s, cases in mine.get(m, {}).items():
                diffs = [(c, abs(get(v) - get(ref[m][s][c]))) for c, v in cases.items() if c in ref.get(m, {}).get(s, {})]
                if not diffs:
                    continue
                worst = max(diffs, key=lambda t: t[1])
                n_bad = sum(d > tol for _, d in diffs); bad += n_bad
                print(f'  {name:6s} {m:16s} {s:28s} n={len(diffs):4d}  max |d|={worst[1]:.4f} ({worst[0]})  over tol: {n_bad}')
    print('ALL CASES WITHIN TOLERANCE' if bad == 0 else f'{bad} case(s) over tolerance')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default=None, help='grid.json in the reference format')
    ap.add_argument('--cohort', default=None, help='cohort.json in the reference format')
    ap.add_argument('--results_dir', default=None, help='results tree written by scripts/run_grid.sh / run_cohort.sh')
    ap.add_argument('--save', default=None, help='with --results_dir: write the collected grid.json / cohort.json here')
    ap.add_argument('--compare', default=None, help='reference directory (e.g. reference_results) to compare against')
    ap.add_argument('--tol', type=float, default=0.005)
    a = ap.parse_args()
    if a.results_dir:
        grid, cohort = collect(a.results_dir)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            json.dump(grid, open(os.path.join(a.save, 'grid.json'), 'w'))
            json.dump(cohort, open(os.path.join(a.save, 'cohort.json'), 'w'))
    else:
        grid, cohort = _load(a.grid), _load(a.cohort)
    if grid:
        table1(grid); table2(grid)
    if cohort:
        table3(cohort)
    if a.compare:
        compare(grid, cohort, a.compare, a.tol)


if __name__ == '__main__':
    main()
