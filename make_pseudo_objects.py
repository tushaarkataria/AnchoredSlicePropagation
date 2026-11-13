"""
CPU job: supervoxel maps for the unlabelled training CTs (design doc 5.4).

    python make_pseudo_objects.py --source SourceA=/.../Chaos --source SourceD=... \
        --out /.../pseudo --workers 16 [--limit N] [--inplane 256] [--n_segments 400,1500]

Writes <out>/<Source>/<case>_img.npy, <case>_sv.npz and <out>/cases.json
(one entry per case; re-running skips cases whose files already exist).
"""
import os, sys, json, glob, time, argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contours.pseudo_objects import process_case


def _one(job):
    src, img, out_img, out_sv, inplane, nseg, comp = job
    try:
        t = time.time()
        meta = process_case(img, out_img, out_sv, inplane, nseg, comp)
        meta.update({'source': src, 'case_id': os.path.basename(img).replace('.nii.gz', ''), 'seconds': time.time() - t})
        return meta, None
    except Exception as e:  # noqa
        return {'source': src, 'image': img}, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', required=True, help='NAME=/root (with imagesTr/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='per source, 0 = all')
    ap.add_argument('--inplane', type=int, default=256)
    ap.add_argument('--n_segments', default='400,1500')
    ap.add_argument('--compactness', type=float, default=0.05)
    a = ap.parse_args()
    nseg = tuple(int(v) for v in a.n_segments.split(','))
    jobs, cases = [], []
    for spec in a.source:
        name, root = spec.split('=', 1)
        imgs = sorted(f for f in glob.glob(os.path.join(root, 'imagesTr', '*.nii.gz')) if not os.path.basename(f).startswith('._'))
        if a.limit: imgs = imgs[:a.limit]
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        for img in imgs:
            cid = os.path.basename(img).replace('.nii.gz', '')
            oi, os_ = os.path.join(a.out, name, f'{cid}_img.npy'), os.path.join(a.out, name, f'{cid}_sv.npz')
            if os.path.exists(oi) and os.path.exists(os_):
                cases.append({'source': name, 'case_id': cid, 'image': img, 'img_npy': oi, 'sv_npz': os_, 'cached': True}); continue
            jobs.append((name, img, oi, os_, a.inplane, nseg, a.compactness))
    print(f'{len(cases)} cached, {len(jobs)} to process, workers={a.workers}', flush=True)
    errs = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            meta, err = f.result()
            if err:
                errs.append((meta, err)); print(f'[ERR] {meta}\n{err}', flush=True)
            else:
                cases.append(meta)
                print(f"[{i+1}/{len(jobs)}] {meta['source']}/{meta['case_id']} shape={meta['shape']} n_sv={meta['n_sv']} {meta['seconds']:.0f}s", flush=True)
    cases.sort(key=lambda c: (c['source'], c['case_id']))
    with open(os.path.join(a.out, 'cases.json'), 'w') as f:
        json.dump(cases, f, indent=1)
    print(f'done: {len(cases)} cases, {len(errs)} errors -> {a.out}/cases.json')


if __name__ == '__main__':
    main()
