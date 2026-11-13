"""
Prepare the test data in the layout evaluate.py expects:

    <DATA>/<NAME>/imagesTr/<case>.nii.gz
    <DATA>/<NAME>/labelsTr/<case>.nii.gz      (binary, or any label > 0 counts as foreground)

Development grid
    python prepare_data.py sliver07 --scans training-scans.zip --labels training-labels.zip --out $DATA
    python prepare_data.py msd --task_dir Task07_Pancreas --name MSD_Pancreas --out $DATA
    python prepare_data.py msd --task_dir Task09_Spleen   --name MSD_Spleen   --out $DATA
Held-out cohort
    python prepare_data.py amos --amos_dir amos22 --out $DATA                    # 15 AMOSva_<organ> folders
    python prepare_data.py hepatic_vessel --task_dir Task08_HepaticVessel --out $DATA   # HV_vessel, HV_tumour
Verify against the folders used for the paper (case lists, shapes, spacing, orientation):
    python prepare_data.py check --data_root $DATA

Images are symlinked, never copied, except SLiver07, which is converted from
MetaImage (.mhd/.raw) to NIfTI with SimpleITK (no resampling).
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))

# AMOS22 label ids -> folder suffix, exactly as in AMOS's dataset.json (including its 'arota' spelling)
AMOS_LABELS = {1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gall_bladder', 5: 'esophagus', 6: 'liver',
               7: 'stomach', 8: 'arota', 9: 'postcava', 10: 'pancreas', 11: 'right_adrenal_gland',
               12: 'left_adrenal_gland', 13: 'duodenum', 14: 'bladder', 15: 'prostate_uterus'}
HV_LABELS = {1: 'vessel', 2: 'tumour'}


def _images(d):
    return sorted(f for f in os.listdir(d) if f.endswith('.nii.gz') and not f.startswith('._'))


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


# ----------------------------------------------------------------------------
# SLiver07 (MetaImage -> NIfTI)
# ----------------------------------------------------------------------------
def _unpack(path, tmp):
    if os.path.isdir(path):
        return path
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    return tmp


def cmd_sliver07(a):
    import SimpleITK as sitk
    out = os.path.join(a.out, 'SLiver07')
    tmp = tempfile.mkdtemp()
    try:
        scans = _unpack(a.scans, os.path.join(tmp, 's'))
        labels = _unpack(a.labels, os.path.join(tmp, 'l'))
        imgs = sorted(glob.glob(os.path.join(scans, '**', 'liver-orig*.mhd'), recursive=True))
        if not imgs:
            raise SystemExit(f'no liver-orig*.mhd under {a.scans}')
        for ip in imgs:
            num = int(re.findall(r'(\d+)', os.path.basename(ip))[-1])
            lp = glob.glob(os.path.join(labels, '**', f'liver-seg{num:03d}.mhd'), recursive=True)
            if not lp:
                print(f'  skip {ip}: no label'); continue
            case = f'sliver_{num:04d}'
            for src, sub in ((ip, 'imagesTr'), (lp[0], 'labelsTr')):
                dst = os.path.join(out, sub, case + '.nii.gz')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    sitk.WriteImage(sitk.ReadImage(src), dst)
            print(f'  {case}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'SLiver07 -> {out}')


# ----------------------------------------------------------------------------
# MSD tasks used as they are (Task07 pancreas: pancreas + tumour = foreground)
# ----------------------------------------------------------------------------
def cmd_msd(a):
    out = os.path.join(a.out, a.name)
    for f in _images(os.path.join(a.task_dir, 'imagesTr')):
        lab = os.path.join(a.task_dir, 'labelsTr', f)
        if os.path.exists(lab):
            _link(os.path.join(a.task_dir, 'imagesTr', f), os.path.join(out, 'imagesTr', f))
            _link(lab, os.path.join(out, 'labelsTr', f))
    print(f'{a.name}: {len(_images(os.path.join(out, "imagesTr")))} cases -> {out}')


# ----------------------------------------------------------------------------
# multi-label -> one binary folder per structure (AMOS validation, MSD hepatic vessel)
# ----------------------------------------------------------------------------
def _split_case(job):
    img_path, lab_path, out_root, prefix, organs = job
    case = os.path.basename(img_path)
    lab_img = nib.load(lab_path)
    lab = np.asanyarray(lab_img.dataobj).astype(np.int16)
    present = []
    for oid, oname in organs.items():
        m = (lab == oid)
        if not m.any():            # structure absent in this case: not a test case for it
            continue
        root = os.path.join(out_root, f'{prefix}_{oname}')
        _link(img_path, os.path.join(root, 'imagesTr', case))
        dst = os.path.join(root, 'labelsTr', case)
        if not os.path.exists(dst):
            nib.save(nib.Nifti1Image(m.astype(np.uint8), lab_img.affine, lab_img.header), dst)
        present.append(oname)
    return case, present


def _split(images, labels, out, prefix, organs, ct_only, workers):
    jobs = []
    for f in _images(images):
        if ct_only:
            num = re.findall(r'(\d+)', f)
            if num and int(num[-1]) >= 500:      # AMOS: ids >= 500 are MRI
                continue
        lab = os.path.join(labels, f)
        if os.path.exists(lab):
            jobs.append((os.path.join(images, f), lab, out, prefix, organs))
    counts = {}
    with ProcessPoolExecutor(workers) as ex:
        for _, present in ex.map(_split_case, jobs):
            for o in present:
                counts[o] = counts.get(o, 0) + 1
    for o, c in sorted(counts.items()):
        print(f'  {prefix}_{o}: {c} cases')


def cmd_amos(a):
    _split(os.path.join(a.amos_dir, 'imagesVa'), os.path.join(a.amos_dir, 'labelsVa'), a.out, 'AMOSva',
           AMOS_LABELS, ct_only=True, workers=a.workers)


def cmd_hepatic_vessel(a):
    _split(os.path.join(a.task_dir, 'imagesTr'), os.path.join(a.task_dir, 'labelsTr'), a.out, 'HV',
           HV_LABELS, ct_only=False, workers=a.workers)


# ----------------------------------------------------------------------------
# check against the paper's folders
# ----------------------------------------------------------------------------
def cmd_check(a):
    man = {k: v for k, v in json.load(open(os.path.join(HERE, 'data_manifest.json'))).items() if not k.startswith('_')}
    names = [a.only] if a.only else list(man)
    n_bad = 0
    for name in names:
        root = os.path.join(a.data_root, name)
        if not os.path.isdir(root):
            print(f'{name:28s} MISSING ({root})'); n_bad += 1; continue
        have = {f[:-7] for f in _images(os.path.join(root, 'imagesTr'))
                if os.path.exists(os.path.join(root, 'labelsTr', f))}
        exp = man[name]
        miss, extra = sorted(set(exp) - have), sorted(have - set(exp))
        hdr_bad = []
        for c in sorted(set(exp) & have):
            im = nib.load(os.path.join(root, 'imagesTr', c + '.nii.gz'))
            got = {'shape': list(map(int, im.shape[:3])), 'spacing': [round(float(z), 4) for z in im.header.get_zooms()[:3]],
                   'axcodes': ''.join(nib.aff2axcodes(im.affine))}
            if got['shape'] != exp[c]['shape'] or got['axcodes'] != exp[c]['axcodes'] or \
                    not np.allclose(got['spacing'], exp[c]['spacing'], atol=1e-3):
                hdr_bad.append((c, got, exp[c]))
        ok = not miss and not extra and not hdr_bad
        n_bad += not ok
        print(f'{name:28s} {"OK " if ok else "BAD"} {len(have)}/{len(exp)} cases'
              + (f', missing {miss[:5]}' if miss else '') + (f', extra {extra[:5]}' if extra else '')
              + (f', header mismatch e.g. {hdr_bad[0]}' if hdr_bad else ''))
    print('all folders match' if n_bad == 0 else f'{n_bad} folder(s) differ from the paper setup')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sliver07'); p.add_argument('--scans', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True); p.set_defaults(fn=cmd_sliver07)
    p = sub.add_parser('msd'); p.add_argument('--task_dir', required=True); p.add_argument('--name', required=True)
    p.add_argument('--out', required=True); p.set_defaults(fn=cmd_msd)
    p = sub.add_parser('amos'); p.add_argument('--amos_dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8); p.set_defaults(fn=cmd_amos)
    p = sub.add_parser('hepatic_vessel'); p.add_argument('--task_dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8); p.set_defaults(fn=cmd_hepatic_vessel)
    p = sub.add_parser('check'); p.add_argument('--data_root', required=True); p.add_argument('--only', default=None)
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
