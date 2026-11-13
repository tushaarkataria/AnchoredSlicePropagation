"""
Train the contour READOUT (contours/readout.py) on supervoxel pseudo-objects
with a K-step unrolled contour loss. The Sli2Vol backbone is frozen.

    python train_readout.py --ckpt <sli2vol best_model.pth> --pseudo <out of make_pseudo_objects.py> \
        --out <dir> [--steps 3000 --K_max 6 --samples_per_step 4 --lr 1e-3]

Sample = (case, pseudo-object, axis, direction, start slice): the polygon of
the object's cross-section at the start slice is tracked for K steps with
contour_step_torch, feeding its own predictions forward; the loss on every
step is  (1 - soft Dice to the true cross-section)  +  Gaussian-NLL Chamfer to
the true contour  +  0.5 BCE(vanish)  +  spacing/curvature regularisers.
"""
import os, sys, json, time, math, argparse, random, threading, queue
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contours import raster as RA, anchors as AN, readout as RO
from contours import pseudo_objects as PO
from contours.pseudo_objects import SupervoxelVolume
from contours.transport import get_corr, get_corr_extra
from models.sli2vol_original import Sli2VolRegistration


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def zoom_chain(sl: np.ndarray, targets: np.ndarray, f: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale a (K+1, A, B) float chain and its bool targets in-plane by f
    (bilinear / nearest). Sizes are rounded to even numbers so the 2x affinity
    pyramid of a multi-scale readout stays aligned."""
    A, B = sl.shape[1:]
    A2, B2 = max(16, int(round(A * f / 2)) * 2), max(16, int(round(B * f / 2)) * 2)
    if (A2, B2) == (A, B):
        return sl, targets
    x = torch.from_numpy(np.ascontiguousarray(sl))[:, None]
    sl2 = F.interpolate(x, (A2, B2), mode='bilinear', align_corners=False)[:, 0].numpy()
    t = torch.from_numpy(targets.astype(np.float32))[:, None]
    t2 = F.interpolate(t, (A2, B2), mode='nearest')[:, 0].numpy() > 0.5
    return np.ascontiguousarray(sl2), t2


class PseudoChainSampler:
    """Yields training samples from <pseudo>/cases.json. One SupervoxelVolume
    per case is cached (LRU of a few cases) because the npz decompression is
    the slow part."""

    def __init__(self, cases: List[dict], K: int, rng: np.random.RandomState, cache_size: int = 6,
                 axis_probs=(0.4, 0.3, 0.3), min_area: int = 150, n_points: int = 64,
                 samples_per_case: int = 12, sv_scale: Optional[str] = None, n_merge_max: int = 5,
                 min_voxels: int = 200, obj_ml: Tuple[float, float] = (0.0, 0.0), obj_fill_min: float = 0.0,
                 max_stride: int = 1, zoom: float = 0.0, merge: str = 'intensity', flow_bias: float = 6.0):
        self.cases, self.K, self.rng = cases, K, rng
        # Multi-scale augmentation: chains with a random slice stride in
        # [1, max_stride] (thick-slice scans) and a random in-plane zoom factor
        # in [1 - zoom, 1 + zoom] applied to slices, targets and the polygon.
        self.max_stride, self.zoom = max_stride, zoom
        # 'intensity': neighbours merged by mean-intensity similarity (default);
        # 'flow': by the frozen affinity's cross-slice flow (needs <case>_flow.npz).
        self.merge, self.flow_bias = merge, flow_bias
        # Pseudo-object mix: supervoxel scale ('400' coarse / '1500' fine / None = random),
        # how many neighbours may be merged, and acceptance filters on the merged
        # object's volume in ml (0 = off) and bounding-box fill (compactness, 0 = off).
        self.sv_scale, self.n_merge_max, self.min_voxels = sv_scale, n_merge_max, min_voxels
        self.obj_ml, self.obj_fill_min = obj_ml, obj_fill_min
        self.n_rejected = 0
        self.cache, self.cache_size = {}, cache_size
        self.axis_probs, self.min_area, self.n_points = axis_probs, min_area, n_points
        # Loading a case (npz decompression + adjacency) costs seconds, a sample
        # ~0.1 s: draw several samples per case before moving on.
        self.samples_per_case, self._cur, self._cur_n = samples_per_case, None, 0

    def _next_case(self) -> dict:
        if self._cur is None or self._cur_n >= self.samples_per_case:
            self._cur, self._cur_n = self.cases[self.rng.randint(len(self.cases))], 0
        self._cur_n += 1
        return self._cur

    def _sv(self, case) -> SupervoxelVolume:
        k = case['sv_npz']
        if k not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = SupervoxelVolume(k)
        return self.cache[k]

    @staticmethod
    def object_stats(obj_s: np.ndarray, factor: int, spacing_zhw) -> Tuple[float, float]:
        """(volume in ml, bounding-box fill) of a low-res (Z, Hs, Ws) object whose
        in-plane voxels are `factor` native voxels wide."""
        n = int(obj_s.sum())
        if n == 0:
            return 0.0, 0.0
        sp = np.asarray(spacing_zhw if spacing_zhw is not None else (1.0, 1.0, 1.0), float)
        ml = n * float(sp[0] * sp[1] * sp[2]) * factor * factor / 1000.0
        bbox = 1
        for ax in range(3):
            nz = np.nonzero(obj_s.any(axis=tuple(i for i in range(3) if i != ax)))[0]
            bbox *= int(nz[-1] - nz[0] + 1)
        return ml, n / bbox

    def accept_object(self, obj_s: np.ndarray, factor: int, spacing_zhw) -> bool:
        lo, hi = self.obj_ml
        if lo <= 0 and hi <= 0 and self.obj_fill_min <= 0:
            return True
        ml, fill = self.object_stats(obj_s, factor, spacing_zhw)
        if lo > 0 and ml < lo: return False
        if hi > 0 and ml > hi: return False
        if self.obj_fill_min > 0 and fill < self.obj_fill_min: return False
        return True

    default_tries = 20

    def sample(self, K: Optional[int] = None, tries: Optional[int] = None):
        K = K or self.K
        tries = tries or self.default_tries
        for _ in range(tries):
            case = self._next_case()
            sv = self._sv(case)
            obj_s = sv.sample(self.rng, scale=self.sv_scale, n_merge_max=self.n_merge_max,
                              min_voxels=self.min_voxels, merge=self.merge, flow_bias=self.flow_bias)
            if obj_s is None:
                continue
            if not self.accept_object(obj_s, sv.factor, case.get('spacing_zhw')):
                self.n_rejected += 1
                continue
            img = np.load(case['img_npy'], mmap_mode='r')                   # (Z,H,W) uint8
            Z, H, W = img.shape
            axis = int(self.rng.choice(3, p=self.axis_probs))
            # Never materialise the full-res (Z,H,W) object (170M voxels for a
            # 660-slice CT): per-slice areas come from the low-res object and
            # only the K+1 chain slices are upsampled.
            areas = PO.object_slice_areas(obj_s, sv.factor, axis, (H, W))      # (n_slices_along_axis,)
            n_ax = areas.shape[0]
            ok = np.nonzero(areas >= self.min_area)[0]
            if ok.size < 2:
                continue
            direction = 1 if self.rng.rand() < 0.5 else -1
            stride = int(self.rng.randint(1, self.max_stride + 1))
            k0 = int(ok[self.rng.randint(ok.size)])
            ks = [k0 + direction * stride * t for t in range(K + 1)]
            if ks[-1] < 0 or ks[-1] >= n_ax:
                direction = -direction; ks = [k0 + direction * stride * t for t in range(K + 1)]
                if ks[-1] < 0 or ks[-1] >= n_ax:
                    continue
            targets = np.stack([PO.upsample_slice(obj_s, sv.factor, axis, k, (H, W)) for k in ks])  # (K+1, A, B)
            # image slices in the axis frame, only the K+1 needed (stride > 1: pick them out)
            if axis == 0:
                sl = np.stack([np.asarray(img[k]) for k in ks])                              # (K+1, H, W)
            elif axis == 1:
                sl = np.stack([np.asarray(img[:, k, :]) for k in ks])                        # (K+1, Z, W)
            else:
                sl = np.stack([np.asarray(img[:, :, k]) for k in ks])                        # (K+1, Z, H)
            sl = np.ascontiguousarray(sl).astype(np.float32) / 255.0
            if self.zoom > 0:
                f = float(self.rng.uniform(1.0 - self.zoom, 1.0 + self.zoom))
                sl, targets = zoom_chain(sl, targets, f)
            polys0 = RA.mask_to_polygons(targets[0], self.n_points)
            if not polys0:
                continue
            poly0 = max(polys0, key=RA.polygon_area)                           # track the largest component
            return {'img': sl, 'targets': targets,
                    'poly0': poly0.astype(np.float32), 'axis': axis, 'case': case['case_id'],
                    'stride': stride}
        return None


class Prefetch:
    """Background thread that keeps a queue of K_max samples ready; the
    training loop truncates to the curriculum K (unroll only reads the first
    K+1 slices). numpy releases the GIL in the heavy ops, so sampling overlaps
    with the GPU work."""

    def __init__(self, sampler: PseudoChainSampler, K: int, depth: int = 16):
        self.sampler, self.K, self.q, self.stop = sampler, K, queue.Queue(depth), False
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        while not self.stop:
            s = self.sampler.sample(self.K)
            if s is not None:
                self.q.put(s)

    def get(self):
        return self.q.get()


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------

def _crop_box(poly: torch.Tensor, target: np.ndarray, shape, margin: int = 12):
    A, B = shape
    r0, c0 = poly.detach().min(0).values.cpu().numpy(); r1, c1 = poly.detach().max(0).values.cpu().numpy()
    if target.any():
        rr, cc = np.nonzero(target); r0, r1, c0, c1 = min(r0, rr.min()), max(r1, rr.max()), min(c0, cc.min()), max(c1, cc.max())
    r0, c0 = max(int(r0) - margin, 0), max(int(c0) - margin, 0)
    r1, c1 = min(int(r1) + margin + 1, A), min(int(c1) + margin + 1, B)
    return r0, r1, c0, c1


def step_loss(poly: torch.Tensor, logvar: torch.Tensor, vanish: torch.Tensor, target: np.ndarray,
              shape, device, w_dice=1.0, w_pt=0.05, w_van=0.5, w_reg=(0.02, 0.002)):
    """Returns (loss, dict of components, dice value). target (A,B) bool."""
    out = {}
    empty = not target.any()
    van_t = torch.tensor(1.0 if empty else 0.0, device=device)
    l_van = F.binary_cross_entropy_with_logits(vanish, van_t)
    if empty:
        return w_van * l_van, {'van': float(l_van)}, None
    r0, r1, c0, c1 = _crop_box(poly, target, shape)
    pc = poly - torch.tensor([r0, c0], device=device, dtype=poly.dtype)
    soft = RA.soft_rasterize(pc, (r1 - r0, c1 - c0), sigma=1.0)
    tgt = torch.from_numpy(target[r0:r1, c0:c1].astype(np.float32)).to(device)
    dice = RA.soft_dice(soft, tgt)
    l_dice = 1.0 - dice
    # Gaussian-NLL Chamfer to the true contour points
    from skimage import measure
    cont = np.concatenate([c for c in measure.find_contours(np.pad(target.astype(np.float32), 1), 0.5)], 0) - 1.0
    if cont.shape[0] > 400:
        cont = cont[np.linspace(0, cont.shape[0] - 1, 400).astype(int)]
    ct = torch.from_numpy(cont.astype(np.float32)).to(device)
    d = torch.cdist(poly, ct).min(1).values                                  # (N,)
    l_pt = (d ** 2 / (2.0 * torch.exp(logvar)) + 0.5 * logvar).mean()
    # regularisers: even spacing, low curvature
    seg = (torch.roll(poly, -1, 0) - poly).norm(dim=1)
    l_sp = seg.var() / seg.mean().clamp_min(1e-6) ** 2
    l_cv = ((torch.roll(poly, -1, 0) - 2 * poly + torch.roll(poly, 1, 0)) ** 2).sum(1).mean()
    loss = w_dice * l_dice + w_pt * l_pt + w_van * l_van + w_reg[0] * l_sp + w_reg[1] * l_cv
    out.update({'dice': float(dice), 'pt': float(l_pt), 'van': float(l_van), 'sp': float(l_sp), 'cv': float(l_cv)})
    return loss, out, float(dice)


def unroll(model, readout, sample, K, device, disp_mode='soft', train=True):
    """Runs K steps; returns (total loss, per-step dice list, logs)."""
    img = torch.from_numpy(sample['img']).to(device)
    shape = tuple(img.shape[1:])
    poly = torch.from_numpy(sample['poly0']).to(device)
    total, dices, logs = 0.0, [], []
    for t in range(K):
        with torch.no_grad():
            corr = get_corr(model, img[t], img[t + 1])
            n_sc = getattr(readout, 'n_scales', 1) if readout is not None else 1
            corr_extra = get_corr_extra(model, img[t], img[t + 1], n_sc) if n_sc > 1 else None
        o = RO.contour_step_torch(corr, poly, readout, disp_mode=disp_mode, corr_extra=corr_extra)
        loss, comp, dice = step_loss(o['new'], o['logvar'], o['vanish'], sample['targets'][t + 1], shape, device)
        total = total + loss; logs.append(comp)
        if dice is None:                                                     # object ended: stop the chain
            break
        dices.append(dice)
        poly = o['new']
        if not train:
            poly = poly.detach()
    return total, dices, logs


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_backbone(args, device):
    model = Sli2VolRegistration(R=args.R, is_training=False, use_edge_profile=True,
                                edge_radius=args.edge_radius, edge_interval=args.edge_interval,
                                temperature=args.corr_temperature).to(device)
    if args.ckpt and args.ckpt != 'none':
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def evaluate(model, readout, val_samples, K, device, disp_mode):
    if readout is not None: readout.eval()
    per_step = np.full((len(val_samples), K), np.nan)
    with torch.no_grad():
        for i, s in enumerate(val_samples):
            _, d, _ = unroll(model, readout, s, K, device, disp_mode, train=False)
            per_step[i, :len(d)] = d
    if readout is not None: readout.train()
    return {'dice_mean': float(np.nanmean(per_step)), 'dice_last': float(np.nanmean(per_step[:, -1])),
            'dice_per_step': [float(v) for v in np.nanmean(per_step, 0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help="Sli2Vol checkpoint ('none' = random weights, smoke tests)")
    ap.add_argument('--pseudo', required=True, help='dir with cases.json from make_pseudo_objects.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--samples_per_step', type=int, default=4)
    ap.add_argument('--K_max', type=int, default=6)
    ap.add_argument('--K_curriculum', default='2:0.2,4:0.5', help='K:fraction-of-steps breakpoints before K_max')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--n_conv', type=int, default=4)
    ap.add_argument('--n_points', type=int, default=64)
    ap.add_argument('--disp_mode', choices=['soft', 'argmax'], default='soft')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--n_val', type=int, default=48)
    ap.add_argument('--samples_per_case', type=int, default=12)
    # pseudo-object mix (variants of the default: all sources, random scale, <=5 merges, no filters)
    ap.add_argument('--sources', default='', help='comma list of pseudo sources to train on (default all)')
    ap.add_argument('--sv_scale', default='', help="supervoxel scale '400' | '1500' (default: random per sample)")
    ap.add_argument('--n_merge_max', type=int, default=5)
    ap.add_argument('--min_voxels', type=int, default=200)
    ap.add_argument('--obj_ml', default='0,0', help='accept only objects with volume in [lo,hi] ml (0 = no bound)')
    ap.add_argument('--obj_fill_min', type=float, default=0.0, help='accept only objects filling >= this fraction of their bounding box')
    ap.add_argument('--sample_tries', type=int, default=20, help='draws per sample before giving up (raise with strict filters)')
    ap.add_argument('--merge', choices=['intensity', 'flow'], default='intensity',
                    help="how supervoxels are merged into pseudo-objects: 'flow' uses <case>_flow.npz from make_pseudo_flow.py")
    ap.add_argument('--flow_bias', type=float, default=6.0)
    # multi-scale: readout input and augmentation
    ap.add_argument('--n_scales', type=int, default=1, help='affinity scales read per point (1 = fine only, 2 = + 2x downsampled)')
    ap.add_argument('--max_stride', type=int, default=1, help='random slice stride in [1, max_stride] for training chains')
    ap.add_argument('--zoom', type=float, default=0.0, help='random in-plane zoom in [1-zoom, 1+zoom] (0 = off)')
    ap.add_argument('--prefetch', type=int, default=16, help='queue depth of the sampler thread (0 = synchronous)')
    ap.add_argument('--val_every', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--R', type=int, default=6)
    ap.add_argument('--edge_radius', type=int, default=3)
    ap.add_argument('--edge_interval', type=int, default=1)
    ap.add_argument('--corr_temperature', type=float, default=0.7)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    cases = json.load(open(os.path.join(args.pseudo, 'cases.json')))
    cases = [c for c in cases if os.path.exists(c['img_npy']) and os.path.exists(c['sv_npz'])]
    if args.sources:
        keep = set(args.sources.split(','))
        cases = [c for c in cases if c.get('source') in keep]
        print(f'sources {sorted(keep)} -> {len(cases)} cases', flush=True)
    obj_ml = tuple(float(x) for x in args.obj_ml.split(','))
    obj_kw = dict(sv_scale=args.sv_scale or None, n_merge_max=args.n_merge_max, min_voxels=args.min_voxels,
                  obj_ml=obj_ml, obj_fill_min=args.obj_fill_min, merge=args.merge, flow_bias=args.flow_bias)
    aug_kw = dict(max_stride=args.max_stride, zoom=args.zoom)
    rng = np.random.RandomState(args.seed); rng.shuffle(cases)
    n_val = max(1, int(len(cases) * args.val_frac))
    val_cases, train_cases = cases[:n_val], cases[n_val:]
    print(f'cases: train {len(train_cases)} val {len(val_cases)}  device {device}', flush=True)

    model = build_backbone(args, device)
    readout = RO.ContourReadout(D2=(2 * args.R + 1) ** 2, hidden=args.hidden, n_conv=args.n_conv,
                                n_scales=args.n_scales).to(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        readout.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['opt']); start = ck['step']
        for _ in range(start): sched.step()          # cosine schedule continues where the resumed run stopped

    val_sampler = PseudoChainSampler(val_cases, args.K_max, np.random.RandomState(1234), n_points=args.n_points, **obj_kw)
    val_samples = [s for s in (val_sampler.sample(tries=args.sample_tries) for _ in range(args.n_val * 2)) if s is not None][:args.n_val]
    print(f'val samples {len(val_samples)} (rejected by object filters: {val_sampler.n_rejected})', flush=True)
    train_sampler = PseudoChainSampler(train_cases, args.K_max, rng, n_points=args.n_points, samples_per_case=args.samples_per_case, **obj_kw, **aug_kw)
    train_sampler.default_tries = args.sample_tries
    prefetch = Prefetch(train_sampler, args.K_max, args.prefetch) if args.prefetch > 0 else None
    curriculum = sorted((int(k), float(fr)) for k, fr in (kv.split(':') for kv in args.K_curriculum.split(',')))

    def K_at(step):
        for k, fr in curriculum:
            if step < fr * args.steps: return k
        return args.K_max

    base = evaluate(model, None, val_samples, args.K_max, device, args.disp_mode)
    print(f'[val] hand-crafted baseline: dice_mean {base["dice_mean"]:.4f} dice_last {base["dice_last"]:.4f} per-step {np.round(base["dice_per_step"], 3).tolist()}', flush=True)
    best = -1.0; log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    log.write(json.dumps({'step': 0, 'val': base, 'baseline': True}) + '\n'); log.flush()

    t0 = time.time(); run = {}
    for step in range(start, args.steps):
        K = K_at(step); opt.zero_grad(); n_ok = 0; acc = {}
        for _ in range(args.samples_per_step):
            s = prefetch.get() if prefetch is not None else train_sampler.sample(K)
            if s is None: continue
            loss, dices, logs = unroll(model, readout, s, K, device, args.disp_mode, train=True)
            (loss / args.samples_per_step).backward(); n_ok += 1
            for c in logs:
                for k, v in c.items(): acc.setdefault(k, []).append(v)
        if n_ok == 0: continue
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 5.0)
        opt.step(); sched.step()
        for k, v in acc.items(): run.setdefault(k, []).extend(v)
        if (step + 1) % 20 == 0:
            msg = ' '.join(f'{k} {np.mean(v):.3f}' for k, v in run.items())
            print(f'step {step+1}/{args.steps} K={K} lr {sched.get_last_lr()[0]:.2e} | {msg} | {(time.time()-t0)/60:.1f} min', flush=True)
            log.write(json.dumps({'step': step + 1, 'K': K, **{k: float(np.mean(v)) for k, v in run.items()}}) + '\n'); log.flush(); run = {}
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            val = evaluate(model, readout, val_samples, args.K_max, device, args.disp_mode)
            print(f'[val] step {step+1}: dice_mean {val["dice_mean"]:.4f} dice_last {val["dice_last"]:.4f} per-step {np.round(val["dice_per_step"], 3).tolist()} (baseline {base["dice_mean"]:.4f}/{base["dice_last"]:.4f})', flush=True)
            log.write(json.dumps({'step': step + 1, 'val': val}) + '\n'); log.flush()
            ck = {'state_dict': readout.state_dict(), 'opt': opt.state_dict(), 'step': step + 1, 'val': val,
                  'config': {'D2': (2 * args.R + 1) ** 2, 'hidden': args.hidden, 'n_conv': args.n_conv, 'kernel': 5,
                             'n_scales': args.n_scales}}
            torch.save(ck, os.path.join(args.out, 'last.pth'))
            if val['dice_last'] > best:
                best = val['dice_last']; torch.save(ck, os.path.join(args.out, 'best.pth'))
                print(f'  -> best.pth (dice_last {best:.4f})', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
