"""
Label-free pseudo-objects for training the contour readout (design doc 5.4).

Pipeline (CPU, make_pseudo_objects.py):
  NIfTI -> RAS canonical -> HU window [-300, 400] -> uint8 (Z, H, W)   [same as the eval loader]
        -> 3D SLIC supervoxels at a reduced in-plane size (default 256)  [two scales]
        -> saved: <case>_img.npy (uint8, native size, memory-mappable)
                  <case>_sv.npz  (uint16 supervoxel maps + per-supervoxel mean intensity)

Pseudo-objects are built ON THE FLY at training time (sample_pseudo_object):
a random supervoxel merged with 0..n_merge adjacent ones, preferring similar
mean intensity, so part of the object's boundary lies on weak edges.
The object's cross-sections along ANY axis are exact -> contour targets.
"""
from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

HU_MIN, HU_MAX = -300.0, 400.0


# ----------------------------------------------------------------------------
# preprocessing identical to the eval loader (MONAI Orientationd RAS + window)
# ----------------------------------------------------------------------------

def fix_hu_offset(data: np.ndarray) -> np.ndarray:
    """Undo the +1024 storage offset of the shared CHAOS CT copy (39/40 files are
    uint16 with air == 0 and soft tissue ~1084). The paper's loader windows those
    as-is and saturates ~60 % of the voxels; pseudo-objects need real anatomy, so
    shift any volume with no negative voxels back to HU. Out-of-FOV padding
    (e.g. 50944) is mapped to air. All test sets and the other training sources
    are already in HU (min -1024/-2048), so they pass through unchanged."""
    if data.min() >= 0:
        data = data - 1024.0
    data[data > 3071.0] = -1024.0
    return data


def load_volume_u8(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """uint8 (Z, H, W) volume in the eval frame; returns spacing (z, h, w) mm."""
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj).astype(np.float32)          # (X, Y, Z) RAS
    if data.ndim == 4:
        data = data[..., 0]
    data = fix_hu_offset(data)
    data = np.clip((data - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
    vol = np.moveaxis(data, -1, 0)                              # (Z, X, Y) == eval (Z, H, W)
    zr = img.header.get_zooms()[:3]
    return np.ascontiguousarray((vol * 255.0 + 0.5).astype(np.uint8)), (float(zr[2]), float(zr[0]), float(zr[1]))


def downsample_inplane(vol_u8: np.ndarray, size: int) -> np.ndarray:
    """Block-mean downsample (Z, H, W) -> (Z, size, size') keeping aspect;
    only integer factors (512 -> 256) are used in practice."""
    Z, H, W = vol_u8.shape
    f = max(1, int(round(H / size)))
    if f == 1:
        return vol_u8.astype(np.float32) / 255.0
    Hc, Wc = (H // f) * f, (W // f) * f
    v = vol_u8[:, :Hc, :Wc].astype(np.float32).reshape(Z, Hc // f, f, Wc // f, f).mean((2, 4))
    return v / 255.0


# ----------------------------------------------------------------------------
# supervoxels
# ----------------------------------------------------------------------------

def supervoxels(vol01: np.ndarray, n_segments: int, compactness: float = 0.05,
                spacing_zhw: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                sigma: float = 1.0) -> np.ndarray:
    """3D SLIC on a (Z, H, W) float volume in [0, 1]. Returns int32 labels
    starting at 1 (0 is never used so that 0 can mean 'background' later)."""
    from skimage.segmentation import slic
    lab = slic(vol01, n_segments=n_segments, compactness=compactness, sigma=sigma,
               spacing=spacing_zhw, channel_axis=None, start_label=1,
               enforce_connectivity=True)
    return lab.astype(np.int32)


def region_stats(lab: np.ndarray, vol01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """mean intensity and voxel count per label id (index = label id)."""
    n = int(lab.max()) + 1
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    s = np.bincount(lab.ravel(), weights=vol01.ravel().astype(np.float64), minlength=n)
    return (s / np.maximum(cnt, 1)).astype(np.float32), cnt


def adjacency(lab: np.ndarray) -> Dict[int, set]:
    """Region adjacency (6-connectivity) as {label: set(neighbour labels)}."""
    # Dense (n x n) bool incidence via fancy indexing: ~0.5 s on a 46M-voxel map
    # with 1500 labels, versus ~7 s for a sort-based np.unique over all pairs.
    n = int(lab.max()) + 1
    M = np.zeros((n, n), bool)
    for ax in range(3):
        a = np.moveaxis(lab, ax, 0)
        p, q = a[:-1], a[1:]
        m = p != q
        M[p[m], q[m]] = True
    M |= M.T
    adj: Dict[int, set] = {}
    for u, v in zip(*np.nonzero(M)):
        adj.setdefault(int(u), set()).add(int(v))
    return adj


# ----------------------------------------------------------------------------
# pseudo-object sampling (training time)
# ----------------------------------------------------------------------------

class SupervoxelVolume:
    """One saved case: supervoxel map(s) + stats + adjacency, lazily built."""

    def __init__(self, sv_npz_path: str):
        z = np.load(sv_npz_path)
        self.scales = sorted(k[len('lab_'):] for k in z.files if k.startswith('lab_'))
        self.lab = {s: z[f'lab_{s}'] for s in self.scales}
        self.n_labels = {s: int(self.lab[s].max()) for s in self.scales}
        self.mean = {s: z[f'mean_{s}'] for s in self.scales}
        self.count = {s: z[f'count_{s}'] for s in self.scales}
        self.factor = int(z['factor']) if 'factor' in z.files else 1
        self._adj: Dict[str, Dict[int, set]] = {}
        # affinity flow between adjacent supervoxels (make_pseudo_flow.py), lazily loaded
        self.flow_path = sv_npz_path[:-len('_sv.npz')] + '_flow.npz' if sv_npz_path.endswith('_sv.npz') else None
        self._flow: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._flow_scales: Optional[List[str]] = None

    def flow_scales(self) -> List[str]:
        """Scales for which this case has affinity-flow data (empty if no flow file)."""
        if self._flow_scales is None:
            if self.flow_path is None or not os.path.exists(self.flow_path):
                self._flow_scales = []
            else:
                with np.load(self.flow_path) as z:
                    self._flow_scales = sorted(k[len('flow_'):] for k in z.files
                                               if k.startswith('flow_') and f'pairs_{k[len("flow_"):]}' in z.files)
        return self._flow_scales

    def flow(self, scale: str) -> Dict[Tuple[int, int], float]:
        """{(a, b): consistency in [0, 1]} for adjacent pairs, symmetric keys."""
        if scale not in self._flow:
            if self.flow_path is None or not os.path.exists(self.flow_path):
                raise FileNotFoundError(f'no affinity flow file for {self.flow_path} (run make_pseudo_flow.py)')
            z = np.load(self.flow_path)
            pairs, val = z[f'pairs_{scale}'], np.asarray(z[f'flow_{scale}'], np.float64)
            # raw consistency is small (median ~0.02, max ~0.35 on real cases) because most
            # affinity mass stays inside a supervoxel; rescale by the per-case 99th percentile
            # so that `flow_bias` acts on [0, 1] regardless of case/scale.
            hi = float(np.percentile(val, 99)) if val.size else 1.0
            if hi > 0:
                val = np.clip(val / hi, 0.0, 1.0)
            d = {}
            for (a, b), v in zip(pairs.tolist(), val.tolist()):
                d[(a, b)] = v; d[(b, a)] = v
            self._flow[scale] = d
        return self._flow[scale]

    def adj(self, scale: str) -> Dict[int, set]:
        if scale not in self._adj:
            self._adj[scale] = adjacency(self.lab[scale])
        return self._adj[scale]

    def sample(self, rng: np.random.RandomState, scale: Optional[str] = None,
               n_merge_max: int = 5, min_voxels: int = 200, similar_bias: float = 8.0,
               merge: str = 'intensity', flow_bias: float = 6.0
               ) -> Optional[np.ndarray]:
        """A random merged region as a bool (Z, Hs, Ws) mask at the supervoxel
        resolution. merge='intensity': neighbours picked with probability
        ~ exp(-similar_bias * |mean diff|) (weak-edge faces preferred).
        merge='flow': ~ exp(flow_bias * c) where c is the frozen affinity's
        cross-slice flow consistency between the candidate and the members
        (objects whose faces the propagation model itself cannot see).
        None if nothing suitable."""
        # scale: None = random over all scales in the file; 'ml10,400' = random over
        # the listed ones that this case has (a case without any of them yields None).
        want = [x for x in (scale.split(',') if scale else self.scales) if x in self.scales]
        if merge == 'flow':
            # a scale added after make_pseudo_flow.py ran has labels but no flow yet:
            # skip it rather than crash the training loop (run make_pseudo_flow.py --add_missing)
            want = [x for x in want if x in self.flow_scales()]
        if not want:
            return None
        scale = want[rng.randint(len(want))]
        lab, mean, cnt, adj = self.lab[scale], self.mean[scale], self.count[scale], self.adj(scale)
        fl = self.flow(scale) if merge == 'flow' else None
        ids = np.nonzero(cnt >= min_voxels)[0]
        ids = ids[ids > 0]
        if ids.size == 0:
            return None
        seed = int(ids[rng.randint(ids.size)])
        members = {seed}
        for _ in range(rng.randint(0, n_merge_max + 1)):
            cand = sorted(set().union(*[adj.get(m, set()) for m in members]) - members)
            if not cand:
                break
            if fl is not None:
                c_ = np.array([max(fl.get((m, c), 0.0) for m in members) for c in cand])
                w = np.exp(flow_bias * c_)
            else:
                cm = np.array([mean[c] for c in cand]); ref = np.mean([mean[m] for m in members])
                w = np.exp(-similar_bias * np.abs(cm - ref))
            w /= w.sum()
            members.add(int(cand[rng.choice(len(cand), p=w)]))
        lut = np.zeros(int(self.n_labels[scale]) + 1, bool); lut[list(members)] = True
        return lut[lab]                                            # LUT: ~10x faster than np.isin on 40M voxels


def _fit(a: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Crop/zero-pad `a` along `axis` to length n (upsample_mask semantics)."""
    L = a.shape[axis]
    if L == n:
        return a
    if L > n:
        return np.take(a, np.arange(n), axis=axis)
    pad = [(0, 0)] * a.ndim; pad[axis] = (0, n - L)
    return np.pad(a, pad)


def upsample_slice(obj_s: np.ndarray, factor: int, axis: int, k: int, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Full-resolution 2-D slice k of permute_to_axis(upsample_mask(obj_s, factor, shape_hw), axis)
    without materialising the volume. axis 0 -> (H, W); 1 -> (Z, W); 2 -> (Z, H)."""
    H, W = shape_hw
    f = factor
    if axis == 0:
        m = np.repeat(np.repeat(obj_s[k], f, 0), f, 1) if f > 1 else obj_s[k]
        return _fit(_fit(m, H, 0), W, 1)
    Z, Hs, Ws = obj_s.shape
    if axis == 1:                                   # full-res row k of H  -> (Z, W)
        if k >= H or k // f >= Hs:
            return np.zeros((Z, W), bool)
        m = obj_s[:, k // f, :]
        return _fit(np.repeat(m, f, 1) if f > 1 else m, W, 1)
    if k >= W or k // f >= Ws:                      # full-res column k of W -> (Z, H)
        return np.zeros((Z, H), bool)
    m = obj_s[:, :, k // f]
    return _fit(np.repeat(m, f, 1) if f > 1 else m, H, 1)


def object_slice_areas(obj_s: np.ndarray, factor: int, axis: int, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Per-slice full-resolution areas along `axis` (length Z, H or W), computed
    on the low-res object. Exact when H, W are multiples of factor * (Hs, Ws)
    (the 512 = 2 x 256 case); otherwise off only at the cropped border."""
    H, W = shape_hw
    f = factor
    if axis == 0:
        return obj_s.reshape(obj_s.shape[0], -1).sum(1) * (f * f)
    if axis == 1:
        a = np.repeat(obj_s.sum((0, 2)) * f, f)
        return _fit(a, H, 0)
    a = np.repeat(obj_s.sum((0, 1)) * f, f)
    return _fit(a, W, 0)


def upsample_mask(mask_s: np.ndarray, factor: int, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest upsample (Z, Hs, Ws) -> (Z, H, W) by integer factor, padded/cropped to shape_hw."""
    if factor == 1:
        m = mask_s
    else:
        m = np.repeat(np.repeat(mask_s, factor, 1), factor, 2)
    Z = m.shape[0]
    out = np.zeros((Z,) + tuple(shape_hw), bool)
    h, w = min(m.shape[1], shape_hw[0]), min(m.shape[2], shape_hw[1])
    out[:, :h, :w] = m[:, :h, :w]
    return out


def process_case(img_path: str, out_img: str, out_sv: str, inplane: int = 256,
                 n_segments: Tuple[int, ...] = (400, 1500), compactness: float = 0.05) -> dict:
    """Full CPU pipeline for one case; writes the two files, returns metadata."""
    vol, spacing = load_volume_u8(img_path)
    np.save(out_img, vol)
    small = downsample_inplane(vol, inplane)
    factor = vol.shape[1] // small.shape[1]
    sp = (spacing[0] / (spacing[1] * factor), 1.0, 1.0)          # relative spacing of the small grid
    payload = {'factor': np.int32(factor), 'shape': np.array(vol.shape, np.int32),
               'spacing_zhw': np.array(spacing, np.float32)}
    for n in n_segments:
        lab = supervoxels(small, n, compactness, sp)
        mean, cnt = region_stats(lab, small)
        payload[f'lab_{n}'] = lab.astype(np.uint16); payload[f'mean_{n}'] = mean; payload[f'count_{n}'] = cnt.astype(np.int32)
    np.savez_compressed(out_sv, **payload)
    return {'image': img_path, 'img_npy': out_img, 'sv_npz': out_sv, 'shape': list(vol.shape),
            'spacing_zhw': list(spacing), 'factor': factor, 'n_segments': list(n_segments),
            'n_sv': {str(n): int(payload[f'lab_{n}'].max()) for n in n_segments}}
