"""
Shared helpers for evaluate.py and contours/transport.py.

Every function below is copied verbatim from the research code that produced
the paper's numbers (eval_registration.py, eval_triplanar.py and
utils/visualization.py of the internal repository); only the module they live
in has changed. Do not edit them if you want to reproduce the paper.
"""
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_erosion, distance_transform_edt

from models.sli2vol_original import Sli2VolRegistration


# ============================================================
# Sli2Vol intensity verification (applied after every propagation step)
# ============================================================
def verification_module(mask_img, mask_collection, img, img_collection, template):
    """
    Refine predicted mask using intensity-based likelihood and morphological operations.

    Args:
        mask_img: (H, W) predicted mask for current slice
        mask_collection: (F, H, W) history of predicted masks
        img: (1, H, W) current slice image
        img_collection: (F, 1, H, W) history of slice images
        template: unused (kept for API compatibility)

    Returns:
        (refined_mask, template)
    """
    num_class = int(np.max(mask_collection[-1]))
    mask_refine = np.zeros(mask_img.shape, dtype=np.int32)

    for nc in range(1, num_class + 1):
        mask_history = (mask_collection[-1] == nc).astype(np.float32)
        mask_img_temp = (mask_img == nc).astype(np.float32)

        mask_neg = binary_dilation(mask_history, iterations=5).astype(np.float32) - mask_history
        mask_neg = (mask_neg > 0).astype(np.float32)

        mask_neg_c = np.tile(mask_neg[None, ...], (img.shape[0], 1, 1))
        mask_pos_c = np.tile(mask_history[None, ...], (img.shape[0], 1, 1))

        pos_means, neg_means = [], []
        for ci in range(mask_pos_c.shape[0]):
            pos_vals = img_collection[-1, ci][mask_pos_c[ci] == 1]
            neg_vals = img_collection[-1, ci][mask_neg_c[ci] == 1]
            pos_means.append(np.mean(pos_vals) if pos_vals.size > 0 else np.nan)
            neg_means.append(np.mean(neg_vals) if neg_vals.size > 0 else np.nan)

        pos_means = np.array(pos_means, dtype=np.float32)
        neg_means = np.array(neg_means, dtype=np.float32)

        feature_positive = np.ones_like(img, dtype=np.float32) * pos_means[:, None, None]
        feature_negative = np.ones_like(img, dtype=np.float32) * neg_means[:, None, None]

        feature_positive = np.sum((img - feature_positive) ** 2, axis=0)
        feature_negative = np.sum((img - feature_negative) ** 2, axis=0)

        positive_likelihood = np.zeros_like(feature_positive, dtype=np.float32)
        positive_likelihood[feature_positive < feature_negative] = 1.0
        positive_likelihood = positive_likelihood * mask_img_temp
        positive_likelihood[positive_likelihood > 0] = 1.0
        positive_likelihood[~np.isfinite(positive_likelihood)] = 0.0

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)
        positive_likelihood = binary_dilation(positive_likelihood, iterations=2)
        positive_likelihood = binary_erosion(positive_likelihood, iterations=2)
        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)

        mask_refine[positive_likelihood > 0] = nc

    return mask_refine, template


# ============================================================
# 2D metric helpers (per-slice diagnostics written to per_case.json)
# ============================================================
def dice_binary(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (den + eps))


def dice_multiclass_slice(pred_hw, gt_hw):
    cls = np.unique(gt_hw)
    cls = cls[cls != 0]
    if cls.size == 0:
        return 1.0 if np.max(pred_hw) == 0 else 0.0
    return float(np.mean([dice_binary(pred_hw == c, gt_hw == c) for c in cls]))


def surface_mask(binary_hw):
    fg = binary_hw.astype(bool)
    if not fg.any():
        return np.zeros_like(fg, dtype=bool)
    er = binary_erosion(fg, iterations=1, border_value=0)
    return np.logical_and(fg, np.logical_not(er))


def symmetric_hausdorff(pred_hw, gt_hw):
    pred_fg = (pred_hw > 0)
    gt_fg = (gt_hw > 0)
    if (not pred_fg.any()) and (not gt_fg.any()):
        return 0.0
    if pred_fg.any() != gt_fg.any():
        return np.nan

    pred_s = surface_mask(pred_fg)
    gt_s = surface_mask(gt_fg)
    if not pred_s.any():
        pred_s = pred_fg
    if not gt_s.any():
        gt_s = gt_fg

    dt_pred = distance_transform_edt(~pred_s)
    dt_gt = distance_transform_edt(~gt_s)

    d1 = dt_pred[gt_s].max() if gt_s.any() else np.nan
    d2 = dt_gt[pred_s].max() if pred_s.any() else np.nan
    return float(max(d1, d2))


def symmetric_mean_surface_distance(pred_hw, gt_hw):
    pred_fg = (pred_hw > 0)
    gt_fg = (gt_hw > 0)
    if (not pred_fg.any()) and (not gt_fg.any()):
        return 0.0
    if pred_fg.any() != gt_fg.any():
        return np.nan

    pred_s = surface_mask(pred_fg)
    gt_s = surface_mask(gt_fg)
    if not pred_s.any() or not gt_s.any():
        pred_s, gt_s = pred_fg, gt_fg

    dt_pred = distance_transform_edt(~pred_s)
    dt_gt = distance_transform_edt(~gt_s)

    d1 = dt_pred[gt_s].mean() if gt_s.any() else np.nan
    d2 = dt_gt[pred_s].mean() if pred_s.any() else np.nan
    return float(0.5 * (d1 + d2))



# ============================================================
# Backbone
# ============================================================
def build_eval_model(args, device):
    """The frozen Sli2Vol affinity network (the only backbone used in the paper)."""
    use_edge = not args.no_edge_profile
    model = Sli2VolRegistration(
        R=args.R,
        is_training=False,
        use_edge_profile=use_edge,
        edge_radius=args.edge_radius,
        edge_interval=args.edge_interval,
        temperature=args.corr_temperature,
    ).to(device)
    return model


# ============================================================
# Axis permutations, seed rule and distance-weighted soft fusion
# ============================================================
# Axis indices below refer to positions in the canonical (Z, H, W) layout:
#   0 = axial    -> identity
#   1 = coronal  -> bring H to front: (Z,H,W) -> (H,Z,W)
#   2 = sagittal -> bring W to front: (Z,H,W) -> (W,Z,H)
_AXIS_PERMS = {
    0: ((0, 1, 2), (0, 1, 2)),
    1: ((1, 0, 2), (1, 0, 2)),
    2: ((2, 0, 1), (1, 2, 0)),
}
_AXIS_NAMES = {0: "axial", 1: "coronal", 2: "sagittal"}


def _permute_to_axis(vol_zhw, axis):
    fwd, _ = _AXIS_PERMS[axis]
    return np.ascontiguousarray(np.transpose(vol_zhw, fwd))


def _permute_from_axis(vol_perm, axis):
    _, inv = _AXIS_PERMS[axis]
    return np.ascontiguousarray(np.transpose(vol_perm, inv))


def _select_seed(gt_perm):
    """argmax of label area along the leading axis; None if no foreground."""
    areas = gt_perm.reshape(gt_perm.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        return None
    return int(np.argmax(areas))



def fuse_soft(soft_per_axis, mode='soft', sigma=30.0, beta=10.0,
              seeds_canonical=None, shape=None):
    """
    Unsupervised fusion using continuous foreground probabilities per axis.

    Modes:
        'soft'             -> straight average across axes.
        'soft_dist'        -> exponential-decay weighting by through-axis
                              distance from each axis's seed. Each axis votes
                              loudly only where it is reliable (near its own
                              seed). Requires seeds_canonical.
        'soft_consistency' -> weight each axis by agreement with the others
                              (exp of -beta * |pred - mean_others|).

    seeds_canonical: dict {axis_name: int} giving the seed coordinate of each
        axis IN CANONICAL (Z,H,W) SPACE -- axial seed -> Z coord, coronal
        seed -> H coord, sagittal seed -> W coord.
    """
    if not soft_per_axis:
        return np.zeros(shape, dtype=np.int32) if shape is not None else None

    Z, H, W = next(iter(soft_per_axis.values())).shape
    ax_names = list(soft_per_axis.keys())
    stacked = np.stack([soft_per_axis[ax] for ax in ax_names], axis=0)  # (N,Z,H,W)

    if mode == 'soft':
        fused_soft = stacked.mean(axis=0)

    elif mode == 'soft_dist':
        if not seeds_canonical:
            raise ValueError("soft_dist requires seeds_canonical")
        z_grid = np.arange(Z, dtype=np.float32)[:, None, None]
        h_grid = np.arange(H, dtype=np.float32)[None, :, None]
        w_grid = np.arange(W, dtype=np.float32)[None, None, :]
        ax_grid = {'axial': z_grid, 'coronal': h_grid, 'sagittal': w_grid}
        num = np.zeros((Z, H, W), dtype=np.float32)
        den = np.zeros((Z, H, W), dtype=np.float32)
        for ax in ax_names:
            seed = seeds_canonical.get(ax)
            if seed is None:
                continue
            w = np.exp(-np.abs(ax_grid[ax] - float(seed)) / sigma).astype(np.float32)
            num += w * soft_per_axis[ax]
            den += np.broadcast_to(w, (Z, H, W))
        fused_soft = num / np.maximum(den, 1e-6)

    elif mode == 'soft_consistency':
        # Entropy-weighted soft averaging. Each axis is weighted by
        #     w_axis(v) = exp(-beta * H(p_axis(v)))
        # where H(p) = -p log p - (1-p) log(1-p) is the binary entropy of
        # the axis's foreground probability at voxel v. Confident axes
        # (probabilities near 0 or 1, low entropy) get weight near 1;
        # uncertain axes (probabilities near 0.5, high entropy) get
        # downweighted. This rewards confident predictions regardless of
        # whether other axes agree, in contrast to the earlier
        # agreement-based weighting which catastrophically punished
        # correct-but-lonely axes at organ boundaries.
        eps = 1e-7
        clipped = np.clip(stacked, eps, 1.0 - eps)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        weights = np.exp(-beta * entropy).astype(np.float32)
        fused_soft = (weights * stacked).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-6)

    else:
        raise ValueError(f"Unknown soft fusion mode: {mode}")

    return (fused_soft > 0.5).astype(np.int32)



# ============================================================
# NIfTI output (identity affine, array order (Z, H, W) as loaded)
# ============================================================
def save_nifti(vol_zhw, out_path, dtype=np.int16):
    """Save a [Z, H, W] volume as NIfTI."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nii = nib.Nifti1Image(vol_zhw.astype(dtype), np.eye(4))
    nib.save(nii, out_path)


def save_overlap_nifti(gt_zhw, pred_zhw, out_path):
    """
    Save GT+Pred overlap as NIfTI for visualization.

    Labels: 0=background, 1=GT only, 2=Pred only, 3=overlap
    """
    gt_fg = (gt_zhw > 0).astype(np.uint8)
    pr_fg = (pred_zhw > 0).astype(np.uint8)
    overlay = gt_fg + pr_fg * 2  # 0, 1, 2, 3
    save_nifti(overlay, out_path, dtype=np.uint8)



CHECKPOINT_URL = 'https://drive.google.com/drive/folders/1JDVemdII0evcbLa8avdCBw__6V_NoCi_?usp=sharing'


def require_checkpoint(path):
    """Fail with download instructions when a released checkpoint has not been downloaded."""
    if path and not os.path.isfile(path):
        raise SystemExit(f'checkpoint not found: {path}\n'
                         f'Download sli2vol_backbone.pth and readout_learned.pth into checkpoints/ from\n{CHECKPOINT_URL}')
