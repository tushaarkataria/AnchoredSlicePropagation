"""
Test-set loader.

Reproduces exactly what the paper's evaluation saw, without the training-set
bookkeeping of the original multi-source loader:

  * a dataset root holds  imagesTr/<case>.nii.gz  and  labelsTr/<case>.nii.gz;
    every image with a label of the same file name is a test case;
  * cases are visited source by source, in the order the sources are given,
    and within a source in sorted file-name order (hidden files such as
    macOS '._*' are ignored, as glob does);
  * each case is loaded with MONAI, reoriented to RAS, intensities clipped to
    [-300, 400] HU and scaled to [0, 1]; the label is binarised (label > 0, so
    e.g. MSD pancreas = pancreas + tumour);
  * arrays are returned as (Z, H, W) with Z the last NIfTI axis after RAS
    reorientation, together with the first/last slice that holds foreground.

The yielded tuples have the same layout as a batch of size 1 from the
original DataLoader, so evaluate.py consumes them unchanged.
"""
import glob
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, EnsureTyped

# identical to base_img_lab_tf of the original loader
BASE_IMG_LAB_TF = Compose([
    LoadImaged(keys=["image", "label_full"]),
    EnsureChannelFirstd(keys=["image", "label_full"]),
    Orientationd(keys=["image", "label_full"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-300, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label_full"]),
])


def stem_niigz(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def list_cases(sources: List[Tuple[str, str]]) -> List[dict]:
    """[(source_name, root), ...] -> list of {image, label_full, source, case_id} in evaluation order."""
    items = []
    for name, root in sources:
        imgs = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
        if not imgs:
            raise RuntimeError(f"[{name}] no images in {root}/imagesTr")
        n0 = len(items)
        for ip in imgs:
            lp = os.path.join(root, "labelsTr", os.path.basename(ip))
            if os.path.exists(lp):
                items.append({"image": ip, "label_full": lp, "source": name, "case_id": stem_niigz(ip)})
        print(f"[data] {name}: {len(items) - n0} labelled cases in {root}", flush=True)
    return items


def load_case(item: dict):
    """One test case in the original batch-of-one layout, or None if the label is empty."""
    d = BASE_IMG_LAB_TF({"image": item["image"], "label_full": item["label_full"]})
    img_np = d["image"].squeeze(0).cpu().numpy().astype(np.float32)            # [H, W, Z]
    lab_np = (d["label_full"].squeeze(0).cpu().numpy() > 0).astype(np.uint8)  # [H, W, Z]
    areas = lab_np.sum(axis=(0, 1))
    if areas.max() == 0:
        return None
    pos = np.nonzero(areas > 0)[0]
    z_start, z_end = int(pos[0]), int(pos[-1])
    img_z = np.moveaxis(img_np, -1, 0)
    lab_z = np.moveaxis(lab_np, -1, 0)
    return (torch.from_numpy(img_z)[None], torch.from_numpy(lab_z)[None], None, None,
            torch.tensor([z_start]), torch.tensor([z_end]), [item["case_id"]], [item["source"]])


def iterate_cases(sources: List[Tuple[str, str]], only: Optional[List[str]] = None) -> Iterator:
    """Yields load_case() for every case; `only` restricts to these case ids (others yield None
    so that the case index used by --limit_cases / --shard is unchanged)."""
    for it in list_cases(sources):
        if only and it["case_id"] not in only:
            yield None
            continue
        yield load_case(it)
