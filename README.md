# Anchored Slice Propagation

Code and models for **Anchored slice propagation and a label-free contour readout for 3D CT segmentation from three
annotated slices** (submitted to ISBI 2027). [Paper](https://arxiv.org/abs/2611.05693)

One axial, one coronal and one sagittal slice are annotated. Each is propagated through a frozen Sli2Vol affinity,
constrained by the other two planes (anchors), and the three results are fused. The default tracker moves the object's
contour with a small readout network trained without any manual labels.

## Installation

```bash
conda create -n asp python=3.9 -y && conda activate asp
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the two model files from
[Google Drive](https://drive.google.com/drive/folders/1JDVemdII0evcbLa8avdCBw__6V_NoCi_?usp=sharing) into
`checkpoints/` (`sli2vol_backbone.pth` and `readout_learned.pth`), then check the install:

```bash
python -m pytest tests -q        # ~5 min on CPU, no data needed; also verifies the checkpoint checksums
```

## Segment your own scan

Draw the object on one axial, one coronal and one sagittal slice (e.g. in ITK-SNAP or 3D Slicer, ideally where it is
largest) and save the three slices as one label file on the scan's grid. Then:

```bash
python segment.py --image scan.nii.gz --annotation three_slices.nii.gz --out mask.nii.gz
```

The mask is written in the scan's own geometry. Use `--method mask_anchors` for branching structures such as vessels.
With a fully labelled scan, `--simulate_from_label label.nii.gz` replaces `--annotation` and uses the label's three
largest cross-sections, as in the paper.

## Reproducing the paper

**1. Data.** Download the public datasets and build the test folders, then check them against the paper's:

| Folder | Source | Command |
|---|---|---|
| `SLiver07` | [SLIVER07](https://sliver07.grand-challenge.org/) training set | `python prepare_data.py sliver07 --scans training-scans.zip --labels training-labels.zip --out $DATA` |
| `MSD_Pancreas` | [MSD](http://medicaldecathlon.com/) Task07 (pancreas + tumour) | `python prepare_data.py msd --task_dir Task07_Pancreas --name MSD_Pancreas --out $DATA` |
| `MSD_Spleen` | MSD Task09 | `python prepare_data.py msd --task_dir Task09_Spleen --name MSD_Spleen --out $DATA` |
| `AMOSva_<organ>` | [AMOS22](https://amos22.grand-challenge.org/) validation set, CT | `python prepare_data.py amos --amos_dir amos22 --out $DATA` |
| `HV_vessel`, `HV_tumour` | MSD Task08 | `python prepare_data.py hepatic_vessel --task_dir Task08_HepaticVessel --out $DATA` |

```bash
python prepare_data.py check --data_root $DATA     # should end with "all folders match"
```

**2. Smoke test.** Checks a few cases against the paper's per-case numbers and ends with `PASS`:

```bash
python smoke_test/smoke_test.py --data_root $DATA --level quick   # ~20 min on one GPU
python smoke_test/smoke_test.py --data_root $DATA --level full    # ~100 min, all six methods
```

**3. Full tables.** Each paper row is one `--method`: `sli2vol_1seed`, `orthogonal`, `mask_anchors`, `sdf_anchors`,
`contour_rule`, `contour_learned`.

```bash
DATA=$DATA bash scripts/run_grid.sh      # Tables 1-2 (~8 GPU-hours per method; SHARD=k/n to split)
DATA=$DATA bash scripts/run_cohort.sh    # Table 3   (~45 GPU-hours per method)
python make_tables.py --results_dir results --compare reference_results
```

`python make_tables.py --grid reference_results/grid.json --cohort reference_results/cohort.json` prints the paper's
tables from the shipped per-case results. Small GPU-dependent differences (about 0.001 Dice) are expected.

**Retraining the readout** (optional): see `scripts/train_readout.sh` and `checkpoints/readout_learned_train_config.json`.

## Checkpoints

* `checkpoints/sli2vol_backbone.pth`: frozen Sli2Vol backbone (self-supervised, no labels).
* `checkpoints/readout_learned.pth`: learned contour readout (trained on pseudo-objects of unlabelled CT, no labels).

Neither model was trained on any of the test datasets.

## License

MIT (see `LICENSE`). `models/sli2vol_original.py` adapts the Sli2Vol network of Yeung et al., MICCAI 2021.
The checkpoints are released for research use.

## Citation

Will be added once the paper is published.

## Acknowledgements

The Sli2Vol backbone, orthogonal seeding and fusion come from Haque et al., *Inference-time orthogonal seeding*,
MLMI 2026, developed by Md Rakibul Haque:
[github.com/RakibulHaqueSajal/SlicePropagation](https://github.com/RakibulHaqueSajal/SlicePropagation).
