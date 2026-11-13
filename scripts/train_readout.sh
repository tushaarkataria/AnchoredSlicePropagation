#!/bin/bash
# Re-train the learned contour readout from scratch (optional: the trained checkpoint is in checkpoints/).
#
# 1) supervoxel pseudo-objects from UNLABELLED CTs (CPU, ~3 GB RAM per worker). The paper used 506 scans from
#    CHAOS (CT), NIH Pancreas-CT, C4KC-KiTS and CT Lymph Nodes, each as <root>/imagesTr/*.nii.gz:
#      PSEUDO=/path/pseudo SOURCES="CHAOS=/d/chaos NIHPancreas=/d/pancreas_ct KiTS=/d/kits CTLN=/d/ctln" bash scripts/train_readout.sh pseudo
# 2) train (one GPU, ~10 h for 3000 steps):
#      PSEUDO=/path/pseudo OUTDIR=checkpoints/readout_retrained bash scripts/train_readout.sh train
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
PSEUDO=${PSEUDO:?set PSEUDO to the pseudo-object folder}
case "${1:-}" in
  pseudo)
    ARGS=(); for s in ${SOURCES:?set SOURCES=\"NAME=ROOT ...\"}; do ARGS+=(--source "$s"); done
    $PY make_pseudo_objects.py "${ARGS[@]}" --out "$PSEUDO" --workers "${WORKERS:-8}" --inplane 256 --n_segments 400,1500 ;;
  train)
    $PY train_readout.py --ckpt checkpoints/sli2vol_backbone.pth --pseudo "$PSEUDO" --out "${OUTDIR:-checkpoints/readout_retrained}" \
      --steps 3000 --samples_per_step 4 --K_max 6 --K_curriculum 2:0.2,4:0.5 --lr 1e-3 \
      --val_every 200 --n_val 48 --R 6 --edge_radius 3 --edge_interval 1 --corr_temperature 0.7 --seed 0 ;;
  *) echo "usage: $0 pseudo|train"; exit 1 ;;
esac
