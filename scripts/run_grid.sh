#!/bin/bash
# Development grid (Tables 1 and 2): every method on SLiver07, MSD pancreas and MSD spleen (342 cases).
#
#   DATA=/path/to/data bash scripts/run_grid.sh                       # all six methods, sequentially
#   DATA=/path/to/data METHODS="contour_learned" bash scripts/run_grid.sh
#   DATA=/path/to/data SHARD=0/4 bash scripts/run_grid.sh             # one quarter of the cases (run 0/4 .. 3/4 in parallel)
#
# One three-seed case takes 70-90 s on a single modern GPU, i.e. roughly 8 GPU-hours per method.
# Results: $OUT/grid/<method>/{per_case.json, niftis/, scores_mm.json}. Then:
#   python make_tables.py --results_dir $OUT --compare reference_results
set -e
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$(dirname "$0")/.."
DATA=${DATA:?set DATA to the folder holding SLiver07/ MSD_Pancreas/ MSD_Spleen/}
OUT=${OUT:-results}
METHODS=${METHODS:-"sli2vol_1seed orthogonal mask_anchors sdf_anchors contour_rule contour_learned"}
SHARD=${SHARD:-}
PY=${PY:-python}
SRC="SLiver07 MSD_Pancreas MSD_Spleen"
for M in $METHODS; do
  ARGS=(); ROOTS=()
  for S in $SRC; do ARGS+=(--data "$S=$DATA/$S"); ROOTS+=(--root "$S=$DATA/$S"); done
  $PY evaluate.py --method "$M" "${ARGS[@]}" ${SHARD:+--shard "$SHARD"} --out_root "$OUT/grid/$M"
  # NSD / ASSD (CPU) once all shards of this method are present
  if [ -z "$SHARD" ] || [ "$($PY -c "import json;print(json.load(open('$OUT/grid/$M/aggregate_stats.json')).get('complete', True))")" = "True" ]; then
    $PY rescore.py "${ROOTS[@]}" --result "$M=$OUT/grid/$M" --out "$OUT/grid/$M/scores_mm.json"
  fi
done
