#!/bin/bash
# Held-out cohort (Table 3): every method on the 17 structures (2091 cases) prepared by prepare_data.py.
#
#   DATA=/path/to/data bash scripts/run_cohort.sh
#   DATA=/path/to/data STRUCTURES="AMOSva_pancreas HV_vessel" METHODS="mask_anchors contour_learned" bash scripts/run_cohort.sh
#   DATA=/path/to/data SHARD=0/8 bash scripts/run_cohort.sh            # split every structure into 8 parallel jobs
#
# Roughly 45 GPU-hours per method for the full cohort. Results: $OUT/cohort/<structure>/<method>/per_case.json
#   python make_tables.py --results_dir $OUT --compare reference_results
set -e
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$(dirname "$0")/.."
DATA=${DATA:?set DATA to the folder holding the AMOSva_* and HV_* folders}
OUT=${OUT:-results}
METHODS=${METHODS:-"sli2vol_1seed orthogonal mask_anchors sdf_anchors contour_rule contour_learned"}
STRUCTURES=${STRUCTURES:-"AMOSva_liver AMOSva_spleen AMOSva_left_kidney AMOSva_right_kidney AMOSva_stomach AMOSva_bladder AMOSva_prostate_uterus AMOSva_pancreas AMOSva_gall_bladder AMOSva_duodenum AMOSva_esophagus AMOSva_arota AMOSva_postcava AMOSva_left_adrenal_gland AMOSva_right_adrenal_gland HV_tumour HV_vessel"}
SHARD=${SHARD:-}
PY=${PY:-python}
for S in $STRUCTURES; do
  for M in $METHODS; do
    $PY evaluate.py --method "$M" --data "$S=$DATA/$S" ${SHARD:+--shard "$SHARD"} --out_root "$OUT/cohort/$S/$M"
  done
done
