#!/bin/bash
# Example SLURM wrapper: the development grid as 4 shards x 6 methods (24 jobs, one GPU each).
#   DATA=/path/to/data bash scripts/slurm_example.sh
# Edit the #SBATCH-style options below for your cluster.
cd "$(dirname "$0")/.."
DATA=${DATA:?}
for M in sli2vol_1seed orthogonal mask_anchors sdf_anchors contour_rule contour_learned; do
  for K in 0 1 2 3; do
    sbatch --gres=gpu:1 --cpus-per-task=8 --mem=24G --time=12:00:00 --job-name="asp-$M-$K" \
      --wrap="DATA=$DATA METHODS=$M SHARD=$K/4 bash scripts/run_grid.sh"
  done
done
