#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=1-5
#SBATCH --time=35:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --account=pi-cdonnat

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID
module load R
source activate pytorch_geom3.9
echo $1
echo $2
METH=$1
result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/gnumap/experiments
if [[ "$METH" == "DGI" ]]; then
  echo "Found it"
  python3 experiment_set_DGI.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
elif [[ "$METH" == "MVGRL" ]]; then
  python3 experiment_set_MVGRL.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
elif [[ "$METH" == "GNUMAP" ]]; then
  python3 experiment_set_GNUMAP.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
elif [[ "$METH" == "GRACE" ]]; then
  python3 experiment_set_GRACE.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
elif [[ "$METH" == "CCA-SSG" ]]; then
  python3 experiment_set_CCA-SSG.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
elif [[ "$METH" == "BGRL" ]]; then
  python3 experiment_set_BGRL.py --dataset $2 --name_file $result_file --seed $SLURM_ARRAY_TASK_ID
fi;
