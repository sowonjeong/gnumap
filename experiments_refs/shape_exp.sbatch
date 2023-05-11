#!/bin/bash
#SBATCH --job-name=exp_shapes
#SBATCH --output=logs/shapes_array_%A_%a.out
#SBATCH --error=logs/shapes_array_%A_%a.err
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
module load python
source activate pytorch_geom3.9
echo $1
echo $2
PATH2FILE=$SCRATCH/$USER/gnumap/experiments/store/
result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/gnumap/experiments


python3 experiment_shapes.py --noise $1 --n_layers $2 --path $PATH2FILE


