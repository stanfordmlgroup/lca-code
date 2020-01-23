#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="run"
#SBATCH --output=runs/train-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample job
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

python train.py

# done
echo "Done"
