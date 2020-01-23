#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# only use the following on partition with GPUs
#SBATCH --gres=gpu:4
#SBATCH --job-name="train msi model"
#SBATCH --output=sample-%j.out
# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR



# Retrives the absoloute path to your project folder
PROJECT_FOLDER="$(dirname "$PWD")"
DATADIR="/deep/group/data/hpylori/WSIs/S18-19049/"
                                                                                                              X
# Output folders
SAVE_DIR="${PROJECT_FOLDER}/ckpts"
NAME="debugging"
BATCH_SIZE="8"

ITERS_PER_PRINT="30"
ITERS_PER_VISUAL="50"

LR_DECAY_STEP="20"
LR_DECAY_GAMMA="0.50"

NUM_EPOCHS="300"
NUM_VISUALS="10"
MAX_EVAL="10"
EPOCHS_PER_EVAL="10"
LEARNING_RATE="0.0002"
LOSS_FN="hybrid"
FOCAL_GAMMA="5"
MODEL='UNet'


ARGUMENTS="--datadir $DATADIR --save_dir $SAVE_DIR --model $MODEL --num_visuals $NUM_VISUALS --max_eval $MAX_EVAL --epochs_per_eval $EPOCHS_PER_EVAL --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --name $NAME --batch_size $BATCH_SIZE --iters_per_print $ITERS_PER_PRINT --iters_per_visual $ITERS_PER_VISUAL --lr_decay_step $LR_DECAY_STEP --lr_decay_gamma $LR_DECAY_GAMMA"

echo ${ARGUMENTS}

cd ..
srun --nodes=${SLURM_NNODES} python train.py $ARGUMENTS
####echo NPROCS=$NPROC
# done
echo "Done"
