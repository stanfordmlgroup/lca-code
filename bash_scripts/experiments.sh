# This file lets you specify arguments for each experiment you want to run
# This experiments are then executed by
# 1) starting screen
# 2) opening your environment
# 3) . bash_scripts/multi_exp_tmux.sh

#  . bash_scripts/multi_exp_screen.sh

# Arrrays for experiment args and experiment names
EXPERIMENTS_ARGS=()
EXPERIMENTS_NAMES=()


### Parameters common to all experiments
PROJECT_FOLDER="$PWD"
DATADIR="/deep/group/data/hpylori/WSIs/S18-19049/"
# Output folders
SAVE_DIR="${PROJECT_FOLDER}/ckpts"

# Evaluation parameters
ITERS_PER_PRINT="60"
ITERS_PER_VISUAL="30"
NUM_VISUALS="10"
MAX_EVAL="100"
EPOCHS_PER_EVAL="1"

# Concatenate all these arguments into one string
COMMON_ARGUMENTS="
    --num_visuals $NUM_VISUALS --max_eval $MAX_EVAL
    --epochs_per_eval $EPOCHS_PER_EVAL --iters_per_print $ITERS_PER_PRINT
    --iters_per_visual $ITERS_PER_VISUAL --datadir $DATADIR
    --save_dir $SAVE_DIR"

# Below you create a name and list of arguments for each experiment you want to run
# This this experiments will be run when you run batch_experiments.sh


### Experiment 1
NAME="dice_lr_e-4"
ARGUMENTS="
    --name $NAME
    --model UNet
    --learning_rate 0.0001
    --lr_decay_step 100
    --lr_decay_gamma 0.5

    --gpu_ids 0
    --loss_fn dice
    --focal_gamma 2
    --weighted_loss 0
    --weight_vals 0.5048 53.1099
    --batch_size 2
    --task segmentation
    --num_epochs 100
    $COMMON_ARGUMENTS
"


EXPERIMENTS_ARGS+=("${ARGUMENTS}")
EXPERIMENTS_NAMES+=($NAME)


### Experiment 2
NAME="dice_lr_e-5"
ARGUMENTS="
    --name $NAME
    --model UNet
    --learning_rate 0.0001
    --lr_decay_step 100
    --lr_decay_gamma 0.5

    --gpu_ids 1
    --loss_fn dice
    --focal_gamma 2
    --weighted_loss 0
    --weight_vals 0.5048 53.1099
    --batch_size 2
    --task segmentation
    --num_epochs 100
    $COMMON_ARGUMENTS
"


EXPERIMENTS_ARGS+=("${ARGUMENTS}")
EXPERIMENTS_NAMES+=($NAME)

### Experiment 3
NAME="dice_lr_e-6"
ARGUMENTS="
    --name $NAME
    --model UNet
    --learning_rate 0.0001
    --lr_decay_step 100
    --lr_decay_gamma 0.5

    --gpu_ids 2
    --loss_fn dice
    --focal_gamma 2
    --weighted_loss 0
    --weight_vals 0.0094 0.9906
    --batch_size 2
    --task segmentation
    --num_epochs 100
    $COMMON_ARGUMENTS
"


EXPERIMENTS_ARGS+=("${ARGUMENTS}")
EXPERIMENTS_NAMES+=($NAME)


# The experiment below is commented out
# to give an example of how to comment

: '
### Experiment 3

NAME="bce_weighted_rel_freq_lr_e-3"

ARGUMENTS="
    --name $NAME
    --model UNet
    --learning_rate 0.0001
    --lr_decay_step 100
    --lr_decay_gamma 0.5

    --gpu_ids 0
    --loss_fn cross_entropy
    --focal_gamma 2
    --weighted_loss 1
    --weight_vals $WEIGHT_VALS

    --batch_size 2
    --task segmentation
    --num_epochs 100
    $COMMON_ARGUMENTS
"


EXPERIMENTS_ARGS+=($ARGUMENTS)
EXPERIMENTS_NAMES+=($NAME)


'
