# Usage
# start screen
# get interative gpus by running . bash_scripts/multiple_gpus.sh


# fetch the argments list and the names list
# from experiments.sh
. bash_scripts/experiments.sh

num_experiments=${#EXPERIMENTS_ARGS[@]}
echo ${num_experiments}
for ((i = 0; i < $num_experiments; ++i));
do
    COMMANDS="python train.py ${EXPERIMENTS_ARGS[$i]}"
    EXPERIMENT_NAME=${EXPERIMENTS_NAMES[$i]}
    echo $EXPERIMENT_NAME

    # Tmux send command treats deletes all " ".
    # therefore all " " needs to be repacled with "Space"
    COMMANDS=${COMMANDS//" "/" Space "}

    tmux new-session -d -s ${EXPERIMENT_NAME}
    tmux send -t ${EXPERIMENT_NAME} ${COMMANDS}
    tmux send-keys -t ${EXPERIMENT_NAME} ENTER
done

