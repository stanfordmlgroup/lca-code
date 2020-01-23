
# Script that runs experiments in experiments.sh
# by creating multiple screens and running the train script

# fetch the argments list and the names list
# from experiments.sh
. bash_scripts/experiments.sh

# Currently only works for Henrik's env
ENV_CMD="source /sailhome/marklund/miniconda3/bin/activate msi-env2"

num_experiments=${#EXPERIMENTS_ARGS[@]}
echo ${num_experiments}
for ((i = 0; i < $num_experiments; ++i));
do

    EXPERIMENT_NAME=${EXPERIMENTS_NAMES[$i]}
    echo ${EXPERIMENT_NAME}

    COMMANDS="
        echo testing;
        $ENV_CMD
        python train.py ${EXPERIMENTS_ARGS[$i]}
        exec bash;
    "

    echo $COMMANDS
    screen -mdS ${EXPERIMENT_NAME} bash -c "${COMMANDS}"
done
