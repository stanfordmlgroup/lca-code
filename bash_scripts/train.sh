
# Retrives the absoloute path to your project folder
PROJECT_FOLDER="$PWD"

# Output folders
NAME="debugging"

# Training parameters
GPU_IDS="0"

TOY="0"

ARGUMENTS="
    --name nih-DN121-
    --gpu_ids 0

    --nih_train_frac 1
    --su_train_frac 0
    --eval_nih 1
    --eval_su 0

    --scale 224
    --maintain_ratio 1

    --model DenseNet121
    --lr 1e-3
    --clahe 0
    --toy 0"

echo ${ARGUMENTS}
python train.py ${ARGUMENTS}
