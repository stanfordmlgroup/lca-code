# 2d imaging: train and evaluate multiclass classification models for 2D imaging.

Under Construction.

### Usage

1. **Conda**
   - Activate environment: run `source activate chxr`.

2. **Train**
   - Run `python train.py -h` for usage info.

3. **TensorBoard**
   - While training, launch TensorBoard: `tensorboard --logdir=logs --port=5678`
   - Port forward: `ssh -N -f -L localhost:1234:localhost:5678 <SUNET>@bootcamp`
   - View in browser: `http://localhost:1234/`

4. **Test**
   - Run `python test.py -h` for usage info.

5. **Evaluator**
   - Evaluates model during training and testing.
   - Computes performance metrics such as AUPRC, AUROC, and loss.
   - Generates curves such as Precision-Recall (PR) curve and Receiver Operating Characteristic (ROC) curve.
   - See "https://github.com/stanfordmlgroup/aihc-fall18-evaluator" for details on how to use the evaluator submodule

# Train

This file currently include Stanford chest X-ray and NIH datasets, as well as dataset for HIV positive chest X-ray project and the TCGA dataset for the pathology project. Instructions to train on a specific dataset will be added soon.

## Task Sequences

Tasks define the labels for the different datasets. Different datasets might have different tasks. For instance, the Stanford dataset have 'fracture' whilst NIH does not.

Therefore, you need to specify the task sequence that you want your model to output. An instance of LabelMapper will be called in __getitem__. This take care of converting all labels to the target task sequence. Each dataset will have its own LabelMapper.

You can see all possible sequences in dataset/task_sequences.json. These can be accessed in code by `from dataset import TASK_SEQUENCS`. You can add more task sequences as necessary.

Each model will output labels associated with a task sequence. If you have trained a model using a specific task_sequence then you must use the same label sequence when testing. The outputs of the models are never transformed. What is transformed is the targets that are loaded in the dataset.


# Test trained model

Instructions to test a model will be added soon.


# Entities
- `logger`: module for logging information during training and testing. The logger allows the saving of logging information to a log file, saving summary writers for TensorFlow, and printing to stdout.
    - BaseLogger: a base class that is used by TrainLogger and TestLogger. 
    - TrainLogger: logs information for a given epoch and iteration during the training process.
    - TestLogger: logs information for a given epoch and iteration during the testing process.
- `args`: defines arguments that set the configuration of training and testing. Consists of:
    - BaseArgParser: used for based arguments that are shared between the train and test mode.
    - TrainArgParser: arguments that are used only in train mode.
    - TestArgParser: arguments that are used only in test mode.
- `saver`: allows saving a trained model and loading it in a later stage.
- `dataset`: defines dataset loaders, which are iterators of specific datasets with label mappings defined by the task given.
    - Base_dataset.py: a general base class that is used by each specific dataset loader class. Defines general parameters such as dataset directory, sequence of tasks, data transformations, and train/validation split of the data.
    - Get_loader.py: Defines a function that returns a specific dataset loader for either the training set or the validation set.
    - Task_sequences.json: list of available task sequences (or label mappings).
    - <dataset_name>_dataset.py: a customized class for each dataset.
- `train.py`: runs the training process based on the args defined in args/train_arg_parser.py. The file consists of the following steps:
    - Load model: gets the model, either from an existing checkpoint or from a model from models/models.py.
    - Load optimizer and learning rate scheduler, which will be used for the training process.
    - Get data loaders and class weights: loads the train and validation datasets, which will be used during training. The class weights are used for weighted loss function.
    - Get loss functions: get cross entropy loss and weighted loss functions.
    - Get logger, evaluator, and saver: see description of each module above.
    - Run training loop, evaluate and save the model periodically.
- `test.py`: performs evaluation of a trained model on test data and write the results to disk.

# Definitions
- `task`: in the competition we have 5 binary tasks. In the NIH dataset there is a total of 14 tasks. When chatting we sometimes call this a category. But in code, we only use task.
- `task_sequence`: refers to an ordered set of tasks. Each dataset, has exactly one task sequence. The Stanford competition has its own task sequence (of 5 tasks).
- `label`: a label refer to groundtruth for an example. Usually, the label is an array of binary values. The label has an ordering that corresponds to a task sequence. Labels and targets are the same thing.
- `dataset/task_sequences.json`: we list the task sequences that we use.
- `LabelMapper`: class that lets you map a label with one task ordering, to another task ordering. E.g you have a label from the Stanford dataset, and you want to convert that label to be on the NIH format.
- `class`: each task has two classes, positive and negative.
- `class_weights`: the weights for each class for each task.

