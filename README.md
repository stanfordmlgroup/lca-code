## Paper Statistics Scripts
Scripts used for data processing and figure generation for our [manuscript](https://doi.org/10.1038/s41746-020-0232-8) is available [here](https://github.com/stanfordmlgroup/lca-code/tree/master/paper).



# LiverCancerAssistant Deep Learning Model 

### Usage

1. **Conda**
   - Use the `environment.yml` file to create the conda environment.
   - Activate the conda environment.

2. **Train**
   - Run `python train.py -h` for usage info.

3. **TensorBoard**
   - While training, launch TensorBoard: `tensorboard --logdir=logs --port=5678`
   - Port forward: `ssh -N -f -L localhost:1234:localhost:5678 <SUNET>@bootcamp`
   - View in browser: `http://localhost:1234/`

4. **Test**
   - Run `python test.py -h` for usage info.

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
