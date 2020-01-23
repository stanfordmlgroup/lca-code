from args import TrainArgParser
import util
def train(args):
    """Run training loop with the given args.

    The function consists of the following steps:
        1. Load model: gets the model from a checkpoint or from models/models.py.
        2. Load optimizer and learning rate scheduler.
        3. Get data loaders and class weights.
        4. Get loss functions: cross entropy loss and weighted loss functions.
        5. Get logger, evaluator, and saver.
        6. Run training loop, evaluate and save model periodically.
    """

    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args
    
    print(args)

if __name__ == '__main__':
    parser = TrainArgParser()
    args = util.get_auto_args(parser)
    train(args)
    #train(parser.parse_args())
