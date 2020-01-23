import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import models
from args import TrainArgParser
from dataset import TASK_SEQUENCES


def main(args):
    """File used for us to manually look up what the last layer
     before the global average pooling is"""

    model_args = args.model_args
    task_sequence = TASK_SEQUENCES[args.data_args.task_sequence]

    #model_fn = models.__dict__['Inceptionv4']
    model_fn = models.__dict__['ResNet152']
    model = model_fn(task_sequence, model_args)

    print(model.model)

if __name__ == '__main__':
    parser = TrainArgParser()
    args = parser.parse_args()
    main(args)
