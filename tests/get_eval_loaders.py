import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from dataset import get_loader
from dataset import get_eval_loaders
from args import TrainArgParser
from dataset import SUDataset



def test_loader(loader, study_level=True):

    print("len(loader.dataset): ", len(loader.dataset))
    print("len(loader): ", len(loader))
    for i, batch in enumerate(loader):

        if study_level:
            input, target, mask = batch
        else:
            input, target = batch

        if i % 100 == 0:
            print("i: ", i)
            print(f'input.shape: {input.shape}')
            print(f'target.shape: {target.shape}')

        if i == 500:
            break
    print("Loader works for first 1000 examples")
if __name__ == "__main__":

    parser = TrainArgParser()
    args = parser.parse_args()

    eval_loaders = get_eval_loaders(args.eval_su, args.eval_nih, args)

    print("len(eval_loaders): ", len(eval_loaders))
    for loader in eval_loaders:
        test_loader(loader)



