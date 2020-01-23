import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse

from dataset import get_loader
from args import TrainArgParser
from dataset import SUDataset
from dataset import TASK_SEQUENCES




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

    print("Loader works for first 1000 examples")


def test_dataset_subset(args):

    for subset in ['PA', 'AP', 'lateral']:
        su_dataset_train = SUDataset(
                    args.SU_data_dir,
                    args, split='train',
                    tasks_to=args.task_sequence,
                    is_training=True,
                    study_level=False,
                    subset=subset
                    )

    print("len(su_dataset_train): ", len(su_dataset_train))


def run_loaders(args):

    for split in ['train', 'valid']:
        # Stanford loader
        print(f'Testing Stanford {split}  loader')
        loader_SU = get_loader(stanford_frac=1,
                               nih_frac=0,
                               split=split,
                               shuffle=True,
                               args=args)

        test_loader(loader_SU)

        # Stanford loader STUDY
        print(f'Testing Stanford {split}  loader')
        loader_SU_study = get_loader(stanford_frac=1,
                               nih_frac=0,
                               split=split,
                               shuffle=True,
                               args=args,
                               study_level=True)

        test_loader(loader_SU_study)
        # NIH loader
        print(f'Testing NIH {split} loader')
        loader_NIH = get_loader(stanford_frac=0,
                               nih_frac=1,
                               split=split,
                               shuffle=True,
                               args=args)

        #test_loader(loader_NIH)
        # NIH & Stanford combined loaders
        print(f'Testing NIH & SU combo {split} loader')
        loader_combo = get_loader(stanford_frac=1,
                                   nih_frac=1,
                                   split=split,
                                   shuffle=True,
                                   args=args)
        test_loader(loader_combo)


def test_dataset(args):

    su_dataset_train = SUDataset(
                args.SU_data_dir,
                args, split='train',
                tasks_to=args.task_sequence,
                csv_name=args.SU_csv_name,
                is_training=True,
                study_level=False,
                )


    img, label = su_dataset_train[0]

    print("len(su_dataset_train): ", len(su_dataset_train))
if __name__ == "__main__":

    parser = TrainArgParser()

    args = parser.parse_args()
    data_args = args.data_args
    transform_args = args.transform_args
    task_sequence = TASK_SEQUENCES[data_args.task_sequence]

    #test_dataset(args)
    #print(f'Testing Stanford {split}  loader')
    train_loader = get_loader(data_args,
        transform_args, 'train', task_sequence,
        data_args.su_train_frac,
        data_args.nih_train_frac,
        args.batch_size,
        is_training=False, shuffle=False,
        study_level=True)


    test_loader(train_loader)
