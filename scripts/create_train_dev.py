import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import pandas as pd

from dataset import SUDataset

def main():
    """Create an extra validation set from the Stanford train set"""

    data_dir = Path('/deep/group/xray4all/')
    seed = 0

    train_dev_size = 1000
    output_csv = data_dir / 'train_dev.csv'

    df = SUDataset._load_df(data_dir, 'master.csv', 'train')

    df['study'] = df['Path'].apply(lambda p: str(p.parent))
    studies = df['study'].drop_duplicates()

    train_dev_studies = studies.sample(n=train_dev_size, random_state=seed)

    train_dev_studies.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
