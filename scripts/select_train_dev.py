import pandas as pd

from dataset.constants import COL_SPLIT, COL_STUDY, COL_PATH, COL_PATIENT
from pathlib import Path


def load_df(data_dir, basename):

    path = data_dir / (basename + ".csv")

    return pd.read_csv(path)


def select_train_dev(df, train_dev_size, seed):

    df[COL_STUDY] = df[COL_PATH].apply(lambda p: str(Path(p).parent))
    studies = df[df[COL_SPLIT] == "train"][COL_STUDY].drop_duplicates()

    df[COL_PATIENT] = df[COL_STUDY].apply(lambda p: str(Path(p).parent))
    valid_patients = set(df[df[COL_SPLIT].str.contains("valid")][COL_PATIENT].unique())
    test_patients = set(df[df[COL_SPLIT].str.contains("test")][COL_PATIENT].unique())
    
    train_dev_studies = studies.sample(n=train_dev_size, random_state=seed)

    df.loc[df[COL_STUDY].isin(train_dev_studies), COL_SPLIT] = 'train-dev'

    train_patients = set(df[df[COL_SPLIT.str.contains("train")]][COL_PATIENT].unique())

    assert len(train_patients & (valid_patients | test_patients)) == 0

    return df.drop(columns=[COL_STUDY, COL_PATIENT]), df[COL_STUDY].unique()


def load_and_select(data_dir, csv_basename, train_dev_size, seed):

    df = load_df(data_dir, csv_basename)
    output_path = data_dir / (csv_basename + "_with_train_dev.csv")

    df_with_train_dev, train_dev_studies = select_train_dev(df.copy(), train_dev_size, seed)

    assert df.shape == df_with_train_dev.shape

    df_with_train_dev.to_csv(output_path, index=False)

    return train_dev_studies


def main():
    """Sample a train-development set from the Stanford training set."""
    seed = 0
    data_dir = Path('/deep/group/xray4all/')
    train_dev_size = 2000

    master_with_train_dev_studies = load_and_select(data_dir, "master", train_dev_size, seed)
    master_unpostprocessed_with_train_dev_studies = load_and_select(data_dir, "master_unpostprocessed", train_dev_size, seed)
    nih_with_train_dev_studies = load_and_select(data_dir, "nih_master", train_dev_size, seed)

    assert set(master_with_train_dev_studies) == set(master_unpostprocessed_with_train_dev_studies)
    assert set(master_with_train_dev_studies) == set(nih_with_train_dev_studies)


if __name__ == '__main__':
    main()
