import numpy as np
import pandas as pd


def load_valid(df):
    rad_csv_path = 'data/valid_rad_majority.csv'
    rad = pd.read_csv(rad_csv_path)
    study_ids = rad['Study #']
    for study_id in study_ids:
        rad_labels = rad.loc[study_ids == study_id, 'No Finding':'Support Devices']
        assert len(rad_labels) == 1, \
            'Found multiple matches in valid_rad_majority.csv for study_id {}'.format(study_id)
        num_images = sum(df['SimpleValidImageID'] == study_id)
        # Each image in the study, should have the same labels
        rad_labels = np.repeat(rad_labels.values, num_images, axis=0)
        df.loc[df['SimpleValidImageID'] == study_id, 'No Finding':'Support Devices'] = rad_labels

    df = df.reset_index(drop=True)

    return df


def main():
    df = pd.read_csv('data/master.csv')
    df = df[df['DataSplit'] == 'valid']
    df = load_valid(df)
    df.to_csv('data/loaded_valid.csv')


if __name__ == '__main__':
    main()
