import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import numpy as np
import pandas as pd
from .su_dataset import SUDataset
from .label_mapper import TASK_SEQUENCES


def get_studies_subset(df, has_AP, has_PA, has_lateral):
    """Get a subset of studies

        Args:
            df:
            has_AP: if true, all studies that are returned will have at least one AP view.
            has_PA: if true, all studies that are returned will have at least one PA view.
            has_lasteral: if true, all studies that are returned will have at least one lateral view.
    """

    bools = [get_AP, get_PA, get_lateral]

    # Get df for AP, PA and lateral
    PA_df = df[df['AP/PA'] == 'PA']
    AP_df = df[df['AP/PA'] == 'AP']
    lateral_df = df[df['Frontal/Lateral'] == 'Lateral']

    # Get unique studies
    all_studies = set(df['Study'].unique())
    PA_studies = set(PA_df['Study'].unique())
    AP_studies = set(AP_df['Study'].unique())
    lateral_studies = set(lateral_df['Study'].unique())

    studies = [AP_studies, PA_studies, lateral_studies]

    for i, study in enumerate(studies):
        if bools[i]:
            all_studies = all_studies.intersection(study)
        else:
            all_studies = all_studies.difference(study)

    return all_studies

def get_type_dist(df):
    """Get distribution of different types of studies"""

    type_dist = []

    for has_AP in [False, True]:
        for has_PA in [False, True]:
            for has_lateral in [False, True]:
                print(f'AP: {has_AP}, PA:{has_PA}, Lateral: {has_lateral}')
                num = len(get_studies(df, has_AP, has_PA, has_lateral))
                study_type_dist.append((has_AP, has_PA, has_lateral, num))


    return pd.DataFrame(study_type_dist, columns=['AP', 'PA', 'Lateral', '#'])

def main():

    write_to_file = False

    df = SUDataset._load_df(data_dir, 'master.csv', split)
    self.studies = df['Study'].drop_duplicates()

    study_type_dist = get_type_dist(df)

    if write_to_file:
        study_type_dist.to_csv('study_type_dist.csv', index=False)

class DataAnalyzer(object):

    def __init__(self, data_dir, split):

        data_dir = Path(data_dir)
        self.df = SUDataset._load_df(data_dir, 'master.csv', split)
        self.studies = SUDataset._get_studies(self.df)

        study_dist = self.get_dist()
        study_dist.to_csv('study_types.csv', index=False)

    def get_dist(self):
        """Get distribution of different types of studies"""

        study_type_dist = []
        for get_AP in [False, True]:
            for get_PA in [False, True]:
                for get_lateral in [False, True]:
                    print(f'AP: {get_AP}, PA:{get_PA}, Lateral: {get_lateral}')
                    num = len(self.get_studies(get_AP, get_PA, get_lateral))
                    study_type_dist.append((get_AP, get_PA, get_lateral, num))

        return pd.DataFrame(study_type_dist, columns=['AP', 'PA', 'Lateral', '#'])


    @staticmethod
    def _multiple(pathology, PA_df, AP_df):

        rel_freq_PA = len(PA_df[PA_df[pathology] == 1]) / len(PA_df)
        rel_freq_AP = len(AP_df[AP_df[pathology] == 1]) / len(AP_df)

        return rel_freq_PA / rel_freq_AP

