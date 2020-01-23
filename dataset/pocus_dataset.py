from PIL import Image

import numpy as np
import pandas as pd
import torch

from .base_dataset import BaseDataset
from .label_mapper import TASK_SEQUENCES


class PocusDataset(BaseDataset):

    def __init__(self, data_dir, transform_args, split, is_training,
                 tasks_to, frac, toy=False):
        """Neil's dataset from Cape Town.

        Args:
              data_dir (string): Name of the root data directory.
              transform_args (Namespace): Args for data transforms.
              split (string): Train, test or valid.
              is_training (bool): Indicate whether this is needed for training or not.
              tasks_to (dict): The sequence of tasks.
              frac (float): Fraction of training data to use.
              toy (bool): Indicate if only toy data is needed.
        """

        super().__init__(data_dir, transform_args, split,
                         is_training, 'pocus', tasks_to)

        self.study_level = False
        self.frontal_lateral = False

        # load data from csv
        self.df = self.load_df()

        # TODO(nishit): add code for generating toy dataset

        if frac != 1 and is_training:
            self.df = self.df.sample(frac=frac)
            self.df.reset_index(drop=True)

        self.labels = self.get_disease_labels()
        self.img_paths = self.get_paths()
        self._set_class_weights(self.labels)

    def load_df(self):
        """Load the data from data_dir to a Pandas dataframe"""
        csv_path = self.data_dir / (self.split + "_data.csv")
        df = pd.read_csv(csv_path)
        img_dir = self.data_dir / "images"
        df['Path'] = df['Path'].apply(lambda x: img_dir / x)
        df.reset_index(drop=True)
        # self.df is returned just to maintain clarity in the __init__
        # as to when df was created and altered
        return df

    def get_paths(self):
        """Get list of paths to images.

        Return:
            list: List of paths to images.
        """
        path_list = self.df['Path'].tolist()
        return path_list

    def get_disease_labels(self):
        """Return labels as 3-element arrays for the three diseases.

        Return:
            ndarray: (N * 3) numpy array of labels.
        """
        # construct the label matrix
        num_data_points = len(self.df.index)
        num_labels = len(TASK_SEQUENCES['pocus'])
        labels = np.zeros([num_data_points, num_labels])

        # populate the label matrix
        diseases = [dis for dis in TASK_SEQUENCES['pocus']]
        label_df = self.df[diseases].apply(pd.to_numeric, errors='coerce')
        # remove all NaNs that came up because of the above operation
        label_df.fillna(0.0)
        # TODO: Find out what the problem is because loss is running in NaNs
        for i, disease in enumerate(diseases):
            labels[:, i] = label_df[disease].tolist()

        # Convert to binary labels. 
        labels[labels >= 1] = 1

        return labels

    def __getitem__(self, index):

        label = self.labels[index, :]
        if self.label_mapper is not None:
            label = self.label_mapper.map(label)
        label = torch.FloatTensor(label)

        # Get and transform the image
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform(img)
        
        #Extra empty dict returned to match the number of return variables for TCGA
        return img, label, {}
