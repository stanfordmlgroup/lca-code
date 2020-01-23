"""Dataset for whole slide images from TCGA data"""
from pathlib import Path
import os
import h5py
import numpy as np
import os
import pickle
import random
import torch.utils.data as data
import torch
import pandas as pd
from tqdm import tqdm
import util
import staintools
import torchvision.transforms as transforms
from PIL import Image

from .base_dataset import BaseDataset
from .label_mapper import TASK_SEQUENCES
from .constants import COL_TCGA_SLIDE_ID, COL_TCGA_FILE_ID,\
        COL_TCGA_FILE_NAME, COL_TCGA_CASE_ID, COL_TCGA_LABEL,\
        COL_TCGA_PATCH_ID, COL_TCGA_NUM_PATCHES, COL_TCGA_INDICES,\
        COL_TCGA_PATH, COL_TCGA_ENTITIES, SLIDE_METADATA_FILE,\
        SLIDE_PKL_FILE, DEFAULT_PATCH_SIZE, TCGA_MEAN, TCGA_STD 

# TODO: implement flag to turn dataset into toy dataset
class TCGA_Google_Dataset(BaseDataset):
    """Dataset for TCGA classification."""

    def __init__(self, data_path, transform_args, metadata_csv,
                 split='test', num_classes=2,
                 resize_shape=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
                 max_patches=None, tasks_to='tcga',
                 is_training=False, filtered=True, toy=False, 
                 normalize=False, transform=None):
        """Initialize TCGADataset.

        data directory to be organized as follows:
            data_path
                XXX.jpg
                XXX.png
                ...
                XXX.jpg
                metadata_dataset_name.csv

        Args:
            data_path (str): path to data directory
            transform_args (args): arguments to transform data
            metadata_csv (str): path to csv containing metadata information of the dataset
            split (str): either "train", "valid", or "test"
            num_classes (int): number of unique labels
            resize_shape (tuple): shape to resize the inputs to
            max_patches (int): max number of patches to obtain for each slide
            tasks_to (str): corresponds to a task sequence
            is_training (bool): whether the model in in training mode or not
            filtered (bool): whether to filter the images
        """
        #if split not in ["train", "valid", "test"]:
            #raise ValueError("Invalid value for split. Must specify train, valid, or test.")

        super().__init__(data_path, transform_args,
                         split, is_training, 'tcga', tasks_to)
        print("split:" + split)

        self.data_path = data_path

        #self.hdf5_path = os.path.join(self.data_path, "{}.hdf5".format(split))
        #self.hdf5_fh = h5py.File(self.hdf5_path, "r")

        self.split = split
        self.is_training = is_training
        self.dataset_name = 'tcga_google'
        print("dataset: " + self.data_path.split("/")[-2])
        self.metadata_path = os.path.join(self.data_dir, "metadata " + self.data_path.split("/")[-2] + ".csv")
#        print(self.data_dir)
        self.metadata = pd.read_csv(self.metadata_path)
        self.toy = True
        self.filtered = filtered 
        self.num_classes = num_classes
        
        print("number of patches in this test set:" + str(len(self.metadata)))
        print(self.metadata["label"].value_counts())

        self.label_dict = self._get_label_dict(tasks_to)

        #self._set_class_weights(self.labels)
        self.normalize = normalize

        # tools for patch normalization
        self.normalizer_with_constants = transforms.Compose([transforms.Normalize(mean = TCGA_MEAN, std = TCGA_STD)])
        self.ToTensor = transforms.Compose([transforms.ToTensor()])

 
    def __len__(self):
        return len(self.metadata)

    def _get_label_dict(self, tasks_to):
        """Return appropriate label dict for task"""
        return tasks_to

    def _label_conversion(self, label):
        """Turn string label into integer"""
        if label == 'Cholangio':
            label = 'CHOL'
        if label not in self.label_dict:
            raise ValueError("Invalid label: {} entered".format(label))
        return self.label_dict[label]

    def transform_label(self, label):
        """Make label correct shape"""
        label = np.array(label).astype(np.float32).reshape(1)
        return label
    
    def get_patch(self, patch_name):
        if patch_name.split(".")[-1] == 'png':
            patch = Image.open(patch_name).convert('RGB')
        else:
            patch = Image.open(patch_name)
        return patch
  
    def normalize_patch_with_constants(self, patch):
        """ Normalize using pre-calculated data-specific constants"""
        patch = self.normalizer_with_constants(patch)
        return patch        
        
      
    def __getitem__(self, idx):
        """Return element of dataset"""
        patch_name = os.path.join(self.data_path, self.metadata.loc[idx, "file_name"])
        label = self.metadata.loc[idx, "label"]
        label = self._label_conversion(label)
        label = self.transform_label(label)
        label = torch.tensor(label, dtype=torch.float32)
        patch = self.get_patch(patch_name)
        # reize and convert to Tensor
        patch = transforms.Compose([transforms.Resize([DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE]), transforms.ToTensor()])(patch)

        if self.normalize:
            patch = self.normalize_patch_with_constants(patch)
        
        patch = patch.numpy()
        patch = patch.astype(np.float32)
        info_dict = {'patch_name': self.metadata.loc[idx, "file_name"]} 
        return patch, label, info_dict

