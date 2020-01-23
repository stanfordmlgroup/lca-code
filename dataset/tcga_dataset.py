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

from scipy.misc import imsave

from .base_dataset import BaseDataset
from .label_mapper import TASK_SEQUENCES
from .constants import COL_TCGA_SLIDE_ID, COL_TCGA_FILE_ID,\
        COL_TCGA_FILE_NAME, COL_TCGA_CASE_ID, COL_TCGA_LABEL,\
        COL_TCGA_PATCH_ID, COL_TCGA_NUM_PATCHES, COL_TCGA_INDICES,\
        COL_TCGA_PATH, COL_TCGA_ENTITIES, SLIDE_METADATA_FILE,\
        SLIDE_PKL_FILE, DEFAULT_PATCH_SIZE, TCGA_MEAN, TCGA_STD 

# TODO: implement flag to turn dataset into toy dataset
class TCGADataset(BaseDataset):
    """Dataset for TCGA classification."""

    def __init__(self, data_path, transform_args, metadata_csv,
                 split, num_classes=2,
                 resize_shape=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
                 max_patches=None, tasks_to='tcga',
                 is_training=False, filtered=True, toy=False, 
                 normalize=False, transform=None):
        """Initialize TCGADataset.

        data directory to be organized as follows:
            data_path
                slide_list.pkl
                train.hdf5
                val.hdf5
                test.hdf5
                metadata.csv

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
#        if split not in ["train", "valid", "test"]:
#            raise ValueError("Invalid value for split. Must specify train, valid, or test.")

        super().__init__(data_path, transform_args,
                         split, is_training, 'tcga', tasks_to)
        self.data_path = data_path
#        self.slide_list_path = os.path.join(self.data_path, SLIDE_PKL_FILE) 
        self.hdf5_path = os.path.join(self.data_path, "{}.hdf5".format(split))

        #hdf5_fh = h5py.File(self.hdf5_path, "r")
        #if split == "demo":
        #    s = "TCGA-W5-AA2Z-01Z-00-DX1.49AB7E33-EE0C-42DE-9EDE-91E01290BE45.svs"
        #    print("hdf5 test!")
        #    print("slide: {}".format(s))
        #    print("patch 0: {}".format(self.hdf5_fh[s][0, 0, 0, 0]))
        #    print("patch 1: {}".format(self.hdf5_fh[s][1, 0, 0, 0]))

        self.split = split
        self.is_training = is_training
        self.metadata_path = os.path.join(self.data_dir, metadata_csv)
        print("metadata_path: {}".format(self.metadata_path))
        self.metadata = pd.read_csv(self.metadata_path)
        print("hdf5 path: {}".format(self.hdf5_path))

        self.toy = True

        self.filtered = filtered 
#        with open(self.slide_list_path, "rb") as pkl_fh:
#            self.slide_list = pickle.load(pkl_fh)
        with h5py.File(self.hdf5_path, "r") as db:
            self.valid_slides = [slide_id for slide_id in db]

        self.slide_list = self.metadata[COL_TCGA_SLIDE_ID]

        print("Num valid slides {}".format(len(self.valid_slides)))

        self.num_classes = num_classes

        self.resize_shape = resize_shape
        self.max_patches_per_slide = max_patches

        self.patch_list = self._get_patch_list()
        print("Patch list shape: {}".format(self.patch_list.shape))
       
        self.label_dict = self._get_label_dict(tasks_to)

        self.labels = self._get_labels()
        self._set_class_weights(self.labels)
        self.transform = transform
        self.normalize = normalize
        # tools for patch normalization
        self.standardizer = staintools.BrightnessStandardizer()
        self.color_normalizer = staintools.ReinhardColorNormalizer()
        self.normalizer_with_constants = transforms.Compose([transforms.Normalize(mean = TCGA_MEAN, std = TCGA_STD)])
        self.ToTensor = transforms.Compose([transforms.ToTensor()])
        # tools for image augmentation
        self.stain_augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
 
 
    def get_patch(self, slide_name, patch_num):
        """Index into specific patch within slide"""
        patch = self.hdf5_fh[slide_name][int(patch_num), :, :, :]
        patch = patch.astype(np.float32)
        return patch

    def _get_label_dict(self, tasks_to):
        """Return appropriate label dict for task"""
        return tasks_to

    def _get_labels(self):
        train_metadata_path = os.path.join(self.data_dir, self.split + '.csv')
        train_metadata = pd.read_csv(train_metadata_path)
        labels = train_metadata[COL_TCGA_LABEL].replace(self.label_dict)

        return labels

    def _set_class_weights(self, labels):
        """Set class weights for weighted loss.

        Each task, gets its own set of class weights.

        Weights are calculate by taking 1 - the relative
        frequency of the class (positive vs negative)..

        Args:
            labels: Dataframe or numpy array containing
            a list of the labels. Shape should be
            (num_examples, num_labels)


        Example:
            100 examples with two tasks, cardiomegaly and consolidation.
            10 positve cases of cardiomegaly.
            20 positive cases of consolidation.

            We will then have:
            Class weights for cardiomegaly:
            [1-0.9, 1-0.1] = [0.1, 0.9]
            Class weights for consolidation:
            [1-0.8, 1-0.2] = [0.2, 0.8]

            The first element in each list is the wieght for the
            negative examples.
        """

        # Set weights for positive vs negative examples
        self.p_count = (labels == 1).sum(axis=0)
        self.n_count = (labels == 0).sum(axis=0)

        self.total = self.p_count + self.n_count

        self.class_weights = [[self.n_count / self.total],
                        [self.p_count / self.total]]

    def __len__(self):
        return len(self.patch_list)

    def _label_conversion(self, label):
        """Turn string label into integer"""
        if label not in self.label_dict:
            raise ValueError("Invalid label: {} entered".format(label))
        return self.label_dict[label]

    def transform_label(self, label):
        """Make label correct shape"""
        label = np.array(label).astype(np.float32).reshape(1)
        return label

    #TODO: Decide whether to remove this based on our final choice on image transformation techniques
    def normalize_stained_patch_using_staintools(self, patch):
        """ Normalize a stained tissue patch"""
        #TODO: Code Citation
        
        # standardize brightness and then normalize
        patch = self.standardizer.transform(patch)
        patch = self.color_normalizer.transform(patch)
        return patch
  
    def normalize_patch_with_constants(self, patch):
        """ Normalize using pre-calculated data-specific constants"""
        
        '''
        if not self.is_training:  
            patch = torch.from_numpy(patch)
        '''
        patch = torch.from_numpy(patch)
        
        patch = self.normalizer_with_constants(patch)
        patch = patch.numpy()
        patch = patch.astype(np.float32)
        return patch        
        
    def augmentation_with_staintools(self, patch, idx):
        """" implement image augmentation using staintools"""
        patch = torch.from_numpy(patch)
        patch = transforms.Compose([transforms.ToPILImage()])(patch)
        patch = np.array(patch)
        #print("before")
        #print(patch.shape)
        #print(patch)
        #np.savetxt("temp/" + str(idx) + 'before.txt', patch[:,0,0], delimiter=',')
        #temp = Image.fromarray(patch)
        #temp.save("temp/" + str(idx)+ "PILtest.jpeg")
        self.stain_augmentor.fit(patch)
        augmented_patch = self.stain_augmentor.transform()
        #print("after aug")
        #print(augmented_patch.shape)
        #print(augmented_patch)
        #np.savetxt("temp/" + str(idx) + 'after_aug.txt', augmented_patch[:,0,0], delimiter=',')
        #print("after round up")
        augmented_patch = np.uint8(augmented_patch)
        #np.savetxt("temp/" + str(idx) + 'after_rounding.txt', augmented_patch[:,0,0], delimiter=',')
        augmented_patch = Image.fromarray(augmented_patch)
        #augmented_patch.save("temp/" + str(idx)+ "after rounding.jpeg")
        augmented_patch = self.ToTensor(augmented_patch)
        #print("augmented")
        #print(type(augmented_patch))
        return augmented_patch
      
    def __getitem__(self, idx):
        """Return element of dataset"""
        slide_id, patch_num = self.patch_list[idx]
        patch_num = int(patch_num)
        label = self.get_slide_label(slide_id)
        label = self._label_conversion(label)
        label = self.transform_label(label)
        label = torch.tensor(label, dtype=torch.float32)
        with h5py.File(self.hdf5_path, "r") as db:
            patch = db[slide_id][patch_num, :, :, :].astype(np.float32)

        #TODO: Switch the second aug library
        #if self.transform:
        #    patch = self.transform(patch)
        if self.transform == "staintools":
            patch = self.augmentation_with_staintools(patch, idx)
        #print("between if")
        #print(type(patch))
        if self.normalize:
            patch = self.normalize_patch_with_constants(patch)
        info_dict = {COL_TCGA_SLIDE_ID: slide_id,
                     COL_TCGA_PATCH_ID: patch_num}


        return patch, label, info_dict

    def get_slide_label(self, slide_id):
        """Get label for slide"""
        df = self.metadata.set_index(COL_TCGA_SLIDE_ID)
        return df.loc[slide_id, COL_TCGA_LABEL]

    def _all_patches(self, valid_slides, max_patches_per_slide):
        """Return shuffled list of (slide, patch_idx) tuples"""

        with h5py.File(self.hdf5_path, "r") as db:
            patch_list = [[(slide, patch_idx)
                    for patch_idx in range(len(db[slide]))]
                            for slide in valid_slides]

        if max_patches_per_slide is not None:
            patch_list = [random.sample(slide_patches, max_patches_per_slide)
                    for slide_patches in patch_list]
        patch_list = [tup for slide_patches in patch_list
                      for tup in slide_patches] 
        random.shuffle(patch_list)
        return np.array(patch_list)


    def _get_patch_list(self):
        """Return list of patches to be used for training.

        If patches are pre-filtered, simply add all patches from
        every slide. Otherwise, use is_patch_tumor to differentiate
        patches with tumor cells form patches without. 

        Returns:
            patch_list (list): list of (slide_id, patch_idx) tuples
        """
        if self.max_patches_per_slide is not None and self.max_patches_per_slide > 0:
            print(("Using at most {} patches per slide!").format(
                self.max_patches_per_slide))

        valid_slide_list = list(filter(
                lambda x: x in self.valid_slides, self.slide_list))


        print("Generating Patches...")
        patch_list = []
        if self.filtered: 
            return self._all_patches(valid_slide_list, self.max_patches_per_slide)

        for slide_idx, slide in enumerate(valid_slide_list):
            print("Processing slide: {}".format(slide[COL_TCGA_SLIDE_ID]))
            num_patches = slide[COL_TCGA_NUM_PATCHES]
             
            patch_list_in_slide = slide[COL_TCGA_INDICES] if COL_TCGA_INDICES in slide \
                    else list(range(num_patches))
            random.shuffle(patch_list_in_slide)
            count_added_to_patch_list = 0

            for patch_idx in patch_list_in_slide:
                if patch_idx % 1000 == 0:
                    print(patch_idx)
                label = self.get_slide_label(slide[COL_TCGA_SLIDE_ID])
                tup = (slide[COL_TCGA_SLIDE_ID], str(patch_idx))
                if 'indices' not in slide:
                    patch = self.get_patch(slide[COL_TCGA_SLIDE_ID], patch_idx)
                    patch= np.moveaxis(patch, 0, 2) / 255
                    if not util.is_patch_tumor(patch):
                        continue
                patch_list.append(tup)
                count_added_to_patch_list += 1
                if count_added_to_patch_list == self.max_patches_per_slide:
                    break

        return np.array(patch_list)
