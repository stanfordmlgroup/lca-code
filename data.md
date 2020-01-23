### Dataset description: 

The dataset contains biospecimens of cancerous tissues as whole slide diagnostic images. The images are stored as .SVS files. This data is obtained from National Cancer Institute’s Genomic Data Commons portal (https://portal.gdc.cancer.gov/repository). For the purpose of this pilot, we use images from 2 primary sites -  Thyroid and Bladder since they have unique morphologies.  
So far we have downloaded 521 WSI with primary site as Thyroid Gland and 457 having primary site as Bladder. 

### Directory structure:

The dataset is stored in /deep/group/aihc-bootcamp-fall2018/path-cbir/data/. The directory structure of the data folder looks like - 

data_path
                slide_list.pkl
                train.hdf5
                val.hdf5
                test.hdf5
                metadata.csv

The following section contains a brief description of the contents and purpose of the various files. 

### Content description:

Metadata.csv - The metadata file stores information regarding all slide images in a tabular format. The columns of this file are [‘primary_site’, ‘slide_id’,’case_id’, ‘path_to_slide’]. 

Primary_site denotes site of origin of cancer, slide_id is the name of the .SVS WSI, ‘case_id’ is a unique id per patient. Each case can have multiple slide images. ‘Path_to_slide’ stores the absolute path of slide images. The metadata file is used to split dataset into train, test and validation. 

Slide_list.pkl - This pickle file contains other metadata about the slide images such as # of levels, level dimensions, downsamples, microns per pixel, vendor and number of patches per slide. This file is used to create the hdf5 files

Train.hdf5, val.hdf5, test.hdf5 - These are the train, validation and test data files. The test.hdf5 is to be hidden.These hdf5 files are used by the dataset object to return patches.


### Split Method: 

The data is split by stratified sampling. The slide images are grouped by cases and unique cases are sampled based on a 80-10-10 split. 

### Toy set description 

The toy dataset we currently have is data_toy_10_23_2018/ which contains 10 WSI in total, 8 in train, 1 each in valid and test.

### Usage instructions:

# Add New Data

1. Make data directory and subdirectories corresponding to distinct classes
2. Download metadata json file and manifest text file from TCGA website for slides in each class and place them in class subdirectories
3. Use gdc-client to download: 

  - 'gdc-client download -m <manifest.txt>'

4. Run tcga_data_prep.py to create metadata and split the metadata into train, test and validation: 

  - 'python tcga_prep.py --data_path {path to data directory} --output_path {path to output directory} --slides-per-site {amount of slides per class, if you want to limit them}'

  This outputs a slide_metadata.csv file various attributes which is split into train.csv, valid.csv and test.csv, and a slide_list.pkl file which is used in the next step to consolidate data. 

6. Run tcga_to_hdf5 to convert to hdf5 files. Works by splitting up job to create separate hdf5 files, and then merges them together. Be warned, this can be time consuming. The command is: 

  - 'python tcga_to_hdf5.py --data_path {path to data directory} --output_path {path to output directory} --num_files {how many files to split job up into} --num_workers {number of threads to use at once} --merged_file {final destination of merged hdf5 file}'

7. Run tcga_split_data.py to split the hdf5 files into train.hdf5, valid.hdf5 and test.hdf5 and merge with existing .hdf5 files.

### Data Versioning

Currently We are versioning the data folders by appending date created onto the folder name. 
