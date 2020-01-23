import torch.utils.data as data

from .concat_dataset import ConcatDataset
from .su_dataset import SUDataset
from .nih_dataset import NIHDataset
from .tcga_dataset import TCGADataset
from .pocus_dataset import PocusDataset
from .label_mapper import TASK_SEQUENCES
from .pad_collate import PadCollate
from .tcga_google_dataset import TCGA_Google_Dataset
from .tcga_stanford_dataset import TCGA_Stanford_Dataset

def get_loader(data_args,
               transform_args,
               split,
               task_sequence,
               su_frac,
               nih_frac,
               pocus_frac,
               tcga_frac,
               tcga_google_frac,
               tcga_stanford_frac,
               batch_size,
               is_training=False,
               shuffle=False,
               study_level=False,
               frontal_lateral=False,
               return_info_dict=False,
               normalize=False,
               transform=None):

    """Returns a dataset loader.

       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        su_frac: Float that specifies what percentage of stanford to load.
        nih_frac: Float that specifies what percentage of NIH to load.
        pocus_frac: Float that specifies what percentage of Pocus to load.
        # TODO: remove all the frac arguments and instead pass a dictionary
        split: String determining if this is the train, valid, test, or sample split.
        shuffle: If true, the loader will shuffle the data.
        study_level: If true, creates a loader that loads the image on the study level.
            Only applicable for the SU dataset.
        frontal_lateral: If true, loads frontal/lateral labels.
            Only applicable for the SU dataset.
        return_info_dict: If true, return a dict of info with each image.

    Return:
        DataLoader: A dataloader
    """

    if is_training:
        study_level = data_args.train_on_studies

    datasets = []
    if su_frac != 0:
        datasets.append(
                SUDataset(
                    data_args.su_data_dir,
                    transform_args, split=split,
                    is_training=is_training,
                    tasks_to=task_sequence,
                    frac=su_frac,
                    study_level=study_level,
                    frontal_lateral=frontal_lateral,
                    toy=data_args.toy,
                    return_info_dict=return_info_dict
                    )
                )

    if nih_frac != 0:
        datasets.append(
                NIHDataset(
                    data_args.nih_data_dir,
                    transform_args, split=split,
                    is_training=is_training,
                    tasks_to=task_sequence,
                    frac=nih_frac,
                    toy=data_args.toy
                    )
                )
    if tcga_frac != 0:
        datasets.append(
                TCGADataset(
                    data_args.tcga_data_dir, transform_args,
                    data_args.tcga_meta, split=split,
                    is_training=is_training, toy=data_args.toy,
                    tasks_to=task_sequence, normalize=normalize, 
                    transform=transform
                    )        
                )

    if tcga_google_frac != 0:
        datasets.append(
                TCGA_Google_Dataset(
                    data_args.tcga_google_data_dir, transform_args,
                    data_args.tcga_meta, split=split,
                    is_training=is_training, toy=data_args.toy,
                    tasks_to=task_sequence, normalize=normalize,
                    transform=transform
                    )
                )
    
    if tcga_stanford_frac != 0:
        datasets.append(
                TCGA_Stanford_Dataset(
                    data_args.tcga_stanford_data_dir, transform_args,
                    data_args.tcga_meta, split=split,
                    is_training=is_training, toy=data_args.toy,
                    tasks_to=task_sequence, normalize=normalize, 
                    transform=transform
                    )
                )

    if pocus_frac != 0:
        datasets.append(
            PocusDataset(
                data_args.pocus_data_dir,
                transform_args, split=split,
                is_training=is_training,
                tasks_to=task_sequence,
                frac=pocus_frac,
                toy=data_args.toy
            )
        )

    if len(datasets) == 2:
        assert study_level is False, "Currently, you can't create concatenated datasets when training on studies"
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    # Pick collate function
    if study_level and not (data_args.eval_tcga or data_args.eval_tcga_google or data_args.eval_tcga_stanford):
        collate_fn = PadCollate(dim=0)
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             collate_fn=collate_fn)
    else:
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)

    return loader


def get_eval_loaders(data_args, transform_args, task_sequence, batch_size, frontal_lateral, return_info_dict=False, normalize=False):
    """Returns a dataset loader
       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        eval_su: Float that specifes what percentage of stanford to load.
        nih_frac: Float that specifes what percentage of NIH to load.
        args: Additional arguments needed to load the dataset.
        return_info_dict: If true, return a dict of info with each image.

    Return:
        DataLoader: A dataloader

    """

    eval_loaders = []

    if data_args.eval_su:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=1,
                                    nih_frac=0,
                                    pocus_frac=0,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=not frontal_lateral,
                                    frontal_lateral=frontal_lateral,
                                    return_info_dict=return_info_dict)]

    if data_args.eval_nih:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'train',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=1,
                                    pocus_frac=0,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=True,
                                    return_info_dict=return_info_dict),
                         get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=1,
                                    pocus_frac=0,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=True,
                                    return_info_dict=return_info_dict)]
        
    if data_args.eval_tcga:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'train',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=0,
                                    tcga_frac=1,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    normalize=normalize,
                                    transform=None,
                                    return_info_dict=return_info_dict),
                         get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=0,
                                    tcga_frac=1,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    normalize=normalize,
                                    transform=None,
                                    return_info_dict=return_info_dict)]   
    if data_args.eval_tcga_google:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'test',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=0,
                                    tcga_frac=0,
                                    tcga_google_frac=1,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    normalize=normalize,
                                    transform=None,
                                    return_info_dict=return_info_dict)]
                                    
    if data_args.eval_tcga_stanford:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=0,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=1,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    normalize=normalize,
                                    transform=None,
                                    return_info_dict=return_info_dict)]

    if data_args.eval_pocus:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'train',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=1,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    return_info_dict=return_info_dict),
                         get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=0,
                                    pocus_frac=1,
                                    tcga_frac=0,
                                    tcga_google_frac=0,
                                    tcga_stanford_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    return_info_dict=return_info_dict)]


    return eval_loaders

