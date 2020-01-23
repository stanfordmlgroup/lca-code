import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
from args import TrainArgParser
from dataset import get_loader
from dataset import TASK_SEQUENCES
import numpy as np

def train(args):
    model_args = args.model_args
    data_args = args.data_args
    transform_args = args.transform_args

    task_sequence = TASK_SEQUENCES[data_args.task_sequence]

    # Get loaders and class weights
    train_csv_name = 'train'
    if data_args.uncertain_map_path is not None:
        train_csv_name = data_args.uncertain_map_path
    train_loader1 = get_loader(data_args, transform_args, train_csv_name, task_sequence, data_args.su_train_frac,
                              data_args.nih_train_frac, data_args.pocus_train_frac, data_args.tcga_train_frac, 0,
                              args.batch_size,
                              frontal_lateral=model_args.frontal_lateral, is_training=True, shuffle=True)
    train_loader2 = get_loader(data_args, transform_args, train_csv_name, task_sequence, data_args.su_train_frac,
                              data_args.nih_train_frac, data_args.pocus_train_frac, data_args.tcga_train_frac, 0,
                              args.batch_size,
                              frontal_lateral=model_args.frontal_lateral, is_training=True, shuffle=True)

    running_mean = None
    
    for data in train_loader1:
        inputs, targets, info_dict = data
        
        if running_mean is None:
            running_mean = np.mean(inputs.numpy(),axis=(0,2,3)).reshape((1,3))
        else:
            running_mean = np.concatenate((running_mean, np.mean(inputs.numpy(),axis=(0,2,3)).reshape((1,3))), axis=0)

        print(running_mean.shape)
    
    
    means = np.mean(running_mean,axis=0)
    print('means: ', means)
    
    num_total = 0.0
    running_vars = [0.0, 0.0, 0.0]
    for data in train_loader2:
        inputs, targets, info_dict = data
        inputs = inputs.numpy()
        num_total += inputs.shape[0]
        for i in range(3):
            running_vars[i] += np.sum(abs(inputs[:,i,:,:] - means[i]))
    
    stds = [0.0, 0.0, 0.0]
    for i in range(3):
        stds[i] = np.sqrt(running_vars[i] / num_total / 512.0 / 512.0)
    
    print('stds: ', stds)



if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    train(parser.parse_args())
