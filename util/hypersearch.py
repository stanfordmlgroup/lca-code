import argparse
from argparse import Namespace
import sys
import util
import numpy as np
import os
import json

# TODO: should be a class
def get_auto_args(parser):
    def get_data_args():
        agg_mean_max_k = int(np.random.choice([2, 3, 5]))
        data_args = {
            # data things
            'data_args.tcga_data_dir': '/deep/group/aihc-bootcamp-fall2018/path-cbir/datasets/data_20x_liver_11_30_2018', #data_40x_liver_11_30_2018, data_liver_40x_toy_11_27_2018, data_10x_liver_12_01_2018
            'data_args.eval_tcga': True,
            'data_args.task_sequence': 'liver',
            
            
            'data_args.nih_train_frac': 0,
            'data_args.su_train_frac': 0,
            'data_args.tcga_train_frac': 1.0,
            
            'batch_size': batch_size,
            
            'has_tasks_missing': False,
            'is_training': True,
            'start_epoch': 1,

            # model things
            'model_args.model': 'DenseNet121',
            'model_args.normalize': True,
            'loss_fn': 'cross_entropy',
            
            #TODO: unccoment once aug is fixed
            #'model_args.transform': 'staintools',

            # training things
            'logger_args.iters_per_print': 500 * batch_size,
            'logger_args.iters_per_eval':  500 * batch_size,
            'logger_args.iters_per_save': 500 * batch_size,
            'logger_args.iters_per_visual': 60000,
            'logger_args.metric_name': 'tcga-valid_loss',
            'logger_args.maximize_metric': False,

        }
        return data_args


    def get_training_knobs():
        learning_rate = np.random.uniform(1e-6,1.5e-5)
        print(learning_rate)
        fine_lr = learning_rate * 0.1
        weight_decay = 1e-6

        scheduler = 'step'
        training_knobs = {
            # training knobs
            'num_epochs': 20,
            'optim_args.lr': learning_rate,
            'optim_args.lr_scheduler': scheduler,
            'optim_args.weight_decay': weight_decay,
            'logger_args.name' : optimizer,
        }

        if scheduler == 'cosine_warmup':
            lr_warmup_steps = 71 * batch_size
            lr_decay_step = 80000
            scheduler_knobs = {
                'optim_args.lr_decay_step': lr_decay_step,
                'optim_args.lr_warmup_steps': lr_warmup_steps,
            }
        elif scheduler == 'step':
            lr_decay_step = 20000 # ridiculously large
            scheduler_knobs = {
                'optim_args.lr_decay_step': lr_decay_step,
                'optim_args.lr_decay_gamma': 0.1,
            }
        else:
            raise NotImplementedError('Scheduler args not implemented')

        training_knobs.update(scheduler_knobs)

        return training_knobs


    def get_optimizer_knobs():
        print(optimizer)
        if optimizer == 'adam':
            optimizer_knobs = {
                'optimizer': 'adam',
            }
        elif optimizer == 'sgd':
            momentum = 0.9
            sgd_dampening = 0
            optimizer_knobs = {
                # optimizer knobs
                'optimizer': 'sgd',
                'sgd_momentum': momentum,
                'sgd_dampening': sgd_dampening,
            }
        else:
            raise NotImplementedError('Optimizer args not implemented')
        return optimizer_knobs
        
    def fix_nested_namespaces(args):
        """Makes sure that nested namespaces work
            Args:
                args: a argsparse.namespace object containing all the arguments
            e.g args.transform.clahe
            Obs: Only one level of nesting is supported.
        """

        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))


        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    batch_size = 10
    
    optimizer = 'adam' #if np.random.rand() > 0.5 else 'sgd'

    defaults = parser.parser.parse_args()
    dict_def = vars(defaults)

    data_args = get_data_args()
    dict_def.update(data_args)


    training_knobs = get_training_knobs()
    dict_def.update(training_knobs)

    optimizer_knobs = get_optimizer_knobs()
    dict_def.update(optimizer_knobs)

    ns = Namespace(**dict_def)
    #args = parser.process_args(ns)
    #print(args)
    args = parser.process_args(ns)
        
    return args
