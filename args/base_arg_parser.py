import argparse
import os
import copy

import json
import time
import torch
import torch.backends.cudnn as cudnn
import uuid

import util
from dataset import TASK_SEQUENCES


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CXR')

        # MISC
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')

        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')

        # Model args
        self.parser.add_argument('--model', dest='model_args.model', type=str,
                                 choices=('DenseNet121', 'ResNet152', 'Inceptionv4', 'ResNet18', 'ResNet34',
                                          'ResNeXt101', 'SEResNeXt101', 'NASNetA', 'SENet154', 'MNASNet'),
                                 default='DenseNet121',
                                 help='Model name.')
        self.parser.add_argument('--pretrained', dest='model_args.pretrained', type=util.str_to_bool, default=True,
                                 help='Use a petrained network')
        self.parser.add_argument('--ckpt_path', dest='model_args.ckpt_path', type=str, default='',
                                 help='Path to checkpoint to load. If empty, start from scratch.')
        self.parser.add_argument('--hierarchy', dest='model_args.hierarchy', type=util.str_to_bool, default=False,
                                 help='If true, adds a wrapper around the model that takes into account the \
                                 hiearchy of the labels')
        self.parser.add_argument('--model_uncertainty', dest='model_args.model_uncertainty',
                                 type=util.str_to_bool, default=False,
                                 help='If true, model uncertainty explicity with (pos, neg, uncertain) outputs.')
        self.parser.add_argument('--frontal_lateral', dest='model_args.frontal_lateral', type=util.str_to_bool, default=False,
                                 help='If true, train a model to classify frontal vs. lateral.')
        self.parser.add_argument('--normalize', dest='model_args.normalize', type=util.str_to_bool, default=False,
                                 help='If true, train a model with normalized images')
        self.parser.add_argument('--transform', dest='model_args.transform', type=str, 
                                 choices=('None', 'staintools'), default='None',
                                 help='image augmentation approach')

        # Logger args
        self.parser.add_argument('--name', dest='logger_args.name', type=str, default='debugging', help='Experiment name.')
        self.parser.add_argument('--num_visuals', dest='logger_args.num_visuals', type=int, default=4,
                                 help='Maximum number of visuals per evaluation.')

        self.parser.add_argument('--save_dir', dest='logger_args.save_dir', type=str, default='ckpts/',
                                 help='Directory in which to save model checkpoints.')

        # Data Transformations (put data augmentation in train_arg_parser)
        self.parser.add_argument('--scale', dest='transform_args.scale', default=320, type=int)
        self.parser.add_argument('--crop', dest='transform_args.crop', type=int, default=320)
        self.parser.add_argument('--clahe', dest='transform_args.clahe', type=util.str_to_bool, default=False,
                                 help='CLAHE')
        self.parser.add_argument('--normalization', dest='transform_args.normalization', default='imagenet',
                                 choices=('imagenet', 'cxr_norm'))
        self.parser.add_argument('--maintain_ratio', dest='transform_args.maintain_ratio', type=util.str_to_bool, default=True)

        # Stanford and NIH dataset
        self.parser.add_argument('--toy', dest='data_args.toy', type=util.str_to_bool, default=False,
                                 help='Use smaller dataset')
        self.parser.add_argument('--su_data_dir', dest='data_args.su_data_dir', type=str, default='/deep/group/CheXpert/',
                                 help='Path to Stanford data directory.')
        self.parser.add_argument('--su_rad_perf_path', dest='data_args.su_rad_perf_path', type=str,
                                 default=None,
                                 help='Path to csv containing radiologist performance on Stanford test set.')
        self.parser.add_argument('--pocus_data_dir', dest='data_args.pocus_data_dir', type=str,
                                 default='/deep/group/aihc-bootcamp-fall2018/cxr-tb/data/original/TBPOC_CXR_Neil',
                                 help='Path to Neil Pocus data directory')
        self.parser.add_argument('--nih_data_dir', dest='data_args.nih_data_dir', type=str,
                                 default='/deep/group/rad_data', help='Path to NIH data directory')
        self.parser.add_argument('--tcga_data_dir', dest='data_args.tcga_data_dir', type=str,
                                 default='/deep/group/rad_data', help='Path to TCGA data directory')
        self.parser.add_argument('--tcga_meta', dest='data_args.tcga_meta', type=str,
                                 default='slide_metadata.csv', help='Path to csv containing metadata for TCGA dataset')
        self.parser.add_argument('--tcga_google_data_dir', dest='data_args.tcga_google_data_dir', type=str,
                                 default='/deep/group/rad_data', help='Path to TCGA Google data directory')
        self.parser.add_argument('--tcga_stanford_data_dir', dest='data_args.tcga_stanford_data_dir', type=str,
                                 default='/deep/group/rad_data', help='Path to TCGA Stanford data directory')

        
        # User will have to set at least one eval_dataset flag to true
        self.parser.add_argument('--eval_su', dest='data_args.eval_su', type=util.str_to_bool, default=False,
                                 help='If true, evaluates on Stanford during test and train')
        self.parser.add_argument('--eval_nih', dest='data_args.eval_nih', type=util.str_to_bool, default=False,
                                 help='If true, evaluates NIH during test and train')
        self.parser.add_argument('--eval_tcga', dest='data_args.eval_tcga', type=util.str_to_bool, default=False,
                                 help='If true, evaluates TCGA during test and train')
        self.parser.add_argument('--eval_pocus', dest='data_args.eval_pocus', type=util.str_to_bool, default=False,
                                 help='If true, evaluates Pocus during test and train')
        self.parser.add_argument('--eval_tcga_google', dest='data_args.eval_tcga_google', type=util.str_to_bool, default=False,
                                 help='If true, evaluates Google external datasets for path cbir')
        self.parser.add_argument('--eval_tcga_stanford', dest='data_args.eval_tcga_stanford', type=util.str_to_bool, default=False,
                                 help='If true, evaluates Stanford external datasets for path cbir')
        

        

        self.parser.add_argument('--uncertain_map_path', dest='data_args.uncertain_map_path', type=str, default=None,
                                 help='Path to CSV file which will replace the training CSV.')
        self.parser.add_argument('--task_sequence', dest='data_args.task_sequence', type=str, default='stanford',
                                 choices=('stanford', 'stanford_exclude_NF',
                                          'nih', 'su_nih_union', 'pocus',
                                          'competition', 'single_atelectasis',
                                          'single_cardiomegaly', 'single_consolidation',
                                          'single_edema', 'single_pleural_effusion','tcga', 'liver'),
                                 help='Which task sequence to have the model output. This determines how many neurons \
                                 the model will have in its final layer')

        self.is_training = None
        self.has_tasks_missing = None

    @staticmethod

    def are_tasks_missing(task_sequence, eval_su, eval_nih, eval_pocus, eval_tcga,
                          su_train_frac=None, nih_train_frac=None, pocus_train_frac=None, tcga_train_frac=None):

        """Method used to determined if some examples will have tasks for which there is no label

        Example. If we train on both Stanford and NIH, and have as our label sequence, the union of
        Stanford and NIH, then there will be some labels in NIH that are missing from the sequence
        that we are training on."""

        task_sequence = TASK_SEQUENCES[task_sequence]
        task_seq_su = TASK_SEQUENCES['stanford']
        task_seq_nih = TASK_SEQUENCES['nih']
        task_seq_tcga = TASK_SEQUENCES['tcga']
        task_seq_pocus = TASK_SEQUENCES['pocus']

        if eval_su or su_train_frac == 1:
            if set(task_sequence).issubset(set(task_seq_su)) is False:
                return True

        if eval_nih or nih_train_frac == 1:
            if set(task_sequence).issubset(set(task_seq_nih)) is False:
                return True

        if eval_tcga or tcga_train_frac == 1:
            return False
        
        if eval_pocus or pocus_train_frac == 1:
            if set(task_sequence).issubset(set(task_seq_pocus)) is False:
                return True

        return False

    def namespace_to_dict(args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = BaseArgParser.namespace_to_dict(obj)

        return args_dict

    @staticmethod
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

    def parse_args(self):
        args = self.parser.parse_args()
        return self.process_args(args)
    
    
    def process_args(self, args):

        # Make sure nested name spaces work:
        # e.g args.transform.clahe works
        # obs, only one level of nesting
        
        self.fix_nested_namespaces(args)


        # Will labels for some tasks be missing for some examples
        if self.is_training:
            args.has_tasks_missing = self.are_tasks_missing(
                                            args.data_args.task_sequence,
                                            args.data_args.eval_su,
                                            args.data_args.eval_nih,
                                            args.data_args.eval_pocus,
                                            args.data_args.eval_tcga,
                                            args.data_args.su_train_frac,
                                            args.data_args.nih_train_frac,
                                            args.data_args.pocus_train_frac,
                                            args.data_args.tcga_train_frac)
        else:
            args.has_tasks_missing = self.are_tasks_missing(
                                            args.data_args.task_sequence,
                                            args.data_args.eval_su,
                                            args.data_args.eval_nih,
                                            args.data_args.eval_pocus,
                                            args.data_args.eval_tcga)


        # Save args to a JSON file
        random_string = '_' + uuid.uuid4().hex.upper()[0:6]
        date_string = str(int(time.time() * 1000)) + random_string
        args.logger_args.dir_name = '{}_{}'.format(args.logger_args.name, date_string)

        save_dir = os.path.join(args.logger_args.save_dir, args.logger_args.dir_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(BaseArgParser.namespace_to_dict(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.logger_args.save_dir = save_dir


        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        args.start_epoch = 1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.model_args.ckpt_path \
                and not (hasattr(args, 'test_2d') and args.test_2d) \
                and args.config_path is None:
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.logger_args.iters_per_eval % args.batch_size != 0:
            raise ValueError('iters_per_eval must be divisible by batch_size.')
        if args.is_training and args.logger_args.iters_per_save % args.logger_args.iters_per_eval != 0:
            raise ValueError('iters_per_save must be divisible by iters_per_eval.')

        # Make sure the best_ckpt_metric is the right one
        if args.is_training:
            if args.data_args.eval_su and not args.data_args.eval_nih:

                assert(args.logger_args.metric_name in ['stanford-valid_loss', 'stanford-train-dev_loss',
                                            'stanford-valid_5loss', 'stanford-valid_competition_avg_AUROC'])

            if args.data_args.eval_nih and not args.data_args.eval_su:
                assert(args.logger_args.metric_name in ['nih-valid_weighted_loss', 'nih-valid_loss'])

            if args.data_args.eval_nih and args.data_args.eval_su:
                print(f'Watch out: You are evaluating on both NIH and Stanford and your metric_name is {args.logger_args.metric_name}.')
            
            assert ('loss' in args.logger_args.metric_name and not args.logger_args.maximize_metric) or\
                   ('AUROC' in args.logger_args.metric_name and args.logger_args.maximize_metric)

        else:
            # Save dataset name to data_args
            if args.data_args.eval_su:
                args.data_args.dataset_name = 'stanford'
            elif args.data_args.eval_nih:
                args.data_args.dataset_name = 'nih'
            elif args.data_args.eval_tcga:
                args.data_args.dataset_name = 'tcga'
            elif args.data_args.eval_tcga_google:
                args.data_args.dataset_name = 'tcga_google'
            elif args.data_args.eval_tcga_stanford:
                args.data_args.dataset_name = 'tcga_stanford'
            elif args.data_args.eval_pocus:
                args.data_args.dataset_name = 'pocus'
            else:
                raise Exception('need to select a dataset to evaluate on')

        # Set up available GPUs
        args.gpu_ids = util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            cudnn.benchmark = True
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        # Set up output dir (test mode only)
        if not self.is_training:
            args.logger_args.results_dir = os.path.join(args.logger_args.results_dir, args.logger_args.name)
            os.makedirs(args.logger_args.results_dir, exist_ok=True)

        return args
