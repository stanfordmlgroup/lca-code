import models
import os
import queue
import shutil
import torch
import torch.nn as nn

from dataset import TASK_SEQUENCES


class ModelSaver(object):
    """Class to save and load model ckpts."""
    def __init__(self, save_dir, iters_per_save, max_ckpts, metric_name='val_loss',
                 maximize_metric=False, keep_topk=True, **kwargs):
        """
        Args:
            save_dir: Directory to save checkpoints.
            iters_per_save: Number of iterations between each save.
            max_ckpts: Maximum number of checkpoints to keep before overwriting old ones.
            metric_name: Name of metric used to determine best model.
            maximize_metric: If true, best checkpoint is that which maximizes the metric value passed in via save.
            If false, best checkpoint minimizes the metric.
            keep_topk: Keep the top K checkpoints, rather than the most recent K checkpoints.
        """
        super(ModelSaver, self).__init__()

        self.save_dir = save_dir
        self.iters_per_save = iters_per_save
        self.max_ckpts = max_ckpts
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_metric_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.keep_topk = keep_topk

    def _is_best(self, metric_val):
        """Check whether metric_val is the best one we've seen so far."""
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and self.best_metric_val < metric_val)
                or (not self.maximize_metric and self.best_metric_val > metric_val))

    def save(self, iteration, epoch, model, optimizer, lr_scheduler, device, metric_val):
        """If this iteration corresponds to a save iteration, save model parameters to disk.

        Args:
            iteration: Iteration that just finished.
            epoch: epoch to stamp on the checkpoint
            model: Model to save.
            optimizer: Optimizer for model parameters.
            lr_scheduler: Learning rate scheduler for optimizer.
            device: Device where the model/optimizer parameters belong.
            metric_val: Value for determining whether checkpoint is best so far.
        """
        if iteration % self.iters_per_save != 0:
            return

        ckpt_dict = {
            'ckpt_info': {'epoch': epoch, 'iteration': iteration, self.metric_name: metric_val},
            'model_name': model.module.__class__.__name__,
            'task_sequence': model.module.task_sequence,
            'model_state': model.to('cpu').state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict()
        }
        model.to(device)

        ckpt_path = os.path.join(self.save_dir, 'iter_{}.pth.tar'.format(iteration))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            print(f"Saving the model based on metric={self.metric_name} and \
                    maximize={self.maximize_metric} with value={metric_val}")
            self.best_metric_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path)

        # Add checkpoint path to priority queue (lower priority order gets removed first
        if not self.keep_topk:
            priority_order = iteration
        elif self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, ckpt_path))

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.ckpt_paths.qsize() > self.max_ckpts:
            _, oldest_ckpt = self.ckpt_paths.get()
            try:
                os.remove(oldest_ckpt)
            except OSError:
                pass

    @classmethod
    def load_model(cls, ckpt_path, gpu_ids, model_args, data_args):
        """Load model parameters from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.

        Returns:
            Model loaded from checkpoint, dict of additional checkpoint info (e.g. epoch, metric).
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)

        # Build model, load parameters
        model_fn = models.__dict__[ckpt_dict['model_name']]
        if 'task_sequence' in ckpt_dict:
            task_sequence = ckpt_dict['task_sequence']
        else:
            task_sequence = TASK_SEQUENCES[data_args.task_sequence]

        model = model_fn(task_sequence, model_args)
        model = nn.DataParallel(model, gpu_ids)
        model.load_state_dict(ckpt_dict['model_state'])

        return model, ckpt_dict['ckpt_info']

    @classmethod
    def load_optimizer(cls, ckpt_path, gpu_ids, optimizer, lr_scheduler=None):
        """Load optimizer and LR scheduler state from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for loading the state dict.
            optimizer: Optimizer to initialize with parameters from the checkpoint.
            lr_scheduler: Optional learning rate scheduler to initialize with parameters from the checkpoint.
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
