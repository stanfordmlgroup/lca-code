import os
import util

from datetime import datetime
from dataset.constants import IMAGENET_MEAN, IMAGENET_STD
from tensorboardX import SummaryWriter


class BaseLogger(object):
    def __init__(self, logger_args, start_epoch, num_epochs, batch_size, dataset_len, device, is_training=False):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.device = device
        self.save_dir = logger_args.save_dir if is_training else logger_args.results_dir
        self.num_visuals = logger_args.num_visuals
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(logger_args.name))
        log_dir = os.path.join('logs', logger_args.dir_name)
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.epoch = start_epoch
        # Current iteration in epoch (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = round_down((self.epoch - 1) * dataset_len, batch_size)
        self.iter_start_time = None
        self.epoch_start_time = None
        self.un_normalize = util.UnNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step)

    def visualize(self, inputs, logits, targets, phase, unique_id=None):
        """Visualize predictions and targets in TensorBoard.

        Args:
            inputs: Inputs to the model.
            logits: Logits predicted by the model.
            targets: Target labels for the inputs.
            phase: One of 'train', 'val', or 'test'.
            unique_id: A unique ID to append to every image title. Allows
                for displaying all visualizations separately on TensorBoard.

        Returns:
            Number of examples visualized to TensorBoard.
        """
        num_visualized = 0

        for i in range(self.num_visuals):
            if i >= inputs.shape[0]:
                break

            # Get input, output, and label
            input_np = self.un_normalize(inputs[i])

            # Log to TensorBoard
            tag = '{}/example_{}'.format(phase, (i+1) if unique_id is None else unique_id)
            self.summary_writer.add_image(tag, input_np, self.global_step)

            num_visualized += 1

        return num_visualized

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, optimizer):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError
