from eval import AverageMeter
from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, logger_args, start_epoch, num_epochs, batch_size, dataset_len, device):
        super(TrainLogger, self).__init__(logger_args, start_epoch, num_epochs, batch_size, dataset_len, device, is_training=True)

        assert logger_args.iters_per_print % batch_size == 0, "iters_per_print must be divisible by batch_size"
        assert logger_args.iters_per_visual % batch_size == 0, "iters_per_visual must be divisible by batch_size"

        self.iters_per_print = logger_args.iters_per_print
        self.iters_per_visual = logger_args.iters_per_visual
        self.experiment_name = logger_args.name
        self.max_eval = logger_args.max_eval
        self.num_epochs = num_epochs
        self.loss_meter = AverageMeter()
        self.w_loss_meter = AverageMeter()

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def plot_metrics(self, metrics):
        """Plot a dictionary of metrics to TensorBoard."""
        if metrics is not None:
            self._log_scalars(metrics)

    def log_iter(self, inputs, logits, targets, unweighted_loss, weighted_loss, optimizer):
        """Log results from a training iteration."""
        loss = unweighted_loss.item()
        w_loss = weighted_loss.item() if weighted_loss else None

        self.loss_meter.update(loss, inputs.size(0))
        if w_loss is not None:
            self.w_loss_meter.update(w_loss, inputs.size(0))

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, loss: {:.3g}, wloss {:3g}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self.loss_meter.avg, self.w_loss_meter.avg)

            # Write all errors as scalars to the graph
            self._log_scalars({'batch_lr': optimizer.param_groups[0]['lr']}, print_to_stdout=False)
            self._log_scalars({'batch_loss': self.loss_meter.avg}, print_to_stdout=False)
            self._log_scalars({'batch_wloss': self.w_loss_meter.avg}, print_to_stdout=False)
            self.loss_meter.reset()
            self.w_loss_meter.reset()

            self.write(message)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.iters_per_visual == 0:
            self.visualize(inputs, logits, targets, phase='train')

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self, metrics, optimizer):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            optimizer: Optimizer for the model.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}, lr: {}]'
                   .format(self.epoch, time() - self.epoch_start_time, optimizer.param_groups[0]['lr']))
        if metrics is not None:
            self._log_scalars(metrics)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
