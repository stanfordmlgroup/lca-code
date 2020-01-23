import torch.optim as optim

def get_optimizer(parameters, optim_args):
    """Get a PyTorch optimizer for params.

    Args:
        parameters: Iterator of network parameters to optimize (i.e., model.parameters()).
        optim_args: Command line arguments.

    Returns:
        PyTorch optimizer specified by args_.
    """
    if optim_args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, optim_args.lr,
                              momentum=optim_args.sgd_momentum,
                              weight_decay=optim_args.weight_decay,
                              dampening=optim_args.sgd_dampening)
    elif optim_args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, optim_args.lr,
                               betas=(optim_args.adam_beta_1, optim_args.adam_beta_2), weight_decay=optim_args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim_args.optimizer))

    return optimizer


def get_scheduler(optimizer, optim_args):
    """Get a learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate is modified by the returned scheduler.
        args: Command line arguments.

    Returns:
        PyTorch scheduler that update the learning rate for `optimizer`.
    """
    if optim_args.lr_scheduler is None:
        scheduler = None
    elif optim_args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_args.lr_decay_step, gamma=optim_args.lr_decay_gamma)
    elif optim_args.lr_scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim_args.lr_milestones, gamma=optim_args.lr_decay_gamma)
    elif optim_args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=optim_args.lr_decay_gamma,
                                                         patience=optim_args.lr_patience,
                                                         min_lr=[pg['lr'] * 1e-3 for pg in optimizer.param_groups])
    else:
        raise ValueError('Invalid learning rate scheduler: {}.'.format(optim_args.scheduler))

    return scheduler


def step_scheduler(lr_scheduler, metrics, lr_step, best_ckpt_metric='stanford-valid_loss'):
    """Step a LR scheduler.

    Args:
        lr_scheduler: Scheduler to step.
        metrics: Dictionary of metrics.
        lr_step: Number of times step_scheduler has been called.
        best_ckpt_metric: Name of metric to use to determine the best checkpoint.
    """
    if lr_scheduler is not None:
        lr_step += 1

        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if best_ckpt_metric in metrics:
                lr_scheduler.step(metrics[best_ckpt_metric], epoch=lr_step)
            else:
                raise ValueError(f"Chose {best_ckpt_metric} metric which is not in metrics.")
        else:
            lr_scheduler.step(epoch=lr_step)

    return lr_step
