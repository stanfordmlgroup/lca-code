from .base_arg_parser import BaseArgParser
import util


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Logger args
        self.parser.add_argument('--iters_per_save', dest='logger_args.iters_per_save', type=int, default=8192,
                                 help='Number of epochs between saving a checkpoint to save_dir.')
        self.parser.add_argument('--iters_per_print', dest='logger_args.iters_per_print', type=int, default=16,
                                 help='Number of iterations between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--iters_per_eval', dest='logger_args.iters_per_eval', type=int, default=8192,
                                 help='Number of iterations between evaluations of the model.')
        self.parser.add_argument('--iters_per_visual', dest='logger_args.iters_per_visual', type=int, default=16384,
                                 help='Number of iterations between visualizing training examples.')
        self.parser.add_argument('--max_eval', dest='logger_args.max_eval', type=int, default=-1,
                                 help='Max number of examples to evaluate from the training set.')
        self.parser.add_argument('--max_ckpts', dest='logger_args.max_ckpts', type=int, default=10,
                                 help='Number of checkpoints to keep before overwriting old ones.')
        self.parser.add_argument('--keep_topk', dest='logger_args.keep_topk', type=util.str_to_bool,
                                 default=True, help='Keep the top K checkpoints instead of most recent K checkpoints.')

        # MISC
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='Number of epochs to train. If 0, train forever.')
        # Evaluator args
        self.parser.add_argument('--metric_name', type=str, dest='logger_args.metric_name', default='stanford-valid_competition_avg_AUROC',
                                 choices=('stanford-valid_5loss', 'stanford-valid_loss', 'stanford-valid_weighted_loss',
                                          'stanford-train-dev_loss', 'nih-valid_loss', 'nih-valid_weighted_loss',
                                          'stanford-valid_competition_avg_AUROC', 'pocus-valid_loss', 'tcga-valid_loss'))
        self.parser.add_argument('--maximize_metric', dest='logger_args.maximize_metric', type=util.str_to_bool, default=True,
                                 help='If True, maximize the metric specified by metric_name. Otherwise, minimize it.')

        # Optimizer
        self.parser.add_argument('--adam_beta_1', dest='optim_args.adam_beta_1', type=float,
                                 default=0.9, help='Adam beta 1 (Adam only).')
        self.parser.add_argument('--adam_beta_2', dest='optim_args.adam_beta_2', type=float,
                                 default=0.999, help='Adam beta 2 (Adam only).')
        self.parser.add_argument('--optimizer', dest='optim_args.optimizer', type=str, default='adam',
                                 choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum', dest='optim_args.sgd_momentum', type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening', dest='optim_args.sgd_dampening', type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--weight_decay', dest='optim_args.weight_decay', type=float, default=0.0,
                                 help='Weight decay (i.e., L2 regularization factor).')

        # Learning rate
        self.parser.add_argument('--lr', dest='optim_args.lr', type=float, default=1e-4, help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler', dest='optim_args.lr_scheduler', type=str, default=None,
                                 choices=(None, 'step', 'multi_step', 'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma', dest='optim_args.lr_decay_gamma', type=float, default=0.1,
                                 help='Multiply learning rate by this value every LR step (step and multi_step only).')
        self.parser.add_argument('--lr_decay_step', dest='optim_args.lr_decay_step', type=int, default=100,
                                 help='Number of epochs between each multiply-by-gamma step.')
        self.parser.add_argument('--lr_milestones', dest='optim_args.lr_milestones', type=str, default='50,125,250',
                                 help='Epochs to step the LR when using multi_step LR scheduler.')
        self.parser.add_argument('--lr_patience', dest='optim_args.lr_patience',  type=int, default=2,
                                 help='Number of stagnant epochs before stepping LR.')

        # Loss function
        # Taking out weighted_loss because it is buggy with 5 loss
        self.parser.add_argument('--loss_fn', type=str, choices=('cross_entropy','weighted_loss'), default='cross_entropy',
                                 help='loss function, default: cross_entropy and weighted_loss are possible.')

        # Data arguments
        # In order for train.py to work, you need to specify at least one training fraction!
        # TODO: Remove the *_train_frac flags and replace with flag to specify dataset(s). 
        self.parser.add_argument('--nih_train_frac', dest='data_args.nih_train_frac', type=float, default=0,
                                 help='Fraction of NIH examples to train on')
        self.parser.add_argument('--su_train_frac', dest='data_args.su_train_frac', type=float, default=0,
                                 help='Fraction of Stanford examples to train on')
        self.parser.add_argument('--tcga_train_frac', dest='data_args.tcga_train_frac', type=float, default=0,
                                 help='Fraction of TCGA examples to train on')
        self.parser.add_argument('--pocus_train_frac', dest='data_args.pocus_train_frac', type=float, default=0,
                                 help='Fraction of Neil Pocus examples to train on')

        self.parser.add_argument('--train_on_studies', dest='data_args.train_on_studies',
                                 type=util.str_to_bool, default=False,
                                 help='If true, you train on the study level, instead of training on the level \
                                 of the individual image.')

        # Data augmentation
        self.parser.add_argument('--rotate', dest='transform_args.rotate', default=0, type=int)
        self.parser.add_argument('--horizontal_flip', dest='transform_args.horizontal_flip', type=util.str_to_bool,
                                 default=False)
