import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--results_dir', dest='logger_args.results_dir',type=str, default='results/',
                                 help='Save dir for test results.')
        self.parser.add_argument('--save_cams', dest='logger_args.save_cams', type=util.str_to_bool, default=False, help='If true, will save cams to experiment_folder/cams')
        self.parser.add_argument('--output_csv_name', dest='logger_args.output_csv_name', type=str,
                                 default='output_probs.csv', help='Name for output CSV file (distill.py only).')
        self.parser.add_argument('--split', dest='data_args.split', type=str, default='valid')

        self.parser.add_argument('--only_competition_cams', dest='logger_args.only_competition_cams', type=util.str_to_bool, default=True, help='If true, will only generate cams on competition labels. Only relevant if --save_cams is True')
        self.parser.add_argument('--write_results', dest='logger_args.write_results', type=util.str_to_bool, default=True, help='If true, will append results to results.csv')
        self.parser.add_argument('--use_csv_probs', dest='model_args.use_csv_probs', type=util.str_to_bool, default=False,
                                 help='Use a CSV of probabilities instead of an actual model.')
        self.parser.add_argument('--config_path', type=str, default=None)

        # Multi model args
        self.parser.add_argument('--use_multi_model', type=bool, help='Enables use of one model for each view')
        self.parser.add_argument('--ap_ckpt', dest='multi.ap_ckpt_path', type=str, help='Ckpt to ap model')
        self.parser.add_argument('--pa_ckpt', dest='multi.pa_ckpt_path', type=str,  help='Ckpt to pa model.')
        self.parser.add_argument('--lateral_ckpt', dest='multi.lateral_ckpt_path', type=str,  help='Ckpt to lateral model.')
