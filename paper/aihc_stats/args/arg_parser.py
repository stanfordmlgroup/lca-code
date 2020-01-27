import argparse
from pathlib import Path


class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="AIHC Stats")

        self.parser.add_argument("-c", "--config_path", default="config.json",
                                 help="Path to config settings.")
        self.parser.add_argument("-l", "--log_dir", default="logs/",
                                 help="Directory to store metrics and plots.")

    def parse_args(self):
        args = self.parser.parse_args()

        args.config_path = Path(args.config_path)
        args.log_dir = Path(args.log_dir)

        return args
