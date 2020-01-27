import sys
import json


class Logger(object):
    """Class for logging output."""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        log_dir.mkdir(exist_ok=True)
        self.figure_dir = log_dir / "figures"
        self.table_dir = log_dir / "tables"
        self.log_path = log_dir / "log.txt"
        self.log_file = self.log_path.open('w')
        self.output_config_path = log_dir / "config.json"

    def log(self, *args):
        self.log_stdout(*args)
        print(*args, file=self.log_file)
        self.log_file.flush()

    def log_metrics(self, metrics):
        for metric, value in metrics.items():
            msg = f'{metric}:\t{value}'
            self.log_stdout(msg)
            print(msg, file=self.log_file)
            self.log_file.flush()

    def log_table(self, table, table_name):
        with open(self.table_dir / table_name, 'w') as f:
            print(table, file=f)

    def log_stdout(self, *args):
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def save_config(self, config):
        with self.output_config_path.open('w') as f:
            json.dump(config, f)

    def close(self):
        self.log_file.close()