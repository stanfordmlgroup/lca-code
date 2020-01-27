class BaseTableWriter(object):
    def __init__(self, logger):
        self.logger = logger
        self.table_dir = logger.table_dir
        self.table_dir.mkdir(exist_ok=True)

    def write_tables(self):
        raise NotImplementedError
