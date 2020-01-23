import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from dataset import DataAnalyzer

if __name__ == '__main__':
    data_dir = '/deep/group/xray4all'

    for split in ['train']:
        print("split: ", split)
        da = DataAnalyzer(data_dir, split)

        #print(da.get_studies(True,False,True))

    print(da.df.head())
    print(list(da.df.columns.values))



