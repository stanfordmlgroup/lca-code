import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np

from dataset import LabelMapper
from dataset import LABEL_SEQS_DICT
from args import TrainArgParser

if __name__ == "__main__":

    """Testing the LabelSequence methods for
       the following scenario:
        1) Train on all 14 pathologies for the NIH examples.
        2) Train on all pathologies for Stanford examples.
        3) Evaluate on the competition pathologies

    """
    su_seq = LABEL_SEQS_DICT['stanford']
    nih_seq = LABEL_SEQS_DICT['nih']
    nih_su_union = LABEL_SEQS_DICT['nih_su_union']

    su_to_union_mapper = LabelMapper(su_seq, nih_su_union)
    nih_to_union_mapper = LabelMapper(nih_seq, nih_su_union)

    su_mock_data = [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ]
    su_mock_data = np.array(su_mock_data)
    nih_mock_data = [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ]
    nih_mock_data = np.array(nih_mock_data)


    # Get mapping from NIH to NIH-Stanford union
    # and mapping from Stanford to NIH-Stanford union

    ex1 = su_to_union_mapper.map(su_mock_data[0])
    ex2 = nih_to_union_mapper.map(nih_mock_data[0])

    print("ex1: ", ex1)
    print("su_mock_data[0]: ", su_mock_data[0])
    LabelMapper.display(su_seq, su_mock_data[0])
    LabelMapper.display(nih_seq, nih_mock_data[0])

    LabelMapper.display(nih_su_union, ex1)
    LabelMapper.display(nih_su_union, ex2)


