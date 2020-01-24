import operator
from collections import defaultdict

def collate_accuracy_score(patch_true, patch_pred, slide_ids):
    """Function to calculate accuracy for whole slide prediction.

    Currently uses a majority vote as the aggregation function.

    Args:
        patch_true(ndarray): True labels for each patch
        patch_pred(ndarray): Predicted labels for each patch
        slide_ids(list): slide_ids for each patch

    Returns:
        float:  percent of correctly predicted slides
    """
    slide_true = {}
    slide_counts = defaultdict(lambda: defaultdict(int))

    N = len(slide_ids)

    for i, curr_slide_id in enumerate(slide_ids):
        slide_true[curr_slide_id] = patch_true[i]
        slide_counts[curr_slide_id][patch_pred[i]] += 1

    total = 0.0
    count = 0.0
    for slide_id, slide_label in slide_true.items():
        if max(slide_counts[slide_id].items(), key=operator.itemgetter(1))[0] == slide_label:
            count += 1
        total += 1

    return count/total


