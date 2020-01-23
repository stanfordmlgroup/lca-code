import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import os
import pandas as pd
import torch
import util

from args import TestArgParser
from dataset import get_loader, TASK_SEQUENCES, COL_PATH, COL_SPLIT
from saver import ModelSaver
from tqdm import tqdm

NEG_INF = -1e9


def test(args):

    data_args = args.data_args
    model_args = args.model_args
    logger_args = args.logger_args
    transform_args = args.transform_args

    # Get model
    model, ckpt_info = ModelSaver.load_model(model_args.ckpt_path, args.gpu_ids, model_args, data_args)
    model = model.to(args.device)
    model.eval()

    # Get the task sequence that the model outputs.
    # Newer models have an attribute called 'task_sequence'.
    # For older models we need to specify what
    # task sequence was used.
    if hasattr(model.module, 'task_sequence'):
        task_sequence = model.module.task_sequence
    else:
        task_sequence = TASK_SEQUENCES[data_args.task_sequence]
        print(f'WARNING: assuming that the models task sequence is \n {task_sequence}')

    # Get eval loader
    loader = get_loader(data_args, transform_args, data_args.split,
                        task_sequence, su_frac=1, nih_frac=0, batch_size=args.batch_size,
                        is_training=False, shuffle=False, study_level=not model_args.frontal_lateral, 
                        frontal_lateral=model_args.frontal_lateral, return_info_dict=True)

    if model_args.frontal_lateral:
        task_sequence = ["Frontal/Lateral"]

    # Sample from the data loader and record model outputs
    csv_rows = []
    num_evaluated = 0
    num_examples = len(loader.dataset)
    with tqdm(total=num_examples, unit=' ' + data_args.split + ' ' + data_args.dataset_name) as progress_bar:
        for data in loader:
            if num_evaluated >= num_examples:
                break

            with torch.no_grad():

                if loader.dataset.study_level:

                    inputs, targets, mask, info_dict = data
                    # Fuse batch size `b` and study length `s`
                    b, s, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                    # Predict
                    logits = model.forward(inputs.to(args.device))
                    logits = logits.view(b, s, -1)

                    # Mask padding to negative infinity
                    ignore_where = (mask == 0).unsqueeze(-1).repeat(1, 1, logits.size(-1)).to(args.device)
                    logits = torch.where(ignore_where, torch.full_like(logits, NEG_INF), logits)
                    logits, _ = torch.max(logits, 1)
                else:

                    inputs, targets, info_dict = data

                    logits = model.forward(inputs.to(args.device))

                # Save study path and probabilities for each example
                if model_args.model_uncertainty:
                    batch_probs = util.uncertain_logits_to_probs(logits)
                else:
                    batch_probs = torch.sigmoid(logits)
                study_paths = info_dict['paths']
                for probs, path in zip(batch_probs, study_paths):
                    csv_row = {COL_PATH: path, COL_SPLIT: data_args.split}
                    csv_row.update({task: prob.item() for task, prob in zip(task_sequence, probs)})
                    csv_rows.append(csv_row)

            progress_bar.update(targets.size(0))
            num_evaluated += targets.size(0)

    # Write CSV file to disk
    df = pd.DataFrame(csv_rows)
    if model_args.frontal_lateral:
        int2str = {0: "Frontal", 1: "Lateral"}
        df["Frontal/Lateral"] = df["Frontal/Lateral"].apply(lambda x: int2str[int(x > 0.5)])
    csv_cols = [COL_PATH] + [task for task in task_sequence]
    output_path = os.path.join(logger_args.results_dir, logger_args.output_csv_name)
    print('Saving outputs to: {}'.format(output_path))
    df[csv_cols].to_csv(output_path, index=False)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    test(parser.parse_args())
