import json
import numpy as np
import os
import pandas as pd
import torch
import util

from args import TestArgParser
from dataset import get_loader, TASK_SEQUENCES, COL_PATH, CFG_TASK2MODELS, CFG_AGG_METHOD, CFG_CKPT_PATH, CFG_IS_3CLASS
from saver import ModelSaver
from tqdm import tqdm

NEG_INF = -1e9


def predict(args):

    data_args = args.data_args
    logger_args = args.logger_args

    # Get task sequence and CSV column headers
    task_sequence = TASK_SEQUENCES[data_args.task_sequence]
    csv_cols = [COL_PATH] + [task for task in task_sequence]

    # Read config file to get a list of all models that need to be run
    if args.config_path is None:
        raise ValueError("Predict must be run with --config_path set.")
    task2models, aggregation_fn = get_config(args.config_path)

    # Get all unique models
    models = list(set((model_dict[CFG_CKPT_PATH], model_dict[CFG_IS_3CLASS])
                      for models in task2models.values() for model_dict in models))

    # Run models and save predictions to disk
    run_models(args, csv_cols, models, task_sequence)

    output_path = os.path.join(logger_args.results_dir, data_args.split, 'all_combined.csv')
    if os.path.exists(output_path):
        print(f"Aggregate model probabilities already written to {output_path}")
        return

    # Aggregate predictions for each task
    final_df = combine_predictions(args, task2models, task_sequence, aggregation_fn)

    # Write combined predictions to CSV file
    print('Saving combined probabilities to: {}'.format(output_path))
    final_df[csv_cols].to_csv(output_path, index=False)


def combine_predictions(args, task2models, task_sequence, aggregation_fn):
    """Combine predictions of multiple models by reading CSVs from disk.

    Args:
        args: Command-line arguments.
        task2models: Dictionary mapping task names to models (loaded from JSON config file).
        task_sequence: List of tasks to predict.
        aggregation_fn: Aggregation method to use for combine multiple models' predictions.

    Returns:
        Pandas DataFrame with combined probabilities that can be written to disk.
    """
    print('Aggregating predictions...')
    study2probs = {}  # Format: {study_path -> {pathology -> [prob1, prob2, ..., probN]}}

    logger_args = args.logger_args
    data_args = args.data_args

    output_dir = os.path.join(logger_args.results_dir, data_args.split)
    # Load all predictions
    for task in tqdm(task2models.keys()):
        csv_names = [get_checkpoint_identifier(model_dict[CFG_CKPT_PATH]) for model_dict in task2models[task]]
        for csv_name in csv_names:
            # Load CSV of predictions for a single model
            csv_path = os.path.join(output_dir, '{}.csv'.format(csv_name))
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Add predictions to the master dict
                study_path = row[COL_PATH]
                if study_path not in study2probs:
                    study2probs[study_path] = {t: [] for t in task_sequence}

                # Only take the predictions for this task
                study2probs[study_path][task].append(row[task])

    # Combine predictions with the aggregation method from config file, write to CSV
    final_probs = []  # Format: [{example1}, ..., {exampleN}] (each inner dict becomes a row in DataFrame)
    for study_path, task2probs in study2probs.items():
        df_row = {COL_PATH: study_path}
        for task, probs in task2probs.items():
            df_row[task] = aggregation_fn(probs)
        final_probs.append(df_row)

    final_df = pd.DataFrame(final_probs)

    return final_df


def run_models(args, csv_cols, models, task_sequence):
    """Run models and save predicted probabilities to disk.

    Args:
        args: Command-line arguments.
        csv_cols: List of column headers for the CSV files (should be 'Path' + all pathologies).
        models: List of (ckpt_path, is_3class) tuples to use for the models.
        task_sequence: List of tasks to predict for each model.
    """
    data_args = args.data_args
    logger_args = args.logger_args
    model_args = args.model_args
    transform_args = args.transform_args

    # Get eval loader
    data_loader = get_loader(data_args, transform_args, data_args.split,
                             TASK_SEQUENCES[data_args.task_sequence], su_frac=1, nih_frac=0, batch_size=args.batch_size,
                             is_training=False, shuffle=False, study_level=True, return_info_dict=True)

    num_models_finished = 0
    for ckpt_path, is_3class in models:
        
        output_dir = os.path.join(logger_args.results_dir, data_args.split)
        output_path = os.path.join(output_dir, '{}.csv'.format(get_checkpoint_identifier(ckpt_path)))
        if os.path.exists(output_path):
            print(f"Single model probabilities already written to {output_path}")
            continue
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            # Make an empty file so parallel processes run different models
            # open(output_path, 'w').close()
                
        
        util.print_err('Running model {} / {} from path {}'.format(num_models_finished + 1, len(models), ckpt_path))

        # Get model
        model_args.model_uncertainty = is_3class
        model, ckpt_info = ModelSaver.load_model(ckpt_path, args.gpu_ids, model_args, data_args)
        model = model.to(args.device)
        model.eval()

        # Get task sequence predicted by the model
        model_task_sequence = model.module.task_sequence
        if model_task_sequence != task_sequence:
            error_msgs = ['Mismatched task sequences:',
                          'Checkpoint: {}'.format(model_task_sequence),
                          'Args: {}'.format(TASK_SEQUENCES[data_args.task_sequence])]
            raise ValueError('\n  '.join(error_msgs))

        # Sample from the data loader and record model outputs
        csv_rows = []
        num_examples = len(data_loader.dataset)
        with tqdm(total=num_examples, unit=' ' + data_args.split + ' ' + data_args.dataset_name) as progress_bar:
            for inputs, targets, info_dict, mask in data_loader:

                with torch.no_grad():

                    # For Stanford, evaluate on studies
                    if data_args.dataset_name == 'stanford':
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
                    elif data_args.dataset_name == 'nih':
                        logits = model.forward(inputs.to(args.device))
                    else:
                        raise ValueError('Invalid dataset name: {}'.format(data_args.dataset_name))

                    # Save study path and probabilities for each example
                    if is_3class:
                        batch_probs = util.uncertain_logits_to_probs(logits)
                    else:
                        batch_probs = torch.sigmoid(logits)
                    study_paths = info_dict['paths']
                    for probs, path in zip(batch_probs, study_paths):
                        csv_row = {COL_PATH: path, 'DataSplit': data_args.split}
                        csv_row.update({task: prob.item() for task, prob in zip(model_task_sequence, probs)})
                        csv_rows.append(csv_row)

                progress_bar.update(targets.size(0))

        # Write CSV file to disk
        df = pd.DataFrame(csv_rows)
        print('Saving single-model probabilities to: {}'.format(output_path))
        df[csv_cols].to_csv(output_path, index=False)

        num_models_finished += 1


def get_config(config_path):
    """Read configuration from a JSON file.

    Args:
        config_path: Path to configuration JSON file.

    Returns:
        task2models: Dictionary mapping task names to list of dicts.
            Each dict has keys 'ckpt_path' and 'model_uncertainty'.
        aggregation_fn: Aggregation function to combine predictions from multiple models.
    """
    with open(config_path, 'r') as json_fh:
        config_dict = json.load(json_fh)
    task2models = config_dict[CFG_TASK2MODELS]
    agg_method = config_dict[CFG_AGG_METHOD]
    if agg_method == 'max':
        aggregation_fn = np.max
    elif agg_method == 'mean':
        aggregation_fn = np.mean
    else:
        raise ValueError('Invalid configuration: {} = {} (expected "max" or "mean")'.format(CFG_AGG_METHOD, agg_method))

    return task2models, aggregation_fn


def get_checkpoint_identifier(ckpt_path):
    """Get a unique identifier for the checkpoint. Includes experiment name and iteration."""
    path_parts = ckpt_path.split(os.path.sep)
    checkpoint_identifier = path_parts[-2] + "_" + path_parts[-1].replace(".pth.tar", "")

    return checkpoint_identifier


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    predict(parser.parse_args())
