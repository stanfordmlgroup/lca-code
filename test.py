import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pathlib import Path

from args import TestArgParser
from dataset import get_loader, get_eval_loaders, TASK_SEQUENCES
from eval import get_evaluator
from models import CSVReaderModel, MultiModelWrapper, EnsembleModel
from saver import ModelSaver
from visualize import plot
from predict import get_config
from scripts.get_cams import save_grad_cams


NAN = np.nan


def test(args):
    """Run testing with the given args.

    The function consists of the following steps:
        1. Get model for evaluation.
        2. Get task sequence and class weights.
        3. Get data eval loaders and evaluator.
        4. Evaluate and save model performance (metrics and curves).
    """

    model_args = args.model_args
    logger_args = args.logger_args
    data_args = args.data_args
    transform_args = args.transform_args
    
    
    # Get model
    if args.use_multi_model:
        model = load_multi_model(args.multi, model_args, data_args, args.gpu_ids)
    elif model_args.use_csv_probs:
        model = CSVReaderModel(model_args.ckpt_path, TASK_SEQUENCES[data_args.task_sequence])
        ckpt_info = {'epoch': 0}
    elif args.config_path is not None:
        task2models, aggregation_fn = get_config(args.config_path)
        model = EnsembleModel(task2models, aggregation_fn, args.gpu_ids, model_args, data_args)
        ckpt_info = {'epoch': 0}
    else:
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


    # Get train loader in order to get the class weights
    train_csv_name = 'train'
    if data_args.uncertain_map_path is not None:
        train_csv_name = data_args.uncertain_map_path
    '''
    loader = get_loader(data_args,
                        args.transform_args,
                        train_csv_name,
                        task_sequence,
                        su_frac = 1 if data_args.eval_su else 0,
                        nih_frac = 1 if data_args.eval_nih else 0,
                        pocus_frac = 1 if data_args.eval_pocus else 0,
                        tcga_frac = 1 if data_args.eval_tcga else 0,
                        batch_size=args.batch_size)
    '''
    class_weights = [[0.5],[0.5]]#loader.dataset.class_weights

    '''
    # Get eval loaders and radiologist performance
    eval_loader = get_eval_loaders(data_args,
                                   transform_args,
                                   task_sequence,
                                   args.batch_size,
                                   frontal_lateral=model_args.frontal_lateral,
                                   return_info_dict=model_args.use_csv_probs or logger_args.save_cams)[-1] # Evaluate only on valid
    '''
    rad_perf = pd.read_csv(data_args.su_rad_perf_path) if data_args.su_rad_perf_path is not None else None

    eval_loader = get_loader(data_args,
                                 transform_args,
                                 data_args.split,
                                 task_sequence,
                                 su_frac = 1 if data_args.eval_su else 0,
                                 nih_frac = 1 if data_args.eval_nih else 0,
                                 pocus_frac = 1 if data_args.eval_pocus else 0,
                                 tcga_frac = 1 if data_args.eval_tcga else 0,
                                 tcga_google_frac = 1 if data_args.eval_tcga_google else 0,
                                 tcga_stanford_frac = 1 if data_args.eval_tcga_stanford else 0,
                                 batch_size = args.batch_size,
                                 normalize = model_args.normalize,
                                 study_level = not model_args.frontal_lateral, 
                                 frontal_lateral = model_args.frontal_lateral,
                                 return_info_dict=model_args.use_csv_probs or logger_args.save_cams)
                                 

    results_dir = os.path.join(logger_args.results_dir, data_args.split)   
    visuals_dir = Path(results_dir) / 'visuals'
    if args.config_path is None:
        # Get evaluator

        eval_args = {}
        eval_args['num_visuals'] = None
        eval_args['iters_per_eval'] = None
        eval_args['has_missing_tasks'] = args.has_tasks_missing
        eval_args['model_uncertainty'] = model_args.model_uncertainty
        eval_args['class_weights'] = class_weights
        eval_args['max_eval'] = None
        eval_args['device'] = args.device
        evaluator = get_evaluator('classification', [eval_loader], None, eval_args)

        
        metrics, curves = evaluator.evaluate(model, args.device)
        print(metrics)
        
        # TODO: Generalize the plot function. Remove hard-coded values. 
        # Plot Results
        # print(f"Plotting to {visuals_dir}")
        # TODO: uncomment once we have curves to plot
        # plot(curves, metrics, visuals_dir, rad_perf)

        eval_metrics = ['AUPRC', 'AUROC', 'log_loss', 'rads_below_ROC', 'rads_below_PR']

        if logger_args.write_results:
            results_path = os.path.join(results_dir, f'scores.csv')
            write_results(data_args.dataset_name, data_args.split, eval_metrics, metrics, results_path, logger_args.name, ckpt_info)

    # Save visuals
    if logger_args.save_cams:
        cams_dir = visuals_dir / 'cams'
        save_grad_cams(args, eval_loader, model,
                       cams_dir,
                       only_competition=logger_args.only_competition_cams,
                       only_top_task=False)


def load_multi_model(multi_args, model_args, data_args, gpu_ids):
    """Load multi lodel (a frontal model and a lateral model)."""

    model_ap, ckpt_info_ap = ModelSaver.load_model(multi_args.ap_ckpt_path, gpu_ids, model_args, data_args)
    model_pa, ckpt_info_pa = ModelSaver.load_model(multi_args.pa_ckpt_path, gpu_ids, model_args, data_args)
    model_lateral, lateral_ckpt_info = ModelSaver.load_model(multi_args.lateral_ckpt_path, args.gpu_ids, args)

    # Make sure all models used the same task sequence
    assert model_ap.task_sequence == model_pa.task_sequence
    assert model_pa.task_sequence == model_lateral.task_sequence

    models = {'ap': model_ap,
              'pa': model_pa,
              'lateral': model_lateral}

    model = MultiModelWrapper(models)
    model.task_sequence = model_ap.task_sequence

    return model


def write_results(dataset_name, split, eval_metrics, metrics, output_path, name, ckpt_info):
    """Write model performance to a CSV file."""

    # Create the columns
    cols = ['name', 'dataset', 'epoch']
    for metric in eval_metrics:
        cols.append(metric)

    row = {'name': name,
           'dataset': dataset_name,
           'epoch': ckpt_info['epoch'],
           f'{split}_loss': metrics[dataset_name + f'-{split}_loss'],
           'weighted_loss': metrics[dataset_name + f'-{split}_weighted_loss']}
    

    for col in cols[3:]:
        for metric_name, metric_val in metrics.items():
            if col in metric_name:
                row[col] = metric_val
            elif col not in row:
                row[col] = NAN

    print(f"Writing scores to {output_path}")
    results = pd.DataFrame(row, index=[0])

    results.to_csv(output_path, index=False)
    save_collate_preds(metrics['collate_preds'], metrics['collate_true'])


def save_collate_preds(collate_preds, collate_true):
    """Save the predictions of the slides to a csv when in eval mode.
    """
    print(len(collate_preds) == len(collate_true))
    print('Writing collate predictions to collate_preds.csv')

    with open('collate_preds.csv', 'w') as f:
        f.write('slide_ids, gt, pred, \n')
        for key, pred in collate_preds.items():
            f.write('%s,%d,%d,\n' % (key, collate_true[key], pred))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    test(parser.parse_args())
