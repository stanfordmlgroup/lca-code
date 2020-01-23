import torch
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import TestArgParser

from logger import TestLogger
from saver import ModelSaver
from tqdm import tqdm
#from eval import ModelEvaluator
from collections import OrderedDict
from operator import itemgetter


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

from pathlib import Path
import os

import pandas as pd


def plot(curves, metrics, plot_dir, rad_perf):
    if not os.path.exists(plot_dir):
       os.makedirs(plot_dir)
    
    plot_all_pathologies(curves, metrics, plot_dir, rad_perf)


def plot_all_pathologies(curves, metrics, plot_dir, rad_perf, fig_size = 4):
    # TODO: Generalize so that the list isn't hard-coded.
    pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
   
    figs = []
    for mode in ['ROC', 'PRC']:
        fig, ax = plt.subplots(1, 5, sharex='col', sharey='row', figsize=(fig_size*len(pathologies),fig_size))
        fig.tight_layout()
        for path_index, pathology in enumerate(pathologies):
            keys = [key for key in curves.keys() if pathology in key and mode in key]
            assert(len(keys) == 1)
            key, value = keys[0], curves[keys[0]]
            
            if mode is "ROC":
                ax[0].set_ylabel('True Positive Rate')
            else:
                ax[0].set_ylabel('Precision')
            plot_pathology(mode, value, key, pathology, metrics, ax[path_index], rad_perf)
        fig.savefig(plot_dir / (list(curves.keys())[0].split("_")[0] + mode + '.pdf'), format='pdf', bbox_inches='tight', dpi=300)


def plot_pathology(mode, value, key, category, metrics, ax, rad_perf = None):
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    num_below = metrics[next(m for m in metrics.keys()
                             if (category + 'rads_below_' + mode[:2]) in m)] # first 2 letters of the mode
    ax.set_title(category + " (>{:d} rads)".format(num_below))
    
    if mode == 'ROC':
        x_value = 'Specificity'
        x_value_reverse = True
        y_value = 'Sensitivity'
      
        ax.set_xlabel('False Positive Rate')
        model_x, model_y, _ = value
       
    elif mode == 'PRC':
        x_value = 'Sensitivity'
        y_value = 'Precision'
        x_value_reverse = False

        ax.set_xlabel(x_value)
        model_y, model_x, _ = value

    if rad_perf is not None:
        remap = {
            "Lower": "LabelL",
            "Upper": "LabelU",
            "Maj": "RadMaj"
        }

        colors = {
            "Maj": '#073B4C',
            "Rad1": '#EF476F',
            "Rad2": '#06D6A0',
            "Rad3": '#118AB2',
            "Upper": '#FF9B85',
            "Lower": '#FFD97D',
        }

        names = colors.keys()

        def plot_rad_score(name, x, y):
            point = 'o'
            markersize=10
            mew=1
            if name == 'Maj':
                point = 'x'
            elif name == "Upper" or name == "Lower":
                point = 'v'
            remapped_name = remap[name] if name in remap else name
           
            ax.plot(x, y, point, label="{!s}  ({:0.2f},{:0.2f})".format(remapped_name, x, y),
                markersize=markersize, mew=mew, c=colors[name])

        def get_rad_val(name, value, value_reverse=False):
            val = rad_perf.loc[category + ' ' + value, name]
            if value_reverse is True:
                val = 1 - val
            return val
        
        value_reverse = mode is 'ROC'
        
        for name in names:
            rad_x = get_rad_val(name, x_value, value_reverse = value_reverse)
            rad_y = get_rad_val(name, y_value)
            plot_rad_score(name, rad_x, rad_y)

        ax.plot([get_rad_val("Lower", x_value, value_reverse = value_reverse),
                 get_rad_val("Upper", x_value, value_reverse = value_reverse)],
                [get_rad_val("Lower", y_value),
                 get_rad_val("Upper", y_value)],
                color='#bbbbbb',
                ls=':')
    
    summary_value = 'AU' + mode
    summary = metrics[key[:-3] + summary_value]
    if isinstance(summary, list):
        summary = summary[1] # If we performed the bootstrap, take the mean.

    ax.plot(model_x, model_y, label = 'Model (AUC = %0.2f)' % summary, color='#777777', linewidth = 2)

    leg_location = 'lower right' if mode is 'ROC' else 'best'
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(sorted(list(zip(labels, handles)), key=itemgetter(0)))
    ax.legend(by_label.values(), by_label.keys(), loc=leg_location, fontsize = 'x-small', framealpha=0.3, fancybox=False)
    
