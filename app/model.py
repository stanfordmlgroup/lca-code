import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from torchvision import models


def uncertain_logits_to_probs(logits):
    """Convert explicit uncertainty modeling logits to probabilities P(is_abnormal).

    Args:
        logits: Input of shape (batch_size, num_tasks * 3).

    Returns:
        probs: Output of shape (batch_size, num_tasks).
            Position (i, j) interpreted as P(example i has pathology j).
    """
    b, n_times_d = logits.size()
    d = 3
    if n_times_d % d:
        raise ValueError('Expected logits dimension to be divisible by {}, got size {}.'.format(d, n_times_d))
    n = n_times_d // d

    logits = logits.view(b, n, d)
    probs = F.softmax(logits[:, :, 1:], dim=-1)
    probs = probs[:, :, 1]

    return probs


class Model(nn.Module):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """
    def __init__(self, model_fn, task_sequence, model_uncertainty, use_gpu):
        super(Model, self).__init__()

        self.task_sequence = task_sequence
        self.model_uncertainty = model_uncertainty
        self.use_gpu = use_gpu

        # Set pretrained to False to avoid loading weights which will be overwritten
        self.model = model_fn(pretrained=False) 

        self.pool = nn.AdaptiveAvgPool2d(1)

        num_ftrs = self.model.classifier.in_features
        if self.model_uncertainty:
            num_outputs = 3 * len(task_sequence)
        else:
            num_outputs = len(task_sequence)

        self.model.classifier = nn.Linear(num_ftrs, num_outputs)

        self.fmaps = OrderedDict()
    
        for name, module in self.model.named_modules():
            # Only put hooks on the target layer
            if name == "features":
                self.target_module_id = id(module)
                module.register_forward_hook(self.save_fmap)

    def set_weights_attribute(self, weights):
        self.weights = weights

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)

        return x

    def save_fmap(self, m, _, output):
        if self.use_gpu:
            self.fmaps[id(m)] = output
        else:
            self.fmaps[id(m)] = output.to('cpu')

    def infer(self, x, tasks):

        with torch.set_grad_enabled(True):

            self.model.zero_grad()

            preds = self(x)

            get_probs = uncertain_logits_to_probs if self.model_uncertainty else torch.sigmoid

            probs = get_probs(preds)[0]

            fmaps = self.fmaps[self.target_module_id]
            
            task2prob_cam = {}
            for task in tasks:

                idx = self.task_sequence[task]

                # Sum up along the filter dimension
                cam = (fmaps * self.weights[idx:idx+1]).sum(dim=1)

                cam = torch.clamp(cam, min=0, max=float('inf'))

                cam -= cam.min()
                cam /= (cam.max() + 1e-7)

                task_prob = probs.detach().cpu().numpy()[idx]
                task_cam = cam.detach().cpu().numpy()[0]

                task2prob_cam[task] = (task_prob, task_cam)

            return task2prob_cam



class DenseNet121(Model):
    def __init__(self, task_sequence, model_uncertainty, use_gpu):
        super(DenseNet121, self).__init__(models.densenet121, task_sequence, model_uncertainty, use_gpu)


def load_individual(ckpt_path, model_uncertainty, use_gpu=False):

    device = 'cuda:0' if use_gpu else 'cpu'
    ckpt_path  = os.path.join(os.path.dirname(__file__), ckpt_path)
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    # Build model, load parameters
    task_sequence = ckpt_dict['task_sequence']
    model = DenseNet121(task_sequence, model_uncertainty, use_gpu)
    model = nn.DataParallel(model)
    model.load_state_dict(ckpt_dict['model_state'])

    params = list(model.module.parameters())
    weights = np.squeeze(params[-2].data.numpy())
    if model_uncertainty:
        weights = weights[2::3]
    model.module.set_weights_attribute(torch.tensor(weights).to(device).unsqueeze(2).unsqueeze(3))

    return model.eval().to(device), ckpt_dict['ckpt_info']


class Tasks2Models(object):
    """
    Main attribute is a (task tuple) -> {iterator, list} dictionary,
    which loads models iteratively depending on the
    specified task.
    """ 
    def __init__(self, config_path, num_models=1, dynamic=True, use_gpu=False):
    
        super(Tasks2Models).__init__()

        self.get_config(config_path)
        self.dynamic = dynamic
        self.use_gpu = use_gpu

        if dynamic:
            model_loader = self.model_iterator
        else:
            model_loader = self.model_list

        model_dicts2tasks = {}
        for task, model_dicts in self.task2model_dicts.items():
            hashable_model_dict = self.get_hashable(model_dicts)
            if hashable_model_dict in model_dicts2tasks:
                model_dicts2tasks[hashable_model_dict].append(task)
            else:
                model_dicts2tasks[hashable_model_dict] = [task]

        # Initialize the iterators
        self.tasks2models = {}
        for task, model_dicts in self.task2model_dicts.items():
            hashable_model_dict = self.get_hashable(model_dicts)
            tasks = tuple(model_dicts2tasks[hashable_model_dict])

            if tasks not in self.tasks2models:
                self.tasks2models[tasks] = model_loader(model_dicts, num_models=num_models)

        self.tasks = list(self.task2model_dicts.keys())

    def get_hashable(self, model_dicts):
        return tuple([tuple(model_dict.items()) for model_dict in model_dicts])

    @property
    def module(self):
        return self

    def get_config(self, config_path):
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
        self.task2model_dicts = config_dict['task2models']
        agg_method = config_dict['aggregation_method']
        if agg_method == 'max':
            self.aggregation_fn = np.max
        elif agg_method == 'mean':
            self.aggregation_fn = np.mean
        else:
            raise ValueError('Invalid configuration: {} = {} (expected "max" or "mean")'.format('aggregation_method', agg_method))

    def model_iterator(self, model_dicts, num_models):
        
        def iterator():

            for model_dict in model_dicts[:num_models]:

                ckpt_path = model_dict['ckpt_path']
                model_uncertainty = model_dict['is_3class']
                model, ckpt_info = load_individual(ckpt_path, model_uncertainty, self.use_gpu)

                yield model

        return iterator

    def model_list(self, model_dicts, num_models):
        
        loaded_models = []
        for model_dict in model_dicts[:num_models]:
            
            ckpt_path = model_dict['ckpt_path']
            model_uncertainty = model_dict['is_3class']
            model, ckpt_info = load_individual(ckpt_path, model_uncertainty, self.use_gpu)

            loaded_models.append(model)

        def iterator():
            return loaded_models

        return iterator

    def infer(self, img, tasks):
        
        ensemble_probs = []
        cams = []

        model_iterable = self.tasks2models[tasks]
        task2ensemble_probs_cams = {}
        for model in model_iterable():
            individual_task2prob_cam = model.module.infer(img, tasks)

            for task in tasks:
                if task not in task2ensemble_probs_cams:
                    task2ensemble_probs_cams[task] = [individual_task2prob_cam[task]]
                else:
                    task2ensemble_probs_cams[task].append(individual_task2prob_cam[task])

        assert all([task in task2ensemble_probs_cams for task in tasks]), "Not all tasks in task2ensemble_probs_cams"

        task2prob_cam = {}
        for task in tasks:
            ensemble_probs, cams = zip(*task2ensemble_probs_cams[task])
            task2prob_cam[task] = (self.aggregation_fn(ensemble_probs, axis=0), self.aggregation_fn(cams, axis=0))

        assert all([task in task2prob_cam for task in tasks]), "Not all tasks in task2prob_cam"

        return task2prob_cam

    def __iter__(self):
        return iter(self.tasks2models)



