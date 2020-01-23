

class EnsembleModel(object):
    """Ensemble wrapper to be used for CAMs.
       This should be eventually be converted into a class which looks like a PyTorch model."""

    def __init__(self, task2model_dicts, aggregation_fn, gpu_ids, model_args, data_args):
    
        super(EnsembleModel).__init__()

        self.task2model_dicts = task2model_dicts
        self.aggregation_fn = aggregation_fn
        self.gpu_ids = gpu_ids
        self.model_args = model_args
        self.data_args = data_args

        task2loaded_models = {}
        for task in task2model_dicts:
            loaded_models = self.loaded_model_iterator(task)
            task2loaded_models[task] = loaded_models

        all_loaded_models = [loaded_model for loaded_model in loaded_models for task, loaded_models in task2loaded_models.items()]
        assert all([model1.module.task_sequence == model2.module.task_sequence\
                    for model1, model2 in zip(all_loaded_models[1:], all_loaded_models[:-1])])

        self.task_sequence = all_loaded_models[0].module.task_sequence

        assert all([model1.module.__class__.__name__ == model2.module.__class__.__name__\
                    for model1, model2 in zip(all_loaded_models[1:], all_loaded_models[:-1])])

        self.model_name = all_loaded_models[0].module.__class__.__name__

    def forward(self, x):
        raise ValueError("Foward not implemented for ensemble model.")
        
    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        return self

    @property
    def module(self):
        return self

    def loaded_model_iterator(self, task):
        from saver import ModelSaver
        model_dicts = self.task2model_dicts[task]

        for model_dict in model_dicts:

            ckpt_path = model_dict['ckpt_path']
            self.model_args.model_uncertainty = model_dict['is_3class']
            model, ckpt_info = ModelSaver.load_model(ckpt_path, self.gpu_ids, self.model_args, self.data_args)

            yield model




