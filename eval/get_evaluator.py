from .classification_evaluator import ClassificationEvaluator

def get_evaluator(type_eval, data_loaders, logger, eval_args):
    if type_eval == 'classification':
        return ClassificationEvaluator(data_loaders, logger,
                                       eval_args['num_visuals'],
                                       eval_args['iters_per_eval'],
                                       eval_args['has_missing_tasks'],
                                       eval_args['model_uncertainty'],
                                       eval_args['class_weights'],
                                       eval_args['max_eval'],
                                       eval_args['device'],
                                       eval_args['optimizer'])
    else:
        #TODO: Implement RegressionEvaluator
        return None