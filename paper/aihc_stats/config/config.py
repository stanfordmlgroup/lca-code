import json


def load_config(config_path, logger=None):
    with config_path.open('r') as f:
        config = json.load(f)

    validate_config(config)

    if logger is not None:
        logger.save_config(config)

    return config


def validate_config(config):
    if not config["tune_threshold"] and (config["tune_probabilities_path"] is None or
                                         config["tune_groundtruth_path"] is None):
        raise ValueError("Must provide paths to tuning set when tuning threshold.")

    if ("default" not in config["compare_probabilities_paths"]):
        raise ValueError("Must pass in a default experiment to compare to.")

    if sorted(list(config["eval_probabilities_paths"].keys())) != sorted(list(config["tune_probabilities_paths"].keys())):
        raise ValueError("Must pass in identical experiment names for tuning and evaluation.")