import json
import argparse

from pathlib import Path

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default="/deep/group/xray4all")
    parser.add_argument('--experiment_dir', default=None)
    parser.add_argument('--old_final', action="store_true")
    parser.add_argument('--new_final', action="store_true")

    return parser


def get_config_list(data_dir, experiment_dir):

    ckpts_dir = Path(data_dir) / "final_ckpts"
    config_list = []

    for run in ["", "_2", "_3"]:

        full_experiment_dir = ckpts_dir / (experiment_dir + run)

        for ckpt_path in full_experiment_dir.glob("*.tar"):
            if "best.pth.tar" in str(ckpt_path):
                continue
            config_dict = {}
            config_dict["ckpt_path"] = str(ckpt_path)
            with open(ckpt_path.parent / "args.json", 'r') as f:
                run_args = json.load(f)
            config_dict["is_3class"] = run_args["model_args"]["model_uncertainty"]
            config_list.append(config_dict)

    return config_list


if __name__ == "__main__":

    parser = get_parser()

    args = parser.parse_args()

    assert args.experiment_dir is not None or args.new_final or args.old_final

    pathologies = ["No Finding",
                   "Enlarged Cardiomediastinum",
                   "Cardiomegaly",
                   "Lung Lesion",
                   "Airspace Opacity",
                   "Edema",
                   "Consolidation",
                   "Pneumonia",
                   "Atelectasis",
                   "Pneumothorax",
                   "Pleural Effusion",
                   "Pleural Other",
                   "Fracture",
                   "Support Devices"]

    configs_dir = Path("dataset/predict_configs")
    configs_dir.mkdir(exist_ok=True)

    if args.old_final:

        path2experiment_dir = {"Atelectasis": "DenseNet121_320_1e-04_uncertainty_ones_top10",
                               "Cardiomegaly": "DenseNet121_320_1e-04_uncertainty_3-class_top10",
                               "Consolidation": "DenseNet121_320_1e-04_uncertainty_self-train_top10",
                               "Edema": "DenseNet121_320_1e-04_uncertainty_ones_top10",
                               "Pleural Effusion": "DenseNet121_320_1e-04_uncertainty_3-class_top10"}

        config = {}
        config["aggregation_method"] = "mean"
        config["task2models"] = {}
        for pathology, experiment_dir in path2experiment_dir.items():

            config_list = get_config_list(args.data_dir, experiment_dir)

            config["task2models"][pathology] = config_list

        with open(configs_dir / "final.json", 'w') as f:
            json.dump(config, f, indent=4)

    elif args.new_final:

        path2experiment_dir = {"Atelectasis": "CheXpert-Ones",
                               "Cardiomegaly": "CheXpert-3-class",
                               "Consolidation": "CheXpert-Self-Train",
                               "Edema": "CheXpert-Ones",
                               "Pleural Effusion": "CheXpert-3-class"}

        config = {}
        config["aggregation_method"] = "mean"
        config["task2models"] = {}
        for pathology, experiment_dir in path2experiment_dir.items():

            config_list = get_config_list(args.data_dir, experiment_dir)

            config["task2models"][pathology] = config_list

        with open(configs_dir / "CheXpert-final.json", 'w') as f:
            json.dump(config, f, indent=4)


    else:

        config_list = get_config_list(args.data_dir, args.experiment_dir)

        config = {}
        config["aggregation_method"] = "mean"
        config["task2models"] = {}
        for pathology in pathologies:

            config["task2models"][pathology] = config_list

        with open(configs_dir / (args.experiment_dir + ".json"), 'w') as f:
            json.dump(config, f, indent=4)




