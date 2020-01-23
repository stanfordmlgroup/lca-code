python scripts/write_configs.py --experiment_dir CheXpert-3-class --data_dir /deep/group/CheXpert/
python scripts/write_configs.py --experiment_dir CheXpert-Ignore --data_dir /deep/group/CheXpert/
python scripts/write_configs.py --experiment_dir CheXpert-Self-Train --data_dir /deep/group/CheXpert/
python scripts/write_configs.py --experiment_dir CheXpert-Ones --data_dir /deep/group/CheXpert/
python scripts/write_configs.py --experiment_dir CheXpert-Zeros --data_dir /deep/group/CheXpert/

python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-MultiClass-predict --config_path=dataset/predict_configs/CheXpert-3-class.json --split=valid --su_data_dir=/deep/group/CheXpert/
python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Ignore-predict --config_path=dataset/predict_configs/CheXpert-Ignore.json --split=valid --su_data_dir=/deep/group/CheXpert/
python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-SelfTrained-predict --config_path=dataset/predict_configs/CheXpert-Self-Train.json --split=valid --su_data_dir=/deep/group/CheXpert/
python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Ones-predict --config_path=dataset/predict_configs/CheXpert-Ones.json --split=valid --su_data_dir=/deep/group/CheXpert/
python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Zeros-predict --config_path=dataset/predict_configs/CheXpert-Zeros.json --split=valid --su_data_dir=/deep/group/CheXpert/

python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-MultiClass --task_sequence stanford --split=valid --use_csv_probs=True --ckpt_path=results/CheXpert-U-MultiClass-predict/valid/all_combined.csv --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --su_data_dir=/deep/group/CheXpert/
python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Ignore --task_sequence stanford --split=valid --use_csv_probs=True --ckpt_path=results/CheXpert-U-Ignore-predict/valid/all_combined.csv --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --su_data_dir=/deep/group/CheXpert/
python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-SelfTrained --task_sequence stanford --split=valid --use_csv_probs=True --ckpt_path=results/CheXpert-U-SelfTrained-predict/valid/all_combined.csv --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --su_data_dir=/deep/group/CheXpert/
python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Ones --task_sequence stanford --split=valid --use_csv_probs=True --ckpt_path=results/CheXpert-U-Ones-predict/valid/all_combined.csv --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --su_data_dir=/deep/group/CheXpert/
python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name CheXpert-U-Zeros --task_sequence stanford --split=valid --use_csv_probs=True --ckpt_path=results/CheXpert-U-Zeros-predict/valid/all_combined.csv --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --su_data_dir=/deep/group/CheXpert/
