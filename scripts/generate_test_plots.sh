python scripts/write_configs.py --old_final --data_dir /deep/group/xray4all

python predict.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name U-Final-predict --su_data_dir=/deep/group/CheXpert/ --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --config_path=dataset/predict_configs/final.json --split=test

python test.py --crop 320 --scale 320 --batch_size 16 --gpu_ids 0 --name U-Final --su_data_dir=/deep/group/CheXpert/ --su_rad_perf_path=/deep/group/CheXpert/rad_perf_test.csv --task_sequence stanford --split=test --use_csv_probs=True --ckpt_path=results/U-Final-predict/test/all_combined.csv
