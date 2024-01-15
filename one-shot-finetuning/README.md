# Train and eval
Example: Run the following commands to train and evaluate for MMAct split:
```
nohup python -u new_run_multisensor.py -c /your_data_path/save/check_points --training_iterations 256020 --print_freq 100 --query_per_class_test 1 --query_per_class 1 --shot 1 --way 5 --trans_linear_out_dim 1152 --tasks_per_batch 32 --test_iters 3200 --num_test_tasks 10000 --dataset mmact --split 3 -lr 0.01 --img_size 224 --seq_len 8 --start_gpu 3 --num_workers 10 --num_gpus 1 --method vivit --train_num_classes 33 --alpha 1.0 --mae_model_path /your_data_path/mu-mae-checkpoint-179.pth > mmact_mu_mae.log 2>&1 & 
```
Most of these are the default args.See paper for other hyperparams.



