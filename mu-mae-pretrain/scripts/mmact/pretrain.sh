# Set the path to save checkpoints
OUTPUT_DIR='your_output_path'
# Set the path to mmact train set.
DATA_PATH='your_data_path/list/mmact/train.csv'


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port your_ip_port  \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --sensor_mask_type tube\
        --sensor_mask_ratio 0.9\
        --loss_hyper 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
