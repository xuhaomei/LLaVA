#!/bin/bash
# nohup bash script.sh > /mnt/hdd/weiliu/student/xhm/LLaVA_logs/mix665k-shuffled-2e-2-sn50-t128followt32_400.out &
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY="06124ddc00b85edf3d9ff4ca8e5643372d1234f7"
export WANDB_PROJECT="LLaVA-Token-Selector"
export WANDB_DIR="/mnt/hdd/weiliu/student/xhm/LLaVA_logs/llava-token-selector/wandb"
export HF_HOME="/mnt/hdd/weiliu/student/xhm/cache"

sample_num=50
lr=2e-2
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
k=128
lambda_r=0.01
# data_json=Cambrian737k
# data_path="/mnt/hdd/weiliu/student/xhm/data/cambrian_737k/Cambrian737k_shuffled.json"
# image_folder="/mnt/hdd/weiliu/student/xhm/data/cambrian_737k"
data_json=mix665k_shuffled
data_path="/home/weiliu/student/xhm/LLaVA/playground/data/llava_v1_5_mix665k_shuffled.json"
image_folder="/home/weiliu/student/xhm/LLaVA/playground/data"
selector_mode=linear
other_cfg=t128followt32_400

python /home/weiliu/student/xhm/LLaVA/llava/train/train_token_selector.py \
        --mode $mode\
        --k $k\
        --sample_num $sample_num \
        --alpha_pg_loss 1.0 \
        --lambda_r $lambda_r \
        --learning_rate $lr \
        --tokenselector_bin_path "/home/weiliu/student/xhm/LLaVA/checkpoints/llava-v1.5-7b-mix665k_shuffled-k32-lambda0.01-sn50-2e-2-mode0-linearseed42/checkpoint-400/token_selector.bin" \
        --run_name "llava-token-selector-v1.5-7b-$data_json-k$k-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --logging_dir "/mnt/ssd/weiliu/student/xhm/LLaVA_logs/llava-token-selector-$data_json-k$k-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --output_dir "/home/weiliu/student/xhm/LLaVA/checkpoints/llava-v1.5-7b-$data_json-k$k-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --weight_decay 0. \
        --warmup_ratio 0.01 \
        --local_rank 0 \
        --save_steps 10 \
        --eval_steps 20 \
        --model_name_or_path "liuhaotian/llava-v1.5-7b" \
        --vision_tower "openai/clip-vit-large-patch14-336" \
        --pretrain_mm_mlp_adapter "/home/weiliu/student/xhm/LLaVA/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin" \
        --version "v1" \
        --data_path $data_path \
        --image_folder $image_folder \
        --mm_projector_type "mlp2x_gelu" \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio "pad" \
        --group_by_modality_length True \
        --bf16 True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_total_limit 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to "wandb"