#!/bin/bash
# nohup bash script_qwen.sh > /mnt/hdd/weiliu/student/xhm/Qwen_logs/visionselector_dataset%50-2e-3-sn100-k0.1-seed42.out &
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="06124ddc00b85edf3d9ff4ca8e5643372d1234f7"
export WANDB_PROJECT="LLaVA-Token-Selector"
export WANDB_DIR="/mnt/hdd/weiliu/student/xhm/LLaVA_logs/llava-token-selector/wandb"
export HF_HOME="/mnt/hdd/weiliu/student/xhm/cache"

sample_num=100
lr=2e-3
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
keep_ratio=0.1
lambda_r=0.01
data_json=visionselector_dataset%50
selector_mode=linear
other_cfg=seed42

python /home/weiliu/student/xhm/LLaVA/llava/train/train_token_selector_qwen.py \
        --mode $mode \
        --keep_ratio $keep_ratio \
        --sample_num $sample_num \
        --alpha_pg_loss 1.0 \
        --lambda_r $lambda_r \
        --learning_rate $lr \
        --run_name "qwen-token-selector-v2.5-7b-$data_json-k$keep_ratio-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --logging_dir "/mnt/hdd/weiliu/student/xhm/Qwen_logs/qwen-token-selector-$data_json-k$keep_ratio-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --output_dir "/home/weiliu/student/xhm/LLaVA/checkpoints/qwen-v2.5-7b-$data_json-k$keep_ratio-lambda$lambda_r-sn$sample_num-$lr-mode$mode-$selector_mode$other_cfg" \
        --weight_decay 0. \
        --warmup_ratio 0.01 \
        --local_rank 0 \
        --save_steps 10 \
        --eval_steps 10 \
        --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
        --data_flatten True \
        --data_path "/mnt/hdd/weiliu/student/xhm/data/cambrian_737k/Cambrian737k_shuffled.json" \
        --dataset_use "chartqa%50,coco%5,ocr_vqa%50" \
        --image_folder "/mnt/hdd/weiliu/student/xhm/data/cambrian_737k" \
        --bf16 True \
        --tune_selector True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_total_limit 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to "wandb"