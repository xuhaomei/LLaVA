#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/v1_5/eval/pope.sh > /mnt/hdd/weiliu/student/xhm/LLaVA_logs/mix-shuffled-2e-2-sn50-k128-step50-pope-eval.out &
sample_num=50
lr=2e-2
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
k=128
lambda_r=0.01
data_json=mix665k_shuffled
selector_mode=linear
steps=50

CKPT="llava-v1.5-7b-$data_json-lambda$lambda_r-sn$sample_num-$lr-mode$mode"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT-checkpoint-$steps.jsonl \
    --tokenselector-bin-path ./checkpoints/$CKPT/checkpoint-$steps/token_selector.bin \
    --tokenselector-k $k \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT-checkpoint-$steps.jsonl
