#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/v1_5/eval/mmvet.sh > /mnt/hdd/weiliu/student/xhm/LLaVA_logs/mix-shuffled-2e-2-sn50-k64-step50-mmvet-eval.out &
sample_num=50
lr=2e-2
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
k=64
lambda_r=0.01
data_json=mix665k_shuffled
selector_mode=mlp2x_relu
steps=50

CKPT="llava-v1.5-7b-$data_json-lambda$lambda_r-sn$sample_num-$lr-mode$mode"

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$CKPT-checkpoint-$steps-$k.jsonl \
    --temperature 0 \
    --tokenselector-bin-path ./checkpoints/$CKPT/checkpoint-$steps/token_selector.bin \
    --tokenselector-k $k \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT-checkpoint-$steps-$k.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$CKPT-checkpoint-$steps-$k.json

