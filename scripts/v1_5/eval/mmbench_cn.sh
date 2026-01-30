#!/bin/bash
sample_num=50
lr=2e-2
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
k=64
lambda_r=0.01
data_json=mix665k_shuffled
selector_mode=mlp2x_relu
steps=400

CKPT="llava-v1.5-7b-$data_json-lambda$lambda_r-sn$sample_num-$lr-mode$mode"

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$CKPT.jsonl \
    --tokenselector-bin-path ./checkpoints/$CKPT/checkpoint-$steps/token_selector.bin \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment $CKPT
