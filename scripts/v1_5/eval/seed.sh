#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2 nohup bash scripts/v1_5/eval/seed.sh > /mnt/hdd/weiliu/student/xhm/LLaVA_logs/mix-shuffled-2e-2-sn50-k64-step50-seed-eval.out &
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
sample_num=50
lr=2e-2
# 0:gumbel top k 1:bernoulli 2:multinomial
mode=0
k=64
lambda_r=0.01
data_json=mix665k_shuffled
selector_mode=linear
steps=50

CKPT="llava-v1.5-7b-$data_json-lambda$lambda_r-sn$sample_num-$lr-mode$mode"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT-checkpoint-$steps-$k/${CHUNKS}_${IDX}.jsonl \
        --tokenselector-bin-path ./checkpoints/$CKPT/checkpoint-$steps/token_selector.bin \
        --tokenselector-k $k \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT-checkpoint-$steps-$k/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT-checkpoint-$steps-$k/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT-checkpoint-$steps-$k.jsonl

