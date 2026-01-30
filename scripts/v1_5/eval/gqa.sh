#!/bin/bash
#CUDA_VISIBLE_DEVICES=0,1,2 nohup bash scripts/v1_5/eval/gqa.sh > /mnt/hdd/weiliu/student/xhm/LLaVA_logs/mix-shuffled-2e-2-sn50-k128-step50-gqa-eval.out &
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

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
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT-checkpoint-$steps-$k/${CHUNKS}_${IDX}.jsonl \
        --tokenselector-bin-path ./checkpoints/$CKPT/checkpoint-$steps/token_selector.bin \
        --tokenselector-k $k \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT-checkpoint-$steps-$k/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT-checkpoint-$steps-$k/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
