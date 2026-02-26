#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /checkpoints/llava-v1.5-scvm-answerloss/llava-v1.5-7b \
    --question-file /dataset/eval/mmbench/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
