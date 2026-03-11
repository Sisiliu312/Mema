#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-13b-align0.15-lr5e-5/llava-v1.5-13b \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --image-folder /dataset/eval/pope/val2014 \
    --answers-file /dataset/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /dataset/eval/pope/coco \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --result-file /dataset/eval/pope/answers/llava-v1.5-7b.jsonl

python scripts/v1_5/eval/print.py 