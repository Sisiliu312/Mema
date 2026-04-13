#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-scvm-answerloss/llava-v1.5-7b \
    --question-file /dataset/eval/test/test.jsonl \
    --image-folder /dataset/eval/test/images \
    --answers-file /dataset/eval/test/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

