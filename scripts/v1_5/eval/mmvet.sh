#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa \
    --model-path /checkpoints/llava-v1.5-13b-align0.15-lr5e-5/llava-v1.5-13b \
    --question-file /dataset/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /dataset/eval/mm-vet/images \
    --answers-file /dataset/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /dataset/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst /dataset/eval/mm-vet/results/llava-v1.5-7b.json

