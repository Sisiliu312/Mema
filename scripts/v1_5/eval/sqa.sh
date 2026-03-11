#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_science \
    --model-path /checkpoints/llava-v1.5-13b-align0.15-lr5e-5/llava-v1.5-13b \
    --question-file /dataset/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /dataset/eval/scienceqa/test \
    --answers-file /dataset/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /dataset/eval/scienceqa \
    --result-file /dataset/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /dataset/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /dataset/eval/scienceqa/answers/llava-v1.5-13b_result.json

