#!/bin/bash
cd /code/LLaVA-DSU
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_science \
    --model-path /checkpoints/llava-v1.5-DSU-layermix/llava-v1.5-7b \
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

